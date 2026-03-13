"""Utility functions for configuration loading, data I/O, and helper computations."""
import yaml
import pandas as pd
import os
import copy
import numpy as np
import ast


def deep_merge_dict(base, override):
    """Recursively merge two dictionaries, preserving all base values"""
    merged = copy.deepcopy(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Both are dicts, merge recursively
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            # Override the value
            merged[key] = value

    return merged


def _resolve_config_value(key, raw_value, base_config):
    """Infer the correct type for a CLI value by looking up the key in the base config.

    Handles booleans, ints, floats, and lists. Falls back to string if the key
    is not found in the config (allows ad-hoc overrides).
    """
    # Navigate base_config to find the existing value for type inference
    keys = key.split(".")
    node = base_config
    for k in keys:
        if isinstance(node, dict) and k in node:
            node = node[k]
        else:
            node = None
            break

    # Boolean detection (works regardless of config lookup)
    if raw_value.lower() in ("true", "false"):
        return raw_value.lower() == "true"

    # Type inference from config value
    if node is not None:
        if isinstance(node, bool):
            return raw_value.lower() == "true"
        if isinstance(node, int):
            return int(raw_value)
        if isinstance(node, float):
            return float(raw_value)
        if isinstance(node, list):
            # Single value → wrap in list; actual multi-value lists are
            # handled by the caller collecting consecutive non-flag tokens.
            return [raw_value]

    # Fallback: try numeric conversion, otherwise keep as string
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


def parse_config_overrides(argv, base_config):
    """Parse arbitrary ``--dot.notation value`` arguments into a nested config dict.

    Supports multi-value arguments (``--key a b c`` → list) and WandB stringified
    lists (``["a","b"]`` passed as a single token).  Special flags like
    ``--preprocess-only`` and ``--output-dir`` are extracted separately.

    Returns ``(config_updates, special_args)`` where *special_args* is a dict
    with keys ``preprocess_only``, ``output_dir``, ``experiment_id`` (or ``None``).
    """
    config_updates: dict = {}
    special = {"preprocess_only": None, "output_dir": None, "experiment_id": None}

    # Map of special (non-dot-notation) flags to their special_args key
    _special_flags = {
        "--preprocess-only": "preprocess_only",
        "--output-dir": "output_dir",
        "--experiment_id": "experiment_id",
    }

    i = 0
    while i < len(argv):
        token = argv[i]

        if not token.startswith("--"):
            i += 1
            continue

        key = token[2:]  # strip leading --

        # Handle special flags
        if token in _special_flags:
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                special[_special_flags[token]] = argv[i + 1]
                i += 2
            else:
                i += 1
            continue

        # Collect all values until the next flag
        values = []
        i += 1
        while i < len(argv) and not argv[i].startswith("--"):
            values.append(argv[i])
            i += 1

        if not values:
            continue

        # Parse WandB stringified lists (e.g. a single token like "['investor','city']")
        if len(values) == 1:
            s = values[0].strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        values = parsed
                except (ValueError, SyntaxError):
                    pass

        # Convert dot-notation key to nested dict
        config_key = key.replace("-", "_")
        if len(values) == 1:
            resolved = _resolve_config_value(config_key, str(values[0]), base_config)
        else:
            resolved = values  # multi-value → list

        keys = config_key.split(".")
        node = config_updates
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = resolved

    return config_updates, special


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()


def load_csv_with_index(path):
    df = pd.read_csv(path, low_memory=False)
    df.reset_index(drop=True, inplace=True)
    return df


def success_ratio(df, config):
    """
    Calculates success ratio(s) based on prediction mode and config.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration dict with:
            - data_processing.target_mode
            - data_processing.multi_column (for multi-class)
            - data_processing.binary_column (for binary)

    Returns:
        float or dict: success ratio(s)
    """
    mode = config["data_processing"]["target_mode"]
    multi_col = config["data_processing"].get("multi_column")
    binary_col = config["data_processing"].get("binary_column")

    def compute_binary_ratio():
        if not binary_col:
            raise ValueError(
                "Binary column must be specified for binary or multi_task mode."
            )
        return df[binary_col].eq(1).mean()

    def compute_multi_class_ratios():
        if not multi_col:
            raise ValueError(
                "Multi-class column must be specified for multi or multi_task mode."
            )
        return df[multi_col].value_counts(normalize=True).sort_index().to_dict()

    if mode == "binary_prediction":
        return compute_binary_ratio()

    elif mode == "multi_prediction":
        return compute_multi_class_ratios()

    elif mode == "multi_task":
        return {
            "binary": compute_binary_ratio(),
            "multi_class": compute_multi_class_ratios(),
        }

    elif mode == "multi_label":
        # Compute ratios for Fund, Acq, IPO
        ratios = {}
        
        cols_config = config["data_processing"].get("multi_label", {}).get("columns", {})
        col_fund = cols_config.get("funding", "new_funding_round")
        col_acq = cols_config.get("acquisition", "acquired")
        col_ipo = cols_config.get("ipo", "ipo")
        
        # Funding
        if col_fund in df.columns:
            ratios["funding"] = df[col_fund].fillna(0).astype(int).mean()
        
        # Acquisition
        if col_acq in df.columns:
            ratios["acquisition"] = df[col_acq].fillna(0).astype(int).mean()
            
        # IPO
        if col_ipo in df.columns:
            ratios["ipo"] = df[col_ipo].fillna(0).astype(int).mean()
            
        return ratios

    elif mode == "masked_multi_task":
        # Compute ratios for Momentum (Next Funding) and Liquidity (Exit)
        ratios = {}
        
        # Get dynamic column names from config
        cols_config = config["data_processing"].get("multi_label", {}).get("columns", {})
        col_mom = cols_config.get("funding", "new_funding_round")
        col_acq = cols_config.get("acquisition", "new_acquired")
        col_ipo = cols_config.get("ipo", "new_ipo")
        
        # Momentum (Growth)
        if col_mom in df.columns:
             ratios["momentum"] = df[col_mom].fillna(0).astype(int).mean()
             
        # Liquidity (Exit) - Combined Acq/IPO (STRICT GATING)
        if col_acq in df.columns and col_ipo in df.columns:
            # Liquidity event = Acquired OR IPO
            is_liq = (df[col_acq].fillna(0).astype(int) == 1) | (df[col_ipo].fillna(0).astype(int) == 1)
            
            # Use helper to get mask
            is_mature = get_maturity_mask(df, config)
            
            if is_mature is not None:
                # Calculate ratio strictly within Mature set
                n_mature = is_mature.sum()
                if n_mature > 0:
                    ratios["liquidity"] = is_liq[is_mature == 1].sum() / n_mature
                else:
                    ratios["liquidity"] = 0.0
            else:
                 # Fallback if masking fails or not configured
                 ratios["liquidity"] = is_liq.astype(int).mean()
            
        return ratios


    else:
        raise ValueError(f"Unsupported target_mode: {mode}")


def add_t_acq_ipo_seriesA_target(startup_df):
    """
    Add t_acq_ipo_seriesA target column based on acquired, ipo, and seriesA columns.
    Returns 1 if startup was acquired OR went IPO OR reached Series A, otherwise 0.
    """
    if 't_acq_ipo_seriesA' not in startup_df.columns:
        # Ensure required columns exist, defaulting to 0 if missing
        if 'acquired' not in startup_df.columns:
            startup_df['acquired'] = 0
        if 'ipo' not in startup_df.columns:
            startup_df['ipo'] = 0  
        if 'seriesA' not in startup_df.columns:
            startup_df['seriesA'] = 0
            
        # Create the target column
        startup_df['t_acq_ipo_seriesA'] = (
            (startup_df['acquired'] == 1) | 
            (startup_df['ipo'] == 1) | 
            (startup_df['seriesA'] == 1)
        ).astype(int)
        
    return startup_df


def get_maturity_mask(df, config):
    """
    Calculate boolean mask for mature startups based on configuration.
    Returns Series or None if columns invalid.
    """
    if "strict_gating" not in config["data_processing"]:
        return None
        
    gating_cfg = config["data_processing"]["strict_gating"]
    if not gating_cfg.get("enabled", False):
        return None

    has_funding = "total_funding_usd" in df.columns
    has_rounds = "num_funding_rounds" in df.columns
    has_founded = "founded_on" in df.columns
    
    if not (has_funding and has_rounds and has_founded):
        return None
        
    # Get thresholds from config
    late_stage_fund_thresh = gating_cfg.get("late_stage_funding_threshold", 15000000)
    emp_count_thresh = gating_cfg.get("employee_count_threshold", 3)
    comp_age_thresh = gating_cfg.get("compounder_age_threshold", 5)
    comp_fund_thresh = gating_cfg.get("compounder_funding_threshold", 3000000)
    
    snapshot_date = pd.Timestamp("2023-06-01")
    founded_dates = pd.to_datetime(df["founded_on"], errors='coerce')
    
    # Filter valid dates
    mask_valid_dates = (founded_dates > pd.Timestamp("1970-01-01")) & (founded_dates < pd.Timestamp("2030-01-01"))
    
    age_in_days = pd.Series(0, index=df.index)
    age_in_days[mask_valid_dates] = (snapshot_date - founded_dates[mask_valid_dates]).dt.days
    age_in_days = age_in_days.fillna(0)
    age_years = age_in_days / 365.0
    
    # 1. Big Dogs (Stage & Money)
    # New logic using last_funding_stage feature (Series B = 5, PE = 14)
    if "last_funding_stage" in df.columns:
        last_stage_val = pd.to_numeric(df["last_funding_stage"], errors='coerce').fillna(0)
        has_late_stage = (last_stage_val >= 5) # Series B or later
    else:
        # Fallback to column checks if feature missing (legacy data)
        has_series_b = (df["series_b_round"].fillna(0) > 0) if "series_b_round" in df.columns else False
        has_series_c = (df["series_c_round"].fillna(0) > 0) if "series_c_round" in df.columns else False
        has_venture = (df["venture_round"].fillna(0) > 0) if "venture_round" in df.columns else False
        has_late_stage = has_series_b | has_series_c | has_venture

    mask_big_dogs = (
        (df["total_funding_usd"].fillna(0) >= late_stage_fund_thresh) |
        has_late_stage
    )
    
    # 2. Scale Ops (Employees)
    emp_map = {
        '1-10': 1, '11-50': 2, '51-100': 3, '101-250': 4,
        '251-500': 5, '501-1000': 6, '1001-5000': 7, 
        '5001-10000': 8, '10000+': 9, 'unknown': 0
    }
    
    if "employee_count" in df.columns:
        # map values, fill NaN/unknown with 0
        emp_codes = df["employee_count"].map(emp_map).fillna(0).astype(int)
        mask_scale_ops = (emp_codes >= emp_count_thresh)
    else:
        mask_scale_ops = False
    
    # 3. Efficient Compounders (Age + Moderate Funding)
    mask_compounders = (
        (age_years >= comp_age_thresh) & 
        (df["total_funding_usd"].fillna(0) >= comp_fund_thresh)
    )
    
    is_mature = mask_big_dogs | mask_scale_ops | mask_compounders
    
    # Log Statistics
    print(f"\nMaturity Mask Statistics (Advanced Gating):")
    print(f"   - Big Dogs (> ${late_stage_fund_thresh/1e6:.0f}M or Ser B+): {mask_big_dogs.sum()} startups")
    if isinstance(mask_scale_ops, pd.Series) or isinstance(mask_scale_ops, np.ndarray):
         print(f"   - Scale Ops (Emp >= {emp_count_thresh}): {mask_scale_ops.sum()} startups")
    else:
         print(f"   - Scale Ops (Emp >= {emp_count_thresh}): N/A (Col Missing)")
         
    print(f"   - Compounders (Age>={comp_age_thresh} & $>{comp_fund_thresh/1e6:.1f}M): {mask_compounders.sum()} startups")
    print(f"   - Total Mature (Any Cond): {is_mature.sum()} / {len(df)} ({is_mature.sum()/len(df):.1%})")
    print(f"   - Immature (Filtered): {len(df) - is_mature.sum()} ({1 - is_mature.sum()/len(df):.1%})\n")
    
    return is_mature

def load_data(
    startups_filename,
    investors_filename,
    founders_filename,
    cities_filename,
    university_filename,
    sectors_filename,
    startup_investor_filename,
    startup_city_filename,
    startup_founder_filename,
    startup_sector_filename,
    founder_university_filename,
    investor_city_filename,
    university_city_filename,
    investor_sector_filename,
    founder_investor_employment_filename=None,
    founder_coworking_filename=None,
    founder_investor_identity_filename=None,
    startup_descriptively_similar_filename=None,
    founder_descriptively_similar_filename=None,
    founder_co_study_filename=None,
    founder_board_filename=None,
    founder_startup_director_filename=None,
    founder_investor_director_filename=None,
):
    data_path = (
        f"{config['paths']['graph_dir']}"
    )

    startup_df = load_csv_with_index(os.path.join(data_path, startups_filename))
    investor_df = load_csv_with_index(os.path.join(data_path, investors_filename))
    founder_df = load_csv_with_index(os.path.join(data_path, founders_filename))
    city_df = load_csv_with_index(os.path.join(data_path, cities_filename))
    university_df = load_csv_with_index(os.path.join(data_path, university_filename))
    sector_df = load_csv_with_index(os.path.join(data_path, sectors_filename))
    startup_investor_df = load_csv_with_index(
        os.path.join(data_path, startup_investor_filename)
    )
    startup_city_df = load_csv_with_index(
        os.path.join(data_path, startup_city_filename)
    )
    startup_founder_df = load_csv_with_index(
        os.path.join(data_path, startup_founder_filename)
    )
    startup_sector_df = load_csv_with_index(
        os.path.join(data_path, startup_sector_filename)
    )
    founder_university_df = load_csv_with_index(
        os.path.join(data_path, founder_university_filename)
    )
    investor_city_df = load_csv_with_index(
        os.path.join(data_path, investor_city_filename)
    )
    university_city_df = load_csv_with_index(
        os.path.join(data_path, university_city_filename)
    )
    
    # Load optional files
    def load_optional(filename):
        if filename and os.path.exists(os.path.join(data_path, filename)):
            return load_csv_with_index(os.path.join(data_path, filename))
        return pd.DataFrame()

    founder_investor_employment_df = load_optional(founder_investor_employment_filename)
    founder_coworking_df = load_optional(founder_coworking_filename)
    founder_investor_identity_df = load_optional(founder_investor_identity_filename)
    startup_descriptively_similar_df = load_optional(startup_descriptively_similar_filename)
    founder_descriptively_similar_df = load_optional(founder_descriptively_similar_filename)
    founder_co_study_df = load_optional(founder_co_study_filename)
    founder_board_df = load_optional(founder_board_filename)
    founder_startup_director_df = load_optional(founder_startup_director_filename)
    founder_investor_director_df = load_optional(founder_investor_director_filename)

    # Add t_acq_ipo_seriesA target column
    startup_df = add_t_acq_ipo_seriesA_target(startup_df)

    print(f"The success ratio is: {success_ratio(startup_df, config)}")

    return (
        startup_df,
        investor_df,
        founder_df,
        city_df,
        university_df,
        sector_df,
        startup_investor_df,
        startup_city_df,
        startup_founder_df,
        startup_sector_df,
        founder_university_df,
        investor_city_df,
        university_city_df,
        founder_investor_employment_df,
        founder_coworking_df,
        founder_investor_identity_df,
        startup_descriptively_similar_df,
        founder_descriptively_similar_df,
        founder_co_study_df,
        founder_board_df,
        founder_startup_director_df,
        founder_investor_director_df,
    )
