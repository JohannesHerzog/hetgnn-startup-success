"""Preprocessing: load startup_nodes.csv, extract features, encode descriptions, build targets."""
import os
import numpy as np
import pandas as pd


# Columns that contain target labels — never used as input features
TARGET_COLUMNS = {
    "new_funding_round", "new_acquired", "new_ipo", "acq_ipo_funding",
    "acquired", "ipo", "closed", "operating", "future_status",
    "dc_status", "acquirer",
}


def load_startups(config):
    path = os.path.join(config["paths"]["graph_dir"], config["paths"]["startup_file"])
    print(f"Loading {path}...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    return df


def encode_descriptions(df, config):
    """Return a DataFrame of PCA-reduced description embeddings (N × embedding_dim).

    Embeddings are cached to disk so the sentence-transformer only runs once.
    The PCA model is saved to `embeddings_dir/org_pca_64d.pkl` — predict.py
    loads the same file when encoding a user-provided description at inference.
    """
    from sentence_transformers import SentenceTransformer
    import joblib

    emb_dir = config["paths"]["embeddings_dir"]
    dim = config["features"]["description_embedding_dim"]
    pca_path = os.path.join(emb_dir, "org_pca_64d.pkl")
    cache_path = os.path.join(emb_dir, "org_raw_embeddings.npy")

    os.makedirs(emb_dir, exist_ok=True)

    # --- Raw embeddings (sentence-transformer, 384-dim) ---
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"  Loading cached raw embeddings from {cache_path}")
        raw_full = np.load(cache_path)
        # Cache covers the full dataset; select only the rows we need
        if "_orig_idx" in df.columns and len(raw_full) > len(df):
            raw = raw_full[df["_orig_idx"].values]
            print(f"  Selected {len(raw):,} rows from cache")
        else:
            raw = raw_full
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  Encoding descriptions with sentence-transformer (device={device})...")
        model_st = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        descriptions = df["description"].fillna("").tolist()
        raw = model_st.encode(descriptions, batch_size=512, show_progress_bar=True).astype(np.float32)
        np.save(cache_path, raw)
        print(f"  Cached raw embeddings to {cache_path}")
        raw_full = raw

    # --- PCA reduction (fit on full cache for best quality) ---
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        print(f"  Loaded PCA model from {pca_path}")
    else:
        from sklearn.decomposition import PCA
        fit_data = raw_full if len(raw_full) > len(raw) else raw
        print(f"  Fitting PCA ({dim} components) on {len(fit_data):,} samples...")
        pca = PCA(n_components=dim, random_state=42)
        pca.fit(fit_data)
        joblib.dump(pca, pca_path)
        print(f"  Saved PCA model to {pca_path}")

    reduced = pca.transform(raw).astype(np.float32)
    cols = [f"desc_emb_{i}" for i in range(dim)]
    return pd.DataFrame(reduced, columns=cols, index=df.index)


def extract_features(df, config):
    """Select whitelisted tabular features and append description embeddings."""
    whitelist = config["features"]["whitelist"]
    reference_date = pd.to_datetime(config["training"]["reference_date"])

    available = [f for f in whitelist if f in df.columns]
    missing = set(whitelist) - set(available)
    if missing:
        print(f"  Warning: whitelist features not found in CSV: {sorted(missing)}")

    feat_df = df[available].copy()

    # Convert date columns to days-since-reference (numeric)
    date_cols = {"last_funding_on", "founded_on"} & set(feat_df.columns)
    for col in date_cols:
        feat_df[col] = pd.to_datetime(feat_df[col], errors="coerce")
        feat_df[col] = (feat_df[col] - reference_date).dt.days

    # Label-encode remaining string columns (e.g. city) and save mappings to disk.
    cat_mappings = {}
    for col in feat_df.select_dtypes(include=["object", "string"]).columns:
        cat = feat_df[col].astype("category")
        cat_mappings[col] = {name: int(code) for code, name in enumerate(cat.cat.categories)}
        feat_df[col] = cat.cat.codes.replace(-1, np.nan)

    if cat_mappings:
        import json
        emb_dir = config["paths"]["embeddings_dir"]
        os.makedirs(emb_dir, exist_ok=True)
        mapping_path = os.path.join(emb_dir, "cat_mappings.json")
        with open(mapping_path, "w") as f:
            json.dump(cat_mappings, f, ensure_ascii=False, indent=2)
        print(f"  Saved category mappings → {mapping_path}")

    feat_df = feat_df.astype("float32")

    # Description embeddings
    if "description" in df.columns:
        emb_df = encode_descriptions(df, config)
        feat_df = pd.concat([feat_df, emb_df], axis=1)
    else:
        dim = config["features"]["description_embedding_dim"]
        for i in range(dim):
            feat_df[f"desc_emb_{i}"] = np.float32(0.0)

    return feat_df


def build_targets(df):
    """Build Momentum and Liquidity targets with masks.

    Momentum  — predicts next funding round.
                Masked: startups that already had a liquidity event (exit)
                are excluded (their "no new round" is not a failure signal).

    Liquidity — predicts IPO or acquisition.
                Masked: only "mature" startups are included to avoid
                labelling young companies as failures simply because they
                haven't exited yet.
    """
    y_mom = df["new_funding_round"].fillna(0).astype(int).values
    y_liq = ((df.get("new_acquired", 0) == 1) | (df.get("new_ipo", 0) == 1)).astype(int).values

    # Momentum mask: exclude exits
    mask_mom = (1 - y_liq).astype(int)

    # Liquidity mask: only mature startups
    snapshot = pd.Timestamp("2023-06-01")
    founded = pd.to_datetime(df["founded_on"], errors="coerce")
    founded = founded.where(founded >= pd.Timestamp("1900-01-01"), other=pd.NaT)
    age_days = (snapshot - founded).dt.days.fillna(0).values
    funding  = df["total_funding_usd"].fillna(0).values
    rounds   = df["num_funding_rounds"].fillna(0).values

    is_mature = (
        (age_days  > 5 * 365) |
        (funding   > 1_000_000) |
        (rounds    >= 3)
    ).astype(int)
    mask_liq = is_mature

    n_liq_mature = mask_liq.sum()
    n_liq_exits  = y_liq[mask_liq == 1].sum()
    print(f"  Liquidity — mature: {n_liq_mature:,}, exits: {n_liq_exits:,} ({n_liq_exits/max(n_liq_mature,1):.2%})")

    return {
        "momentum":  {"y": y_mom, "mask": mask_mom},
        "liquidity": {"y": y_liq, "mask": mask_liq},
    }


def train_val_test_split(n, config, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * config["training"]["train_ratio"])
    n_val   = int(n * config["training"]["val_ratio"])
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def preprocess(config, filter_fn=None, exclude_uuids=None):
    """Full pipeline: load → filter → features → targets → splits.

    Args:
        filter_fn:      Optional callable (df) -> df to filter startups before training.
        exclude_uuids:  Optional set of startup_uuid strings to exclude (used for
                        test-set isolation across models G / M / K).

    Returns a dict with:
        X             – float32 array (N × F)
        feature_names – list of F column names
        targets       – {"momentum": {y, mask}, "liquidity": {y, mask}}
        splits        – {"train": idx, "val": idx, "test": idx}
        test_uuids    – set of startup_uuid strings in the test split
    """
    df = load_startups(config)
    df["_orig_idx"] = np.arange(len(df))  # preserve row positions before filtering

    if filter_fn is not None:
        df = filter_fn(df)
        print(f"  After filter: {len(df):,} rows")

    if exclude_uuids:
        n_before = len(df)
        df = df[~df["startup_uuid"].isin(exclude_uuids)].reset_index(drop=True)
        print(f"  Excluded {n_before - len(df):,} UUIDs (test sets of other models)")

    print("Extracting features...")
    feat_df = extract_features(df, config)

    print("Building targets...")
    targets = build_targets(df)

    train_idx, val_idx, test_idx = train_val_test_split(len(df), config, seed=config["seed"])

    X = np.nan_to_num(feat_df.values, nan=0.0).astype(np.float32)

    print(f"  Shape: {X.shape} | Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    uuids      = df["startup_uuid"].values if "startup_uuid" in df.columns else np.empty(len(df), dtype=object)
    test_uuids = set(uuids[test_idx].tolist())

    return {
        "X":             X,
        "feature_names": feat_df.columns.tolist(),
        "targets":       targets,
        "splits":        {"train": train_idx, "val": val_idx, "test": test_idx},
        "uuids":         uuids,
        "test_uuids":    test_uuids,
    }
