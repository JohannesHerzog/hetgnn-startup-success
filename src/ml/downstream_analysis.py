"""Downstream analysis simulating VC portfolio strategies to evaluate model predictions via ROI and precision."""
import argparse

import yaml
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

# Add project root to path to allow imports if running as script
import sys
sys.path.append(os.getcwd())
from src.data_engineering.aux_pipeline import convert_to_continent
from src.ml.utils import get_maturity_mask

_THESIS_RCPARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _apply_thesis_style(ax):
    """Remove top/right spines and apply thesis grid style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)


class DownstreamAnalyzer:
    """
    Performs downstream analysis on model predictions, including ROI, Sector, Round Type, Age, and Geography performance.
    """
    def __init__(self, config: Dict):
        """
        Initialize the Downstream Analyzer.
        
        Args:
            config: Configuration dictionary containing paths.
        """
        self.config = config
        self.data_dir = config["paths"]["crunchbase_dir"]
        self.graph_dir = config["paths"]["graph_dir"]
        self.output_dir = config.get("output_dir", "outputs")
        
        print(f"Initializing Downstream Analyzer...")
        
        # Load Data
        self._load_data()
    
        # Dilution Rates for Valuation Estimation (Radicle Model)
        self.DILUTION_RATES = {
            'angel': 0.10,
            'pre_seed': 0.08,
            'seed': 0.10,
            'series_a': 0.20,
            'series_b': 0.16,
            'series_c': 0.13,
            'series_d': 0.12,
            'series_e': 0.10,
            'series_f': 0.09,
            'series_g': 0.08,
            'series_h': 0.06,
            'series_i': 0.05,
            'private_equity': 0.15, # Assumed
            'debt_financing': 0.05, # Assumed low equity impact
            'convertible_note': 0.10, # Assumed similar to seed
            'grant': 0.0, # Non-dilutive
            'post_ipo_equity': 0.05 # Assumed
        }
        
        # Step-Up Multiple for Unrealized Gains (Paper Markups)
        # If a startup raises a new round, we assume our previous stake appreciates by this factor.
        self.STEP_UP_MULTIPLE = 3.0
        
        # Fixed Ticket Size for Simulation
        self.TICKET_SIZE = 1_000_000.0
        
        # Undisclosed Exit Multiple (Capital Returned)
        self.UNDISCLOSED_EXIT_MULTIPLE = 1.0

        # Tiered Step-Ups for Paper Gains (Valuation follows Stage)
        self.TIERED_STEP_UPS = {
            'angel': 3.0,
            'pre_seed': 3.0,
            'seed': 3.0,
            'series_a': 2.5,
            'series_b': 2.0,
            # Late Stage / Growth / Default
            'default': 1.3
        }

        # Historical Funding Multiple (Estimate)
        self.HISTORICAL_FUNDING_MULTIPLE = 1.5
        
        # Benchmark Investors
        self.BENCHMARK_INVESTORS = {
            'a16z': 'ce91bad7-b6d8-e56e-0f45-4763c6c5ca29',
            'Sequoia': '0c867fde-2b9a-df10-fdb9-66b74f355f91',
            'YC': '73633ee4-ea65-2967-6c5d-9b5fec7d2d5e',
            'Benchmark': 'fe2d1e8b-f607-3c9f-fad7-98fb8412f77e',
            'Accel': 'b08efc27-da40-505a-6f9d-c9e14247bf36'
        }



        # Strategy Definitions for Downstream Analysis
        self.STAGE_STRATEGIES = {
            'Angel/Pre-Seed': {'angel', 'pre_seed', 'equity_crowdfunding', 'convertible_note', 'grant'},
            'Seed': {'seed'},
            'Series A': {'series_a'},
            'Series B': {'series_b'},
            'Series C+': {'series_c', 'series_d', 'series_e', 'series_f', 'series_g', 'series_h', 'series_i', 'series_j', 'private_equity'},
            'Early Stage (Cumulative)': {'pre_seed', 'seed', 'angel', 'series_a', 'equity_crowdfunding', 'convertible_note'},
            'Growth (Cumulative)': {'series_b', 'series_c', 'series_d', 'series_e', 'series_f', 'series_g', 'series_h', 'series_i', 'series_j', 'private_equity'}
        }

        self.CONTINENT_STRATEGIES = {
            'North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania'
        }
        
        self.FUNDING_SOURCE_STRATEGIES = {
            'Private Backed': {'private'},
            'Public (Gov) Backed': {'public'},
            'Hybrid Backed': {'hybrid'}
        }



    def _load_data(self):
        """
        Load necessary CSVs for analysis.
        
        Files Loaded:
        - Funding Rounds (2023 & 2025): For ROI calculation and target window definition.
        - Organizations (2023): For sector, geography, and founding year features.
        - IPOs & Acquisitions (2025): For exit value calculation (Ground Truth).
        - Investments (2023): For benchmarking against investor portfolios.
        """
        # 1. Funding Rounds (for ROI)
        funding_path = os.path.join(self.data_dir, "2025", "funding_rounds.csv")
        if os.path.exists(funding_path):
            usecols = ['uuid', 'org_uuid', 'announced_on', 'raised_amount_usd', 'post_money_valuation_usd', 'investment_type']
            self.funding_df = pd.read_csv(funding_path, usecols=usecols)
            self.funding_df.rename(columns={'uuid': 'funding_round_uuid'}, inplace=True)
            self.funding_df['announced_on'] = pd.to_datetime(self.funding_df['announced_on'], errors='coerce')
            
            # Determine End Date from 2025 snapshot
            self.roi_end_date = self.funding_df['announced_on'].max()
            print(f"   Determined ROI End Date (from 2025 snapshot): {self.roi_end_date.date()}")
            
            # Filter for target window (Placeholder, will refine after loading 2023 start date)
            start_date = pd.Timestamp('2023-01-01')
            end_date = self.roi_end_date
            self.target_window_df = self.funding_df[
                (self.funding_df['announced_on'] >= start_date) & 
                (self.funding_df['announced_on'] <= end_date)
            ].copy()
            print(f"   Loaded {len(self.target_window_df)} funding rounds (2023-2025)")
            
            # Compute Last Stage Map (Pre-2023)
            # Filter for rounds before 2023
            pre_2023_df = self.funding_df[self.funding_df['announced_on'] < start_date].copy()
            if not pre_2023_df.empty:
                # Sort by date
                pre_2023_df = pre_2023_df.sort_values('announced_on')
                # Group by org and take last investment_type
                self.last_stage_map = pre_2023_df.groupby('org_uuid')['investment_type'].last().to_dict()
                print(f"   Computed last stage for {len(self.last_stage_map)} organizations (Pre-2023)")
            else:
                self.last_stage_map = {}
        else:
            print(f"WARNING: Funding rounds file not found: {funding_path}")
            self.target_window_df = None
            self.last_stage_map = {}
            self.roi_end_date = '2025-12-31' # Fallback

        # 1.1 Load 2023 Funding Rounds (for Start Date)
        funding_2023_path = os.path.join(self.data_dir, "2023", "funding_rounds.csv")
        if os.path.exists(funding_2023_path):
            usecols = ['announced_on']
            funding_2023_df = pd.read_csv(funding_2023_path, usecols=usecols)
            funding_2023_df['announced_on'] = pd.to_datetime(funding_2023_df['announced_on'], errors='coerce')
            self.roi_start_date = funding_2023_df['announced_on'].max()
            print(f"   Determined ROI Start Date (from 2023 snapshot): {self.roi_start_date.date()}")
        else:
            print(f"WARNING: 2023 Funding rounds file not found: {funding_2023_path}")
            self.roi_start_date = pd.Timestamp('2023-01-01') # Fallback

        # Update target window with dynamic start date
        if self.target_window_df is not None:
                self.target_window_df = self.funding_df[
                (self.funding_df['announced_on'] >= self.roi_start_date) & 
                (self.funding_df['announced_on'] <= self.roi_end_date)
            ].copy()
                print(f"   Refined target window to {len(self.target_window_df)} rounds ({self.roi_start_date.date()} - {self.roi_end_date.date()})")

        # 2. Organizations (for Sector, Round Type, Funding Source)
        orgs_path = os.path.join(self.data_dir, "2023", "organizations.csv") # Use 2023 for "at prediction time" features
        
        # Check if we need to load from startup_nodes.csv for investor_type
        # Since investor_type is a calculated feature not in raw organizations.csv
        nodes_path = os.path.join(self.graph_dir, "startup_nodes.csv")
        
        if os.path.exists(orgs_path):
            # Load category_list, num_funding_rounds, total_funding_usd
            # Also try to load investor_type if it exists, otherwise we merge it from nodes
            usecols = ['uuid', 'name', 'category_list', 'num_funding_rounds', 'total_funding_usd', 'country_code', 'founded_on']
            self.orgs_df = pd.read_csv(orgs_path, usecols=usecols)
            self.orgs_df.rename(columns={'uuid': 'org_uuid'}, inplace=True)
            self.orgs_df['founded_on'] = pd.to_datetime(self.orgs_df['founded_on'], errors='coerce')
            print(f"   Loaded {len(self.orgs_df)} organizations (2023)")
            
            # Merge investor_type from startup_nodes.csv if available
            if os.path.exists(nodes_path):
                try:
                    nodes_df = pd.read_csv(nodes_path, usecols=['startup_uuid', 'investor_type'])
                    nodes_df.rename(columns={'startup_uuid': 'org_uuid'}, inplace=True)
                    # Merge left on orgs_df
                    self.orgs_df = self.orgs_df.merge(nodes_df, on='org_uuid', how='left')
                    print(f"   Merged investor_type from startup_nodes.csv for {len(nodes_df)} startups")
                except Exception as e:
                    print(f"   WARNING: Failed to load investor_type from startup_nodes.csv: {e}")
        else:
            print(f"WARNING: Organizations file not found: {orgs_path}")
            self.orgs_df = None
            
        # 3. IPOs (Target)
        ipos_path = os.path.join(self.data_dir, "2025", "ipos.csv") # Use 2025 for future labels
        if os.path.exists(ipos_path):
            self.ipos_df = pd.read_csv(ipos_path, usecols=['org_uuid', 'valuation_price_usd', 'went_public_on'])
            self.ipos_df['went_public_on'] = pd.to_datetime(self.ipos_df['went_public_on'], errors='coerce')
            print(f"   Loaded {len(self.ipos_df)} IPOs (2025)")
        else:
            print(f"WARNING: IPOs file not found: {ipos_path}")
            self.ipos_df = None

        # 4. Acquisitions (Target)
        acq_path = os.path.join(self.data_dir, "2025", "acquisitions.csv") # Use 2025 for future labels
        if os.path.exists(acq_path):
            self.acq_df = pd.read_csv(acq_path, usecols=['acquiree_uuid', 'price_usd', 'acquired_on'])
            self.acq_df.rename(columns={'acquiree_uuid': 'org_uuid', 'acquired_on': 'announced_on'}, inplace=True)
            self.acq_df['announced_on'] = pd.to_datetime(self.acq_df['announced_on'], errors='coerce')
            print(f"   Loaded {len(self.acq_df)} acquisitions (2025)")
        else:
            print(f"WARNING: Acquisitions file not found: {acq_path}")
            self.acq_df = None

        # 5. Investments (for Benchmarking)
        investments_path = os.path.join(self.data_dir, "2023", "investments.csv")
        if os.path.exists(investments_path):
            usecols = ['investor_uuid', 'funding_round_uuid']
            self.investments_df = pd.read_csv(investments_path, usecols=usecols)
            print(f"   Loaded {len(self.investments_df)} investments (2023)")
        else:
            print(f"WARNING: Investments file not found: {investments_path}")
            self.investments_df = None


    def perform_downstream_analysis(self, predictions: List[Tuple[str, any, any]]):
        """
        Run all downstream analyses.
        Supports both single-task (float scores) and multi-label (dict scores).
        """
        if not predictions:
            print("WARNING: No predictions provided for analysis.")
            return

        # 1. Detect Mode
        sample_score = predictions[0][1]
        task_queue = []

        if isinstance(sample_score, dict):
            # Check for Masked Multi-Task (New Strategy-Aware Logic)
            if 'mom' in sample_score and 'liq' in sample_score:
                print("\nDetected Masked Multi-Task Predictions (Momentum & Liquidity).")
                
                # 1. Venture Fund (Standard / Growth)
                # Universe: All, Signal: Momentum
                task_queue.append(('mom', 'Venture Fund', False)) 
                
                # 2. Liquidity Fund (Pure / Exit)
                # Universe: Mature, Signal: Liquidity
                task_queue.append(('liq', 'Liquidity Fund', True))

                # 3. Momentum Fund (Mature Growth)
                # Universe: Mature, Signal: Momentum
                #task_queue.append(('mom', 'Momentum Fund (Mature)', True))

                # 4. Balanced Fund (Composite)
                # Universe: Mature, Signal: Average(Mom, Liq)
                #task_queue.append(('balanced', 'Balanced Fund (Composite)', True))
            
            # Check for Multi-Label (Legacy/Alternative)
            elif 'fund' in sample_score or 'acq' in sample_score or 'ipo' in sample_score:
                print("\nDetected Multi-Label Predictions. Running Combined Analysis...")
                # Add Combined Task
                if 'fund' in sample_score and 'acq' in sample_score and 'ipo' in sample_score:
                    task_queue.append(('combined', 'Combined', False))
                else:
                     print("WARNING: Missing keys for combined analysis (fund, acq, ipo required)")
            
            else:
                 print(f"WARNING: Unknown dictionary output keys: {sample_score.keys()}")
                 
        else:
            task_queue.append(('default', 'Standard', False))

        # 2. Run Analysis Loop
        for task_item in task_queue:
            # Unpack task item
            if len(task_item) == 3:
                task_key, task_name, use_mature_filter = task_item
            else:
                task_key, task_name = task_item
                use_mature_filter = False
            
            if task_key == 'default':
                task_preds = predictions
                suffix = ""
                title_suffix = ""
            else:
                # Extract specific task predictions
                # Handle potential missing keys gracefully (though they should exist)
                task_preds = []
                
                if task_key == 'combined':
                     # Retrieve weights from config or use defaults
                     # Config structure: self.config["data_processing"]["multi_label"]["combined_metric_weights"]
                     # Safety check for config path
                     dp_config = self.config.get("data_processing", {})
                     ml_config = dp_config.get("multi_label", {}) if isinstance(dp_config, dict) else {}
                     weights_config = ml_config.get("combined_metric_weights", {}) if isinstance(ml_config, dict) else {}
                     
                     w_fund = float(weights_config.get("funding", 0.2))
                     w_acq = float(weights_config.get("acquisition", 0.3))
                     w_ipo = float(weights_config.get("ipo", 0.5))
                     
                     for u, s, l in predictions:
                         if 'fund' in s and 'acq' in s and 'ipo' in s:
                             # Weighted Score
                             combined_score = (w_fund * s['fund']) + (w_acq * s['acq']) + (w_ipo * s['ipo'])
                             # Combined Label (Any success is success? Or align with specific goal?)
                             # For ROI analysis, the specific label (FUNDED/EXIT/FAIL) depends on ground truth lookup.
                             # But 'label' arg is used to flag 'FUNDED' (paper gain) if not exited.
                             # Let's say if predicted to fund, we count it.
                             combined_label = max(l.get('fund', 0), l.get('acq', 0), l.get('ipo', 0))
                             task_preds.append((u, combined_score, combined_label))
                else:
                    # Specific Task (or Balanced Composite)
                    for u, s, l in predictions:
                        if task_key == 'balanced':
                             # Composite Score: (Mom + Liq) / 2
                             if 'mom' in s and 'liq' in s:
                                 score = (s['mom'] + s['liq']) / 2.0
                                 # Label? Doesn't matter for ROI (ground truth), but for plotting AUC:
                                 # Use max label logic or just placeholder
                                 label = max(l.get('mom', 0), l.get('liq', 0))
                                 task_preds.append((u, score, label))
                        elif task_key in s and task_key in l:
                            task_preds.append((u, s[task_key], l[task_key]))
                
                suffix = f"_{task_key}"
                title_suffix = f" ({task_name})"
            if not task_preds:
                print(f"   WARNING: No predictions found for task {task_name}")
                continue

            # Convert to DataFrame
            try:
                pred_df = pd.DataFrame(task_preds, columns=['org_uuid', 'score', 'gt_label'])
            except Exception as e:
                print(f"   WARNING: Failed to convert predictions to DataFrame: {e}")
                continue
            
            # Merge with Metadata
            full_df = pred_df
            if self.orgs_df is not None:
                full_df = pred_df.merge(self.orgs_df, on='org_uuid', how='left')
                
            # 1. Filter if needed (Liquidity Strategy)
            if use_mature_filter:
                print(f"   Applying Maturity Filter (for {task_name})...")
                task_preds = self._filter_mature_startups(task_preds)
                if not task_preds:
                    print("   WARNING: No mature startups found in predictions. Skipping ROI.")
                    continue
            
            # 2. ROI Analysis
            roi_metrics = self.calculate_roi(
                task_preds, 
                filename_suffix=suffix, 
                title_suffix=title_suffix
            )
            
            # 1.1 Investor Benchmark
            benchmark_metrics = self.analyze_investor_benchmark()
            if benchmark_metrics and task_key == 'default': # Only print full benchmark once to avoid clutter
                print(f"\nBenchmark Results:")
                for name, metrics in benchmark_metrics.items():
                    print(f"   - {name}: Precision={metrics['precision']:.1%}, ROI={metrics['roi']:.1%}, k={metrics['k']}")

            # 1.2 Portfolio Comparison
            print(f"\nPortfolio Comparison (Model{title_suffix} vs Investors)")
            for investor_name in ['Sequoia', 'a16z']:
                if investor_name in self.BENCHMARK_INVESTORS:
                    self.compare_portfolios(task_preds, investor_name)
                
            # 1.3 Stage-Based Strategy Analysis
            strategy_results = {}
            for strategy_name, stages in self.STAGE_STRATEGIES.items():
                print(f"   Analyzing Strategy (Stage): {strategy_name}...") # One print? No, loop.
                strat_preds = self.analyze_portfolio_by_stage(task_preds, strategy_name, stages, filename_suffix=suffix, verbose=False, plot_charts=False)
                if strat_preds:
                    strategy_results[strategy_name] = strat_preds

            # 1.4 Comparative Precision Plot (Strategies)
            if strategy_results:
                # Add Standard Model for comparison
                strategy_results['Standard (All Stages)'] = task_preds
                self._plot_comparative_precision_at_k(strategy_results, benchmark_metrics, filename_suffix=suffix)

            # 1.5 Geography-Based Strategy Analysis
            if 'country_code' in full_df.columns:
                print(f"\n   Analyzing Geography Strategies...")
                # Create Content Map on the fly - Ensure continent column exists
                if 'continent' not in full_df.columns:
                     full_df['continent'] = full_df['country_code'].apply(convert_to_continent)
                
                # Get available continents
                available_continents = full_df['continent'].dropna().unique()
                
                geo_results = {}
                print(f"\n   Analyzing Geography Strategies ({len(available_continents)} Continents)...")
                for continent in available_continents:
                    if continent == 'Unknown': continue
                    
                    cont_preds = self.analyze_portfolio_by_continent(task_preds, continent, full_df, filename_suffix=suffix, verbose=False, plot_charts=False)
                    if cont_preds:
                        geo_results[continent] = cont_preds

                
            # 3. Round Type Analysis
            if 'num_funding_rounds' in full_df.columns:
                self.analyze_funding_stage(full_df, top_k=None, filename_suffix=suffix)
                
            # 4. Founding Year Analysis
            if 'founded_on' in full_df.columns:
                self.analyze_founding_year(full_df, top_k=None, filename_suffix=suffix)
                
                # 1.6 Comparative Precision Plot (Geography)
                if geo_results:
                    # Add Standard for comparison
                    geo_results['Global (All Regions)'] = task_preds
                    self._plot_comparative_precision_at_k(geo_results, benchmark_metrics=None, filename_suffix=f"{suffix}_geography")

            # 1.7 Funding Source Strategy Analysis
            if 'investor_type' in full_df.columns:
                print(f"\n   Analyzing Funding Source Strategies...")
                source_results = {}
                for strategy_name, allowed_types in self.FUNDING_SOURCE_STRATEGIES.items():
                    try:
                        source_preds = self.analyze_portfolio_by_funding_source(task_preds, strategy_name, allowed_types, full_df, filename_suffix=suffix, verbose=False, plot_charts=False)
                        if source_preds:
                            source_results[strategy_name] = source_preds
                    except Exception as e:
                        print(f"   WARNING: Failed {strategy_name}: {e}")
                
                # Comparative Plot
                if source_results:
                   source_results['All Sources'] = task_preds
                   self._plot_comparative_precision_at_k(source_results, benchmark_metrics=None, filename_suffix=f"{suffix}_funding_source")

            # 2. Sector Analysis
            if 'category_list' in full_df.columns:
                self.analyze_sectors(full_df, top_k=None, filename_suffix=suffix)
                
            # 3. Round Type Analysis
            if 'num_funding_rounds' in full_df.columns:
                self.analyze_funding_stage(full_df, top_k=None, filename_suffix=suffix)
                
            # 4. Founding Year Analysis
            if 'founded_on' in full_df.columns:
                self.analyze_founding_year(full_df, top_k=None, filename_suffix=suffix)
                
            # 5. Legacy Geography Analysis (Country/Continent Distribution)
            if 'country_code' in full_df.columns:
                self.analyze_geography(full_df, top_k=None, filename_suffix=suffix)
                self.analyze_continents(full_df, top_k=None, filename_suffix=suffix)


    def _estimate_valuation(self, raised_amount: float, round_type: str) -> float:
        """
        Estimate Post-Money Valuation using Dilution Model.
        Formula: Post-Valuation = Raised Amount / Dilution Rate
        """
        if pd.isna(raised_amount) or raised_amount <= 0:
            return 0.0
            
        # Normalize round_type
        if pd.isna(round_type):
            round_type = 'seed' # Default
        else:
            round_type = str(round_type).lower().replace(' ', '_').replace('-', '_')
            
        dilution = self.DILUTION_RATES.get(round_type, 0.15) # Default to 15% if unknown
        
        if dilution <= 0: # Handle non-dilutive grants
            return raised_amount # Conservative: value = cash raised
            
        return raised_amount / dilution

    def _plot_net_profit_curve(self, ranks: np.ndarray, costs: np.ndarray, values: np.ndarray, labels: np.ndarray, filename_suffix: str = ""):
        """
        Generate Net Profit Curve (Cumulative Value - Cumulative Cost) for ALL predictions.
        Also includes Recall @ k on secondary axis.
        """
        if len(ranks) == 0:
            return

        profits = np.nan_to_num(values - costs)
        total_positives = np.sum(labels)
        recall_at_k = np.cumsum(labels) / total_positives if total_positives > 0 else np.zeros_like(labels)

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

                line1 = ax1.plot(ranks, profits, label='Net Profit', color='#1f77b4', linewidth=1.5)
                ax1.set_xlabel('Portfolio Rank (Number of Startups)')
                ax1.set_ylabel('Net Profit (USD)', color='#1f77b4')
                ax1.tick_params(axis='y', labelcolor='#1f77b4')

                peak_idx = np.argmax(profits)
                peak_rank = ranks[peak_idx]
                peak_profit = profits[peak_idx]
                ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

                def currency_formatter(x, pos):
                    if abs(x) >= 1e9: return f'${x/1e9:.1f}B'
                    if abs(x) >= 1e6: return f'${x/1e6:.0f}M'
                    return f'${x:.0f}'
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))

                ax2 = ax1.twinx()
                line2 = ax2.plot(ranks, recall_at_k, label='Recall @ k', color='#ff7f0e', linewidth=1.5, linestyle='--')
                ax2.set_ylabel('Recall / Precision', color='0.4')
                ax2.tick_params(axis='y', labelcolor='0.4')
                ax2.set_ylim(0, 1.05)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

                precision_at_k = np.cumsum(labels) / ranks
                line3 = ax2.plot(ranks, precision_at_k, label='Precision @ k', color='#2ca02c', linewidth=1.5, linestyle='--')

                ax1.scatter([peak_rank], [peak_profit], color='#d62728', s=60, zorder=5, label='Peak Profit')
                ax1.annotate(f'${peak_profit/1e6:.0f}M (k={peak_rank})',
                             (peak_rank, peak_profit),
                             xytext=(peak_rank + len(ranks)*0.05, peak_profit),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),
                             fontsize=8)

                lines = line1 + line2 + line3 + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=6, label='Peak Profit')]
                labs = [l.get_label() for l in lines]
                ax1.legend(lines, labs, loc='center right', fontsize=8)

                ax1.spines["top"].set_visible(False)
                ax2.spines["top"].set_visible(False)
                ax1.grid(True, alpha=0.2, linewidth=0.5)
                fig.tight_layout()

                filename = f'roi_net_profit{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/roi_net_profit{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Net Profit Curve: {e}")

    def _plot_precision_at_k(self, ranks: np.ndarray, labels: np.ndarray, filename_suffix: str = "", benchmark_metrics: Optional[Dict] = None, global_base_rate: Optional[float] = None):
        """
        Generate Precision @ k Curve (Cumulative Precision vs Rank).
        """
        if len(ranks) == 0:
            return

        # Calculate Cumulative Precision
        cum_successes = np.cumsum(labels)
        precision_at_k = cum_successes / ranks

        # Base Rate: use global if provided, else compute from labels
        base_rate = global_base_rate if global_base_rate is not None else np.mean(labels)

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                ax.plot(ranks, precision_at_k, color='#9467bd', linewidth=1.5, label='Precision @ k')

                # Base Rate Line
                ax.axhline(y=base_rate, color='black', linestyle='--', linewidth=1, alpha=0.7)
                ax.text(ranks[-1] * 0.85, base_rate + 0.02, f'Base Rate ({base_rate:.1%})',
                        fontsize=8, ha='right', va='bottom', color='0.3')

                # Benchmark
                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        bm_k = metrics['k']
                        bm_prec = metrics['precision']
                        color = bm_colors[i % len(bm_colors)]
                        marker = bm_markers[i % len(bm_markers)]
                        ax.scatter([bm_k], [bm_prec], color=color, marker=marker, s=80, zorder=10,
                                   edgecolors='white', linewidths=0.5, label=f"{name} ({bm_prec:.1%})")

                ax.set_xlabel('Portfolio Rank (k)')
                ax.set_ylabel('Precision')
                ax.legend(fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                _apply_thesis_style(ax)
                fig.tight_layout()

                filename = f'precision_at_k{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/precision_at_k{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Precision @ k Curve: {e}")

    def _plot_comparative_precision_at_k(self, strategy_predictions: Dict[str, List[Tuple[str, float, int]]], 
                                         benchmark_metrics: Optional[Dict] = None,
                                         filename_suffix: str = ""):
        """
        Generate Comparative Precision @ k Curve for multiple strategies.
        Args:
            strategy_predictions: Dict mapping Strategy Name -> List of Predictions
        """
        if not strategy_predictions:
            return

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))

                # Color palette
                n = len(strategy_predictions)
                palette = plt.cm.tab10(np.linspace(0, 0.9, min(n, 10))) if n <= 10 else plt.cm.tab20(np.linspace(0, 0.95, n))

                for i, (strat_name, preds) in enumerate(strategy_predictions.items()):
                    if not preds: continue
                    sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)
                    labs = np.array([p[2] for p in sorted_preds])
                    if len(labs) == 0: continue
                    rnk = np.arange(1, len(labs) + 1)
                    prec = np.cumsum(labs) / rnk
                    ax.plot(rnk, prec, label=strat_name, color=palette[i], linewidth=1.5, alpha=0.85)

                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        bm_k = metrics['k']
                        bm_prec = metrics['precision']
                        ax.scatter([bm_k], [bm_prec], color=bm_colors[i % len(bm_colors)],
                                   marker=bm_markers[i % len(bm_markers)], s=80, zorder=10,
                                   edgecolors='white', linewidths=0.5, label=f"{name} ({bm_prec:.1%})")

                ax.set_xlabel('Portfolio Rank (k)')
                ax.set_ylabel('Precision')
                ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.9)
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

                max_k = 1000
                if benchmark_metrics:
                    max_bm_k = max([m['k'] for m in benchmark_metrics.values()])
                    max_k = max(1000, max_bm_k + 100)
                ax.set_xlim(0, max_k)
                _apply_thesis_style(ax)
                fig.tight_layout()

                filename = f'precision_at_k_comparative_strategies{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/precision_at_k_comparative_strategies{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Comparative Precision Strategies: {e}")

    def _plot_precision_vs_roi(self, ranks: np.ndarray, labels: np.ndarray, costs: np.ndarray, values: np.ndarray, filename_suffix: str = "", benchmark_metrics: Optional[Dict] = None):
        """
        Generate Precision vs ROI Scatter Plot.
        """
        if len(ranks) == 0:
            return

        cum_successes = np.cumsum(labels)
        precision_at_k = cum_successes / ranks
        cum_costs = np.cumsum(costs)
        cum_values = np.cumsum(values)
        roi_at_k = (cum_values - cum_costs) / cum_costs

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.5))

                sc = ax.scatter(precision_at_k, roi_at_k, c=ranks, cmap='viridis', s=15, alpha=0.6)
                cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label('Portfolio Size (k)', fontsize=9)
                ax.plot(precision_at_k, roi_at_k, color='gray', linewidth=0.5, alpha=0.4)

                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        ax.scatter([metrics['precision']], [metrics['roi']],
                                   color=bm_colors[i % len(bm_colors)],
                                   marker=bm_markers[i % len(bm_markers)],
                                   s=100, zorder=10, edgecolors='white', linewidths=0.5, label=name)

                k_to_annotate = [10, 50, 100, 500, 1000, len(ranks)]
                for k in k_to_annotate:
                    if k <= len(ranks):
                        idx = k - 1
                        ax.annotate(f'k={k}', (precision_at_k[idx], roi_at_k[idx]),
                                    xytext=(5, 5), textcoords='offset points', fontsize=7)
                        ax.scatter([precision_at_k[idx]], [roi_at_k[idx]], color='#d62728', s=30, zorder=5)

                ax.set_xlabel('Precision @ k')
                ax.set_ylabel('ROI @ k (Multiple)')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
                if benchmark_metrics:
                    ax.legend(fontsize=8)
                _apply_thesis_style(ax)
                fig.tight_layout()

                filename = f'precision_vs_roi{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/precision_vs_roi{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Precision vs ROI: {e}")

    def _plot_k_vs_precision_and_roi(self, ranks: np.ndarray, labels: np.ndarray, costs: np.ndarray, values: np.ndarray, filename_suffix: str = "", benchmark_metrics: Optional[Dict] = None):
        """
        Generate Dual-Axis Plot: k vs Precision and ROI.
        """
        if len(ranks) == 0:
            return

        cum_successes = np.cumsum(labels)
        precision_at_k = cum_successes / ranks
        cum_costs = np.cumsum(costs)
        cum_values = np.cumsum(values)
        roi_at_k = (cum_values - cum_costs) / cum_costs

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

                color_prec = '#9467bd'
                ax1.set_xlabel('Portfolio Size (k)')
                ax1.set_ylabel('Precision', color=color_prec)
                ax1.plot(ranks, precision_at_k, color=color_prec, linewidth=1.5, label='Precision')
                ax1.tick_params(axis='y', labelcolor=color_prec)
                ax1.set_ylim(0, 1.05)
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax1.spines["top"].set_visible(False)
                ax1.grid(True, alpha=0.2, linewidth=0.5)

                ax2 = ax1.twinx()
                color_roi = '#2ca02c'
                ax2.set_ylabel('ROI (Multiple)', color=color_roi)
                ax2.plot(ranks, roi_at_k, color=color_roi, linewidth=1.5, linestyle='--', label='ROI')
                ax2.tick_params(axis='y', labelcolor=color_roi)
                ax2.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.3)
                ax2.spines["top"].set_visible(False)

                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#d62728', '#e377c2', '#8c564b', '#bcbd22']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        bm_k = metrics['k']
                        c = bm_colors[i % len(bm_colors)]
                        m = bm_markers[i % len(bm_markers)]
                        ax1.scatter([bm_k], [metrics['precision']], color=c, marker=m, s=80, zorder=10,
                                    edgecolors='white', linewidths=0.5, label=f"{name}")

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
                fig.tight_layout()

                filename = f'k_vs_precision_and_roi{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/k_vs_precision_and_roi{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot k vs Precision & ROI: {e}")

    def _plot_k_vs_precision_and_recall(self, ranks: np.ndarray, labels: np.ndarray, filename_suffix: str = "", benchmark_metrics: Optional[Dict] = None):
        """
        Generate Dual-Axis Plot: k vs Precision and Recall.
        X-Axis: k (Rank)
        Left Y-Axis: Precision
        Right Y-Axis: Recall
        """
        if len(ranks) == 0:
            return

        total_positives = labels.sum()
        if total_positives == 0:
            return

        cum_successes = np.cumsum(labels)
        precision_at_k = cum_successes / ranks
        recall_at_k = cum_successes / total_positives

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

                color_prec = '#9467bd'
                ax1.set_xlabel('Portfolio Size (k)')
                ax1.set_ylabel('Precision', color=color_prec)
                ax1.plot(ranks, precision_at_k, color=color_prec, linewidth=1.5, label='Precision')
                ax1.tick_params(axis='y', labelcolor=color_prec)
                ax1.set_ylim(0, 1.05)
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax1.spines["top"].set_visible(False)
                ax1.grid(True, alpha=0.2, linewidth=0.5)

                ax2 = ax1.twinx()
                color_rec = '#1f77b4'
                ax2.set_ylabel('Recall', color=color_rec)
                ax2.plot(ranks, recall_at_k, color=color_rec, linewidth=1.5, linestyle='--', label='Recall')
                ax2.tick_params(axis='y', labelcolor=color_rec)
                ax2.set_ylim(0, 1.05)
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                ax2.spines["top"].set_visible(False)

                if benchmark_metrics:
                    bm_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#e377c2', '#8c564b']
                    bm_markers = ['D', 's', '^', 'v', '*']
                    for i, (name, metrics) in enumerate(benchmark_metrics.items()):
                        bm_k = metrics['k']
                        bm_prec = metrics['precision']
                        ax1.scatter([bm_k], [bm_prec], color=bm_colors[i % len(bm_colors)],
                                    marker=bm_markers[i % len(bm_markers)], s=80, zorder=10,
                                    edgecolors='white', linewidths=0.5, label=f"{name}")

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
                fig.tight_layout()

                filename = f'k_vs_precision_and_recall{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                try:
                    wandb.log({f"analysis/k_vs_precision_and_recall{filename_suffix}": wandb.Image(plot_path)})
                except Exception:
                    pass

        except Exception as e:
            print(f"Failed to plot k vs Precision & Recall: {e}")

    def _filter_mature_startups(self, predictions: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
        """
        Filter predictions to only include Mature Startups using get_maturity_mask logic.
        """
        if self.orgs_df is None:
            print("   WARNING: Organization metadata missing. Cannot filter for maturity.")
            return predictions

        # Calculate mask on orgs_df
        mature_mask = get_maturity_mask(self.orgs_df, self.config)
        
        if mature_mask is None:
            print("   WARNING: Maturity mask generation failed (check config). Using all startups.")
            return predictions
            
        # Get Set of Mature UUIDs
        # orgs_df needs to be aligned with the mask
        mature_uuids = set(self.orgs_df[mature_mask == 1]['org_uuid'])
        
        filtered_preds = []
        for uuid, score, label in predictions:
            if uuid in mature_uuids:
                filtered_preds.append((uuid, score, label))
                
        print(f"   Filtered: {len(predictions)} -> {len(filtered_preds)} mature startups ({len(filtered_preds)/len(predictions):.1%})")
        
        return filtered_preds

    def analyze_portfolio_by_continent(self, predictions: List[Tuple[str, float, int]], 
                                       continent_name: str, 
                                       full_df: pd.DataFrame, 
                                       filename_suffix: str = "",
                                       verbose: bool = True,
                                       plot_charts: bool = True) -> List[Tuple[str, float, int]]:
        """
        Analyze performance when constrained to a specific continent.
        """
        # Create a set of UUIDs belonging to this continent
        continent_uuids = set(full_df[full_df['continent'] == continent_name]['org_uuid'])
        
        constrained_preds = []
        for uuid, score, label in predictions:
            if uuid in continent_uuids:
                constrained_preds.append((uuid, score, label))
                
        if len(constrained_preds) == 0:
            if verbose:
                print(f"      Scanning... 0 startups found.")
            return []
            
        if verbose:
            print(f"      Scanning... {len(constrained_preds)} startups found.")

        # Run ROI Analysis
        clean_name = continent_name.replace(" ", "_").replace("/", "_")
        self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{clean_name}", title_suffix=f" ({continent_name})", verbose=verbose, plot_charts=plot_charts)
        
        return constrained_preds

    
    def analyze_portfolio_by_stage(self, predictions: List[Tuple[str, float, int]], 
                                   strategy_name: str, 
                                   target_stages: List[str], 
                                   filename_suffix: str = "",
                                   verbose: bool = True,
                                   plot_charts: bool = True) -> List[Tuple[str, float, int]]:
        """
        Analyze performance when constrained to specific investment stages.
        """
        constrained_preds = []
        for uuid, score, label in predictions:
            # Check stage at prediction time (Pre-2023)
            stage = self.last_stage_map.get(uuid, 'unknown')
            stage_norm = str(stage).lower().replace(' ', '_').replace('-', '_')
            
            if stage_norm in target_stages:
                constrained_preds.append((uuid, score, label))
                
        if len(constrained_preds) == 0:
            if verbose:
                print(f"   WARNING: No startups found for {strategy_name}.")
            return []
            
        if verbose:
            print(f"   {strategy_name}: {len(constrained_preds)} startups found.")

        # Run ROI Analysis
        clean_name = strategy_name.replace(" ", "_").replace("/", "_").replace("+", "Plus")
        self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{clean_name}", title_suffix=f" ({strategy_name})", verbose=verbose, plot_charts=plot_charts)
        
        return constrained_preds


    def analyze_portfolio_by_funding_source(self, predictions: List[Tuple[str, float, int]], 
                                            strategy_name: str, 
                                            allowed_types: List[str], 
                                            full_df: pd.DataFrame, 
                                            filename_suffix: str = "",
                                            verbose: bool = True,
                                            plot_charts: bool = True) -> List[Tuple[str, float, int]]:
        """
        Analyze performance when constrained to specific funding sources (investor types).
        """
        # Create a set of UUIDs belonging to this strategy
        # Filter full_df where investor_type is in allowed_types
        # Handle potential NaNs
        
        valid_uuids = set(full_df[full_df['investor_type'].isin(allowed_types)]['org_uuid'])
        
        constrained_preds = []
        for uuid, score, label in predictions:
            if uuid in valid_uuids:
                constrained_preds.append((uuid, score, label))
                
        if len(constrained_preds) == 0:
            if verbose:
                print(f"      Scanning... 0 startups found.")
            return []
             
        if verbose:
            print(f"      Scanning... {len(constrained_preds)} startups found.")
        
        # Run ROI Analysis
        clean_name = strategy_name.replace(" ", "_").replace("(", "").replace(")", "")
        self.calculate_roi(constrained_preds, filename_suffix=f"{filename_suffix}_{clean_name}", title_suffix=f" ({strategy_name})", verbose=verbose, plot_charts=plot_charts)
        
        return constrained_preds


    def analyze_investor_benchmark(self) -> Dict[str, Dict]:
        """
        Analyze the performance of benchmark investors during the target period.
        Period: Jan 2022 - June 2023 (18 months prior to prediction date).
        """
        if self.investments_df is None or self.funding_df is None:
            return {}
            
        results = {}
        
        # Pre-calculate common maps to avoid re-doing it for every investor
        # Funding after June 2023
        future_funding = self.funding_df[self.funding_df['announced_on'] > self.roi_start_date]
        funded_orgs = set(future_funding['org_uuid'].unique())
        
        # IPOs
        ipo_orgs = set()
        if self.ipos_df is not None:
            ipo_orgs = set(self.ipos_df[self.ipos_df['went_public_on'] >= self.roi_start_date]['org_uuid'])
            
        # Acquisitions
        acq_orgs = set()
        if self.acq_df is not None:
            acq_orgs = set(self.acq_df[self.acq_df['announced_on'] >= self.roi_start_date]['org_uuid'])
            
        # Maps for Value
        future_funding_map = future_funding.groupby('org_uuid').agg({
            'raised_amount_usd': 'sum',
            'post_money_valuation_usd': 'max',
            'investment_type': 'last'
        }).to_dict('index')
        
        ipo_map = {}
        if self.ipos_df is not None:
            ipo_map = self.ipos_df[self.ipos_df['went_public_on'] >= self.roi_start_date].set_index('org_uuid')['valuation_price_usd'].to_dict()
            
        acq_map = {}
        if self.acq_df is not None:
            acq_map = self.acq_df[self.acq_df['announced_on'] >= self.roi_start_date].set_index('org_uuid')['price_usd'].to_dict()
            
        total_funding_map = {}
        if self.orgs_df is not None:
            total_funding_map = self.orgs_df.set_index('org_uuid')['total_funding_usd'].to_dict()

        successful_uuids = funded_orgs.union(ipo_orgs).union(acq_orgs)
        
        for investor_name, investor_uuid in self.BENCHMARK_INVESTORS.items():
            # 1. Filter Investments by Investor
            inv_investments = self.investments_df[self.investments_df['investor_uuid'] == investor_uuid].copy()
            
            if inv_investments.empty:
                print(f"   WARNING: No investments found for {investor_name}.")
                continue
                
            merged_df = inv_investments.merge(self.funding_df, on='funding_round_uuid', how='inner')
            
            # 3. Filter by Date Range (Jan 2022 - June 2023)
            start_date = pd.Timestamp('2022-01-01')
            end_date = self.roi_start_date # June 2023
            
            benchmark_df = merged_df[
                (merged_df['announced_on'] >= start_date) & 
                (merged_df['announced_on'] <= end_date)
            ].copy()
            
            if benchmark_df.empty:
                print(f"   WARNING: No {investor_name} investments found in benchmark period.")
                continue
                
            unique_orgs = benchmark_df['org_uuid'].unique()
            k = len(unique_orgs)
            
            # Calculate Precision
            successes = 0
            for uuid in unique_orgs:
                if uuid in successful_uuids:
                    successes += 1
            precision = successes / k if k > 0 else 0
            
            # Calculate ROI
            total_cost = k * self.TICKET_SIZE
            total_value = 0.0
            
            for uuid in unique_orgs:
                cost = self.TICKET_SIZE
                
                # Entry Val Logic
                data = future_funding_map.get(uuid, {})
                raised = data.get('raised_amount_usd', 0)
                if pd.isna(raised): raised = 0
                
                round_type = data.get('investment_type', None)
                if pd.isna(round_type):
                    dilution_rate = 0.10
                else:
                    round_type_str = str(round_type).lower().replace(' ', '_').replace('-', '_')
                    dilution_rate = self.DILUTION_RATES.get(round_type_str, 0.15)
                    
                current_post_val = raised / dilution_rate if dilution_rate > 0 else raised
                current_pre_val = current_post_val - raised
                entry_val = current_pre_val / self.STEP_UP_MULTIPLE if self.STEP_UP_MULTIPLE > 0 else current_pre_val
                
                if raised == 0 or entry_val <= 100_000:
                     total_funding = total_funding_map.get(uuid, 0)
                     if pd.notna(total_funding) and total_funding > 0:
                         entry_val = total_funding * self.HISTORICAL_FUNDING_MULTIPLE
                     else:
                         entry_val = 5_000_000
                         
                ownership = self.TICKET_SIZE / entry_val
                if ownership > 0.20: ownership = 0.20
                
                # Exit Value
                exit_val = 0.0
                status = 'FAIL'
                
                if uuid in ipo_map:
                    exit_val = ipo_map[uuid] * ownership
                    status = 'EXIT'
                elif uuid in acq_map:
                    exit_val = acq_map[uuid] * ownership
                    status = 'EXIT'
                    
                if status == 'EXIT' and (pd.isna(exit_val) or exit_val == 0):
                     total_funding = total_funding_map.get(uuid, 0)
                     if pd.isna(total_funding): total_funding = 0
                     if total_funding > 0:
                         exit_val = (total_funding * self.UNDISCLOSED_EXIT_MULTIPLE) * ownership
                     else:
                         exit_val = cost
                
                if status != 'EXIT':
                    if uuid in funded_orgs:
                        exit_val = cost * self.STEP_UP_MULTIPLE # Paper Gain
                        status = 'FUNDED'
                    else:
                        status = 'FAIL'
                
                total_value += exit_val
                
            roi = (total_value - total_cost) / total_cost if total_cost > 0 else 0
            
            results[investor_name] = {
            'precision': precision,
            'roi': roi,
            'k': k
        }
            
        return results


    def compare_portfolios(self, predictions: List[Tuple[str, float, int]], investor_name: str):
        """
        Compare the model's top predictions against a specific investor's portfolio.
        Analyzes Sector, Stage, and Missed Winners.
        """
        if investor_name not in self.BENCHMARK_INVESTORS:
            return
        if self.investments_df is None or self.funding_df is None:
            return

        investor_uuid = self.BENCHMARK_INVESTORS[investor_name]
        print(f"\nComparing vs {investor_name}...")
        
        # 1. Get Investor Portfolio (Jan 2022 - June 2023)
        inv_investments = self.investments_df[self.investments_df['investor_uuid'] == investor_uuid]
        merged_df = inv_investments.merge(self.funding_df, on='funding_round_uuid', how='inner')
        
        start_date = pd.Timestamp('2022-01-01')
        end_date = self.roi_start_date
        
        benchmark_df = merged_df[
            (merged_df['announced_on'] >= start_date) & 
            (merged_df['announced_on'] <= end_date)
        ].copy()
        
        investor_uuids = set(benchmark_df['org_uuid'].unique())
        k = len(investor_uuids)
        
        if k == 0:
            print(f"   No investments found for {investor_name} in target period.")
            return

        # 2. Get Model Portfolio (Top k)
        # Sort predictions by score
        sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
        model_top_k = sorted_preds[:k]
        model_uuids = set([p[0] for p in model_top_k])
        
        # 3. Overlap
        overlap = investor_uuids.intersection(model_uuids)
        jaccard = len(overlap) / len(investor_uuids.union(model_uuids))
        
        # 3.1 Test Set Coverage (New)
        all_test_uuids = set([p[0] for p in predictions])
        investor_in_test_set = investor_uuids.intersection(all_test_uuids)
        
        
        if len(investor_in_test_set) > 0:
            # Analyze Rank distribution of these bets
            ranks = []
            scores = []
            for uuid in investor_in_test_set:
                 for i, p in enumerate(sorted_preds):
                     if p[0] == uuid:
                         ranks.append(i+1)
                         scores.append(p[1])
                         break
            
            ranks = np.array(ranks)
        
        # 4. Sector Analysis

        # Helper to get sectors
        def get_sectors(uuids):
            sectors = []
            if self.orgs_df is not None:
                subset = self.orgs_df[self.orgs_df['org_uuid'].isin(uuids)]
                for cats in subset['category_list'].dropna():
                    # Split by | or ,
                    parts = str(cats).split(',') # Crunchbase usually uses comma or pipe. Let's assume comma based on standard CSV but check data.
                    # Actually often it's '|'. Let's try both.
                    if '|' in str(cats): parts = str(cats).split('|')
                    
                    # Take top 1 or all? Let's take all unique categories
                    sectors.extend([p.strip() for p in parts])
            return pd.Series(sectors).value_counts(normalize=True).head(5)

        
        # 5. Stage Analysis (at time of investment)
        # For investor: use the investment_type from the round they invested in
        # For model: use the last_stage_map (stage at prediction time)
        
        inv_stages = benchmark_df['investment_type'].value_counts(normalize=True).head(5)
        
        model_stages = []
        for uuid in model_uuids:
            stage = self.last_stage_map.get(uuid, 'Unknown')
            model_stages.append(stage)
        model_stage_counts = pd.Series(model_stages).value_counts(normalize=True).head(5)
        
        
        # 6. Missed Winners (High Conviction Investor, Low Model Score)
        # Find startups in Investor Portfolio that were SUCCESSFUL but NOT in Model Top k
        # Only consider those IN THE TEST SET (Addressable)
        
        # Re-identify success (simplified)
        future_funding = self.funding_df[self.funding_df['announced_on'] > self.roi_start_date]
        funded_orgs = set(future_funding['org_uuid'].unique())
        ipo_orgs = set(self.ipos_df['org_uuid']) if self.ipos_df is not None else set()
        acq_orgs = set(self.acq_df['org_uuid']) if self.acq_df is not None else set()
        successful_uuids = funded_orgs.union(ipo_orgs).union(acq_orgs)
        
        missed_winners = []
        for uuid in investor_in_test_set: # Only check addressable ones
            if uuid in successful_uuids and uuid not in model_uuids:
                # Find rank in model
                rank = -1
                score = 0.0
                for i, p in enumerate(sorted_preds):
                    if p[0] == uuid:
                        rank = i + 1
                        score = p[1]
                        break
                
                # Get Name
                name = "Unknown"
                if self.orgs_df is not None:
                    name_row = self.orgs_df[self.orgs_df['org_uuid'] == uuid]
                    if not name_row.empty:
                        name = name_row.iloc[0]['name']
                        
                missed_winners.append((name, rank, score))
        
        # Sort by Rank (closest to being picked)
        missed_winners.sort(key=lambda x: x[1])
        
        # if not missed_winners:
        # for name, rank, score in missed_winners[:5]:
    

    def calculate_roi(self, predictions: List[Tuple[str, float, int]], top_k: int = 100, 
                      filename_suffix: str = "", title_suffix: str = "", verbose: bool = True, plot_charts: bool = True):
        """
        Calculate Investor-Centric ROI for the predictions.

        Methodology (Strategy-Aware):
        -----------------------------
        1.  **Fixed Ticket Size**: 
            - We simulate investing a fixed $1,000,000 ticket into each predicted startup.

        2.  **Entry Valuation & Ownership**:
            - **Entry Valuation**: Estimated using the formula: `(Raised / Dilution) / Step_Up`.
            - **Step-Up Multiple**: NOW VARIABLE.
                - Venture Strategy: 3.0x (Aggressive)
                - Liquidity Strategy: 1.5x (Conservative)
            
        3.  **Exit Value Calculation (DECOUPLED FROM PREDICTION)**:
            - **IPO**: `Market_Cap * Ownership`.
            - **Acquisition**: `Acquisition_Price * Ownership`.
            - **Funded (Paper Gain)**: 
                - Condition: Did it ACTUALLY raise money in the target window? (Ground Truth)
                - Value: `Cost * Step_Up`.
                - Note: This credits the asset regardless of whether we predicted 'Exit' or 'Funding'. 
                  Crucially, for Liquidity Strategy, the Step_Up is lower (1.5x), accurately reflecting a "Liquidity Trap" (Good asset, but not the Exit we wanted).
            - **Fail**: $0.

        Args:
            predictions (list): List of (uuid, score, label) tuples.
            filename_suffix (str): Suffix for saved plot files.
            title_suffix (str): Suffix for plot titles.
            verbose (bool): If True, print detailed output.
            plot_charts (bool): If True, generate and save plots.
        """
        if self.target_window_df is None:
            return {}
            
        if verbose:
            print(f"\nROI Analysis{title_suffix}")
        
        metrics = {}
        
        # Group funding by org
        org_funding = self.target_window_df.groupby('org_uuid').agg({
            'raised_amount_usd': 'sum',
            'post_money_valuation_usd': 'max',
            'investment_type': 'last'
        }).reset_index()
        
        successful_orgs = set(org_funding['org_uuid'].unique())
        funding_map = org_funding.set_index('org_uuid').to_dict('index')
        
        # Sort predictions by score
        sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        # --- Pre-calculate Costs and Values for ALL predictions for plotting ---
        all_costs = []
        all_values = []
        full_data = [] # Store all data for filtering
        
        # Load Total Funding for Valuation Fallback
        total_funding_map = {}
        if self.orgs_df is not None:
            total_funding_map = self.orgs_df.set_index('org_uuid')['total_funding_usd'].to_dict()
        
        # Load IPOs and Acquisitions for Exit Value lookup
        # Filter for Exits occurring AFTER ROI Start Date (Future Exits)
        ipo_map = {}
        if self.ipos_df is not None:
            # Filter by date
            future_ipos = self.ipos_df[self.ipos_df['went_public_on'] >= self.roi_start_date]
            ipo_map = future_ipos.set_index('org_uuid')['valuation_price_usd'].to_dict()
            
        acq_map = {}
        if self.acq_df is not None:
            # Filter by date
            future_acqs = self.acq_df[self.acq_df['announced_on'] >= self.roi_start_date]
            acq_map = future_acqs.set_index('org_uuid')['price_usd'].to_dict()

        # Load Names for Export
        name_map = {}
        if self.orgs_df is not None:
            # Optimize: Create a map instead of filtering DF in loop
            name_map = self.orgs_df.set_index('org_uuid')['name'].to_dict()

        for uuid, score, label in sorted_preds:
            # Get Funding Data
            data = funding_map.get(uuid, {})
            raised = data.get('raised_amount_usd', 0)
            if pd.isna(raised): raised = 0
            
            # Determine Dilution Rate (Ownership Share)
            round_type = data.get('investment_type', None)
            
            if pd.isna(round_type):
                round_type_str = None # Export as None/NaN
                dilution_rate = 0.10 # Default to Seed rate for safety if needed
            else:
                round_type_str = str(round_type).lower().replace(' ', '_').replace('-', '_')
                dilution_rate = self.DILUTION_RATES.get(round_type_str, 0.15)
            
            # --- Fixed Ticket Size Simulation ---
            # Cost: Fixed Ticket Size
            cost = self.TICKET_SIZE
            all_costs.append(cost)
            
            # Calculate Entry Valuation (Pre-Money of the round we "invested" in)
            # We assume we invested in the PREVIOUS round.
            # Current Post-Val = Raised / Dilution
            # Current Pre-Val = Post-Val - Raised
            # Entry Val = Current Pre-Val / Step_Up (Approximate)
            
            current_post_val = raised / dilution_rate if dilution_rate > 0 else raised
            current_pre_val = current_post_val - raised
            
            # Use Tiered Step Up to estimate entry valuation backward
            # (Approximation: We invert the step up logic)
            # Default to 2.0x for backward estimation if unknown
            entry_step_divisor = 2.0 
            entry_val = current_pre_val / entry_step_divisor
            
            # Default Valuation if data missing or weird
            if raised == 0 or entry_val <= 100_000: 
                # Check Historical Funding
                total_funding = total_funding_map.get(uuid, 0)
                if pd.notna(total_funding) and total_funding > 0:
                     entry_val = total_funding * self.HISTORICAL_FUNDING_MULTIPLE
                else:
                     entry_val = 5_000_000 # Default Seed Pre-Money
            

            # Ownership: Ticket / Entry Val (Capped at 20%)
            ownership = self.TICKET_SIZE / entry_val
            if ownership > 0.20:
                ownership = 0.20
            
            # --- Exit Value Calculation ---
            exit_val = 0.0
            status = 'FAIL'
            
            # Check for IPO
            if uuid in ipo_map:
                # IPO Value = Market Cap * Ownership
                market_cap = ipo_map[uuid]
                exit_val = market_cap * ownership
                status = 'EXIT'
                
            # Check for Acquisition
            elif uuid in acq_map:
                # Acquisition Value = Price * Ownership
                acq_price = acq_map[uuid]
                exit_val = acq_price * ownership
                status = 'EXIT'
            
            # Check for Undisclosed Exit (Status=EXIT but Value=0)
            if status == 'EXIT' and (pd.isna(exit_val) or exit_val == 0):
                 # Fallback Estimation for Undisclosed Exit
                 total_funding = total_funding_map.get(uuid, 0)
                 if pd.isna(total_funding): total_funding = 0
                 
                 if total_funding > 0:
                     # Estimate: Modest Win (Capital Returned)
                     estimated_exit_valuation = total_funding * self.UNDISCLOSED_EXIT_MULTIPLE
                     exit_val = estimated_exit_valuation * ownership
                 else:
                     # Fallback: Break-even (1x Cost)
                     exit_val = cost
            
            # Check for Funding (Paper Gain) if not Exited
            if status != 'EXIT':
                # Did it actually raise money? (Ground Truth)
                if raised > 0:
                    # Determine Step-Up based on Stage of the NEW Round
                    # "Valuation follows Stage"
                    if round_type_str in self.TIERED_STEP_UPS:
                        current_step_up = self.TIERED_STEP_UPS[round_type_str]
                    else:
                        current_step_up = self.TIERED_STEP_UPS['default']
                        
                    # Paper Gain = Cost * current_step_up
                    exit_val = cost * current_step_up
                    status = 'FUNDED'
                else:
                    status = 'FAIL'
            
            all_values.append(exit_val)
            
            # Collect Data for Export
            profit = exit_val - cost
            roi_mult = exit_val / cost if cost > 0 else 0
            # Status is already determined above
            
            # Get Name (Optimized)
            name = name_map.get(uuid, "Unknown")
            
            full_data.append({
                'UUID': uuid,
                'Name': name,
                'Score': score,
                'Status': status,
                'Cost': cost,
                'Value': exit_val,
                'Profit': profit,
                'ROI_Multiple': roi_mult,
                'Raised': raised,
                'Entry_Val': entry_val,
                'Ownership': ownership,
                'Stage': round_type_str,
                'Dilution': dilution_rate
            })

        all_costs = np.array(all_costs)
        all_values = np.array(all_values)
        
        # --- Process Top 100 Lists ---
        df_full = pd.DataFrame(full_data)
        
        # 1. Standard Top 100 (All Predictions)
        df_standard = df_full.head(100).copy()
        df_standard['Rank'] = range(1, len(df_standard) + 1)
        
        export_path = os.path.join(self.output_dir, f'top_100_investments{filename_suffix}.csv')
        df_standard.to_csv(export_path, index=False)
        
        # Calculate Standard Metrics
        std_cost = df_standard['Cost'].sum()
        std_val = df_standard['Value'].sum()
        std_roi = (std_val - std_cost) / std_cost if std_cost > 0 else 0
        std_prec = len(df_standard[df_standard['Status'] != 'FAIL']) / len(df_standard)
        
        if verbose:
            print(f"   Top-100 Portfolio (Standard){title_suffix}:")
            print(f"      Precision: {std_prec:.2%}")
            print(f"      ROI: {std_roi:.1%} ({std_val/std_cost:.2f}x)")
            print(f"      Total Capital: ${std_cost/1e6:.1f}M")
            print(f"      Total Value: ${std_val/1e6:.1f}M")

        if wandb.run is not None:
            wandb.log({
                f"analysis/top_100_precision{filename_suffix}": std_prec,
                f"analysis/top_100_roi{filename_suffix}": std_roi,
                f"analysis/top_100_capital{filename_suffix}": std_cost,
                f"analysis/top_100_value{filename_suffix}": std_val
            })



        # Calculate Cumulative Arrays for Plotting (Standard)
        cum_costs = np.cumsum(all_costs)
        cum_values = np.cumsum(all_values)
        ranks = np.arange(1, len(sorted_preds) + 1)
        sorted_labels = np.array([p[2] for p in sorted_preds]) # Labels in rank order
        
        # 1.1 Investor Benchmark (a16z)
        # We need to call this BEFORE plotting to pass the metrics
        benchmark_metrics = self.analyze_investor_benchmark()
        
        # --- Plotting (Cumulative) ---
        if plot_charts:
            # self._plot_roi_j_curve(ranks, sorted_costs_cum, sorted_values_cum, top_k, filename_suffix=filename_suffix) # Disabled
            self._plot_net_profit_curve(ranks, cum_costs, cum_values, labels=sorted_labels, filename_suffix=filename_suffix)
            
            # Plot Precision @ k (Standard) - NO BENCHMARK
            # self._plot_precision_at_k(ranks, sorted_labels, filename_suffix=filename_suffix)
            
            # Plot Precision vs ROI (Standard)
            # self._plot_precision_vs_roi(ranks, sorted_labels, sorted_costs_cum, sorted_values_cum, filename_suffix=filename_suffix)
        
        # Plot Precision @ k (Standard) - NO BENCHMARK
        # We need labels in rank order
        # sorted_labels = np.array([p[2] for p in sorted_preds])
        if plot_charts:
            self._plot_precision_at_k(ranks, sorted_labels, filename_suffix=filename_suffix)

            # Plot Precision vs ROI (Standard) - WITH BENCHMARK
            self._plot_precision_vs_roi(ranks, sorted_labels, all_costs, all_values, filename_suffix=filename_suffix, benchmark_metrics=benchmark_metrics)

            # Plot k vs Precision & ROI (Standard) - NO BENCHMARK
            self._plot_k_vs_precision_and_roi(ranks, sorted_labels, all_costs, all_values, filename_suffix=filename_suffix)

            # Plot k vs Precision & Recall (Standard) - NO BENCHMARK
            self._plot_k_vs_precision_and_recall(ranks, sorted_labels, filename_suffix=filename_suffix)
        
        # --- Zoomed Plots (k=1000) ---
        zoom_k = 1000
        if len(ranks) > 0 and plot_charts:
            # Slice data
            zoom_idx = min(len(ranks), zoom_k)
            zoom_ranks = ranks[:zoom_idx]
            zoom_labels = sorted_labels[:zoom_idx]
            zoom_costs = all_costs[:zoom_idx]
            zoom_values = all_values[:zoom_idx]
            
            # Plot Precision @ k (Zoomed) - WITH BENCHMARK, global base rate
            self._plot_precision_at_k(zoom_ranks, zoom_labels, filename_suffix=f"{filename_suffix}_zoomed", benchmark_metrics=benchmark_metrics, global_base_rate=np.mean(sorted_labels))
            
            # Plot k vs Precision & ROI (Zoomed) - WITH BENCHMARK
            self._plot_k_vs_precision_and_roi(zoom_ranks, zoom_labels, zoom_costs, zoom_values, filename_suffix=f"{filename_suffix}_zoomed", benchmark_metrics=benchmark_metrics)

            # Plot k vs Precision & Recall (Zoomed) - WITH BENCHMARK
            self._plot_k_vs_precision_and_recall(zoom_ranks, zoom_labels, filename_suffix=f"{filename_suffix}_zoomed", benchmark_metrics=benchmark_metrics)

        

        
        k_values = [10, 50, 100, 500, 1000]
        for k in k_values:
            if k > len(sorted_preds):
                continue
                
            top_k_preds = sorted_preds[:k]
            top_k_uuids = [p[0] for p in top_k_preds]
            
            # 1. Success Rate (Precision)
            successes = [uuid for uuid in top_k_uuids if uuid in successful_orgs]
            num_successes = len(successes)
            precision = num_successes / k
            
            # 2. Portfolio ROI (Dilution Model)
            # Sum of Costs and Values for top k
            k_cost = np.sum(all_costs[:k])
            k_value = np.sum(all_values[:k])
            
            roi_multiple = k_value / k_cost if k_cost > 0 else 0.0
            roi_percentage = (roi_multiple - 1) * 100
            
            # Metrics
            metrics[f'roi_precision_at_{k}'] = precision
            metrics[f'roi_total_raised_at_{k}'] = k_cost
            metrics[f'roi_percentage_at_{k}'] = roi_percentage
            metrics[f'roi_multiple_at_{k}'] = roi_multiple
            
            if k == 100:
                print(f"   Top-100 Portfolio:")
                print(f"      Precision: {precision:.2%}")
                print(f"      ROI: {roi_percentage:.1f}% ({roi_multiple:.2f}x)")
                print(f"      Total Capital Deployed: ${k_cost/1e6:.1f}M")
                print(f"      Total Portfolio Value: ${k_value/1e6:.1f}M")
                
                # Plot J-Curve for Top 100
                # self._plot_roi_j_curve(ranks, cum_costs, cum_values, top_k=100, filename_suffix=filename_suffix) # Disabled

        # Log to W&B
        if wandb.run is not None:
            wandb.log({f"roi_{k}": v for k, v in metrics.items()})
            
        # Add Curves to Metrics for external plotting
        metrics['curves'] = (ranks, cum_costs, cum_values, sorted_labels)

        return metrics

    def _calculate_group_metrics(self, df: pd.DataFrame, group_col: str, min_support: int = 10) -> pd.DataFrame:
        """
        Calculate comprehensive metrics for each group in the dataframe.
        Metrics: AUC-ROC, AUC-PR, F1, Precision, Recall, Accuracy, Support.
        """
        metrics_list = []
        
        # Get unique groups
        groups = df[group_col].unique()
        
        for group in groups:
            group_df = df[df[group_col] == group]
            
            # Skip if support is too low
            if len(group_df) < min_support:
                continue
                
            y_true = group_df['gt_label'].values
            y_score = group_df['score'].values
            y_pred = (y_score >= 0.5).astype(int) # Default threshold
            
            # Skip if only one class present (cannot calc AUC)
            if len(np.unique(y_true)) < 2:
                auc_roc = np.nan
                auc_pr = np.nan
            else:
                auc_roc = roc_auc_score(y_true, y_score)
                auc_pr = average_precision_score(y_true, y_score)
                
            metrics = {
                group_col: group,
                'support': len(group_df),
                'auc_roc': auc_roc,
                'auc_pr': auc_pr,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred),
                'mean_score': np.mean(y_score),
                'success_rate': np.mean(y_true)
            }
            metrics_list.append(metrics)
            
        if not metrics_list:
            return pd.DataFrame()
            
        metrics_df = pd.DataFrame(metrics_list)
        return metrics_df.sort_values('auc_pr', ascending=False)

    def _plot_analysis(self, stats_df: pd.DataFrame, raw_df: pd.DataFrame, group_col: str,
                      metric: str = 'auc_pr', title_suffix: str = "", filename_suffix: str = "", color: str = '#1f77b4',
                      order: Optional[List[str]] = None):
        """
        Generate Bar Plot for Metric and Box Plot for Scores.
        """
        if stats_df.empty:
            return

        if order is None:
            order = stats_df[group_col].tolist()
        raw_df_filtered = raw_df[raw_df[group_col].isin(stats_df[group_col])].copy()

        # 1. Metric Bar Plot
        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                sns.barplot(data=stats_df, x=group_col, y=metric, order=order, color=color, ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                _apply_thesis_style(ax)
                fig.tight_layout()
                plot_path = os.path.join(self.output_dir, f'{group_col}_performance{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)
            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_performance_plot": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} metric: {e}")

        # 2. Score Box Plot
        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                sns.boxplot(data=raw_df_filtered, x=group_col, y='score', order=order, color=color, ax=ax)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                _apply_thesis_style(ax)
                fig.tight_layout()
                plot_path = os.path.join(self.output_dir, f'{group_col}_scores_boxplot{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)
            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_scores_boxplot": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} boxplot: {e}")

    def _plot_correlation(self, stats_df: pd.DataFrame, group_col: str, metric: str = 'auc_pr', title_suffix: str = "", filename_suffix: str = ""):
        """
        Generate Scatter Plot correlating Support with Metric.
        """
        if stats_df.empty:
            return

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.0))
                ax.scatter(stats_df['support'], stats_df[metric], alpha=0.7, color='#1f77b4',
                           edgecolors='white', linewidths=0.5, s=50)

                if len(stats_df) < 50:
                    for _, row in stats_df.iterrows():
                        ax.annotate(str(row[group_col]), (row['support'], row[metric]),
                                    fontsize=7, alpha=0.7, xytext=(4, 2), textcoords='offset points')

                ax.set_xlabel('Support (Number of Samples)')
                ax.set_ylabel(metric.upper())
                _apply_thesis_style(ax)
                fig.tight_layout()

                plot_path = os.path.join(self.output_dir, f'{group_col}_correlation{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_correlation_plot": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} correlation: {e}")

    def _plot_success_correlation(self, stats_df: pd.DataFrame, group_col: str, metric: str = 'auc_pr', title_suffix: str = "", filename_suffix: str = ""):
        """
        Generate Scatter Plot correlating Success Rate with Metric.
        """
        if stats_df.empty:
            return

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.5))

                min_s, max_s = stats_df['support'].min(), stats_df['support'].max()
                if max_s > min_s:
                    norm = (stats_df['support'] - min_s) / (max_s - min_s)
                else:
                    norm = pd.Series([0.5] * len(stats_df))
                sizes = 50 + norm * 300

                ax.scatter(stats_df['success_rate'], stats_df[metric], s=sizes, alpha=0.7,
                           color='#1f77b4', edgecolors='white', linewidths=0.5)

                if len(stats_df) > 2:
                    sns.regplot(data=stats_df, x='success_rate', y=metric, scatter=False,
                                color='#d62728', line_kws={'linestyle': '--', 'alpha': 0.5, 'linewidth': 1}, ax=ax)

                top_metric = stats_df.nlargest(5, metric).index
                top_success = stats_df.nlargest(5, 'success_rate').index
                for idx in set(top_metric).union(set(top_success)):
                    row = stats_df.loc[idx]
                    ax.annotate(str(row[group_col]), (row['success_rate'], row[metric]),
                                fontsize=7, alpha=0.8, xytext=(4, 2), textcoords='offset points')

                ax.set_xlabel('Success Rate (Mean Ground Truth)')
                ax.set_ylabel(metric.upper())
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                _apply_thesis_style(ax)
                fig.tight_layout()

                plot_path = os.path.join(self.output_dir, f'{group_col}_success_correlation{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_success_correlation_plot": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} success correlation: {e}")

    def _plot_dual_metric_correlation(self, stats_df: pd.DataFrame, group_col: str,
                                       title_suffix: str = "", filename_suffix: str = ""):
        """
        Plot Success Rate vs both AUC-PR and AUC-ROC, showing that AUC-ROC is stable
        while AUC-PR varies with base rate. Bubble size = support. All points labelled.
        """
        if stats_df.empty or 'auc_pr' not in stats_df.columns or 'auc_roc' not in stats_df.columns:
            return

        try:
            from adjustText import adjust_text
        except ImportError:
            adjust_text = None

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.5))

                df = stats_df.sort_values('success_rate').copy()
                point_labels = df[group_col].astype(str).values

                min_s, max_s = df['support'].min(), df['support'].max()
                if max_s > min_s:
                    norm_support = (df['support'] - min_s) / (max_s - min_s)
                else:
                    norm_support = pd.Series([0.5] * len(df))
                sizes = 60 + norm_support * 300

                # AUC-PR scatter + trendline
                ax.scatter(df['success_rate'], df['auc_pr'], s=sizes, alpha=0.8,
                          color='#1f77b4', edgecolors='white', linewidths=0.5, zorder=3, label='AUC-PR')
                if len(df) > 2:
                    sns.regplot(data=df, x='success_rate', y='auc_pr', scatter=False,
                               color='#1f77b4', line_kws={'linestyle': '--', 'alpha': 0.5, 'linewidth': 1}, ax=ax)

                # AUC-ROC scatter + trendline
                ax.scatter(df['success_rate'], df['auc_roc'], s=sizes, alpha=0.8,
                          color='#ff7f0e', edgecolors='white', linewidths=0.5, zorder=3, label='AUC-ROC')
                if len(df) > 2:
                    sns.regplot(data=df, x='success_rate', y='auc_roc', scatter=False,
                               color='#ff7f0e', line_kws={'linestyle': '--', 'alpha': 0.5, 'linewidth': 1}, ax=ax)

                # Label all points (use adjustText to prevent overlap)
                texts = []
                for i, label in enumerate(point_labels):
                    row = df.iloc[i]
                    texts.append(ax.text(row['success_rate'], row['auc_pr'], label,
                                         fontsize=7, color='#1f77b4', alpha=0.9))
                    texts.append(ax.text(row['success_rate'], row['auc_roc'], label,
                                         fontsize=7, color='#ff7f0e', alpha=0.9))

                if adjust_text is not None:
                    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='grey', alpha=0.4, lw=0.5),
                                expand=(1.3, 1.5), force_text=(0.5, 0.8))

                ax.set_xlabel('Success Rate (Base Rate)')
                ax.set_ylabel('Metric Value')

                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                           markersize=8, label='AUC-PR'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
                           markersize=8, label='AUC-ROC'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', alpha=0.4,
                           markersize=5, label=f'n={int(min_s)}'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', alpha=0.4,
                           markersize=11, label=f'n={int(max_s)}'),
                    Line2D([0], [0], color='grey', linestyle='--', alpha=0.5, linewidth=1,
                           label='Trend'),
                    Patch(facecolor='grey', alpha=0.15, label='95% CI'),
                ]
                ax.legend(handles=legend_elements, fontsize=8, loc='best', framealpha=0.9)
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                _apply_thesis_style(ax)
                fig.tight_layout()

                plot_path = os.path.join(self.output_dir, f'{group_col}_dual_metric{filename_suffix}.pdf')
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/{group_col}_dual_metric{filename_suffix}": wandb.Image(plot_path)})
        except Exception as e:
            print(f"Failed to plot {group_col} dual metric correlation: {e}")

    def analyze_sectors(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by sector (category_list)."""
        k_str = f"Top {top_k}" if top_k else "All"
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
        
        top_df['category_list'] = top_df['category_list'].fillna('Unknown').astype(str)
        top_df['sector'] = top_df['category_list'].apply(lambda x: [s.strip() for s in x.split(',')])
        exploded_df = top_df.explode('sector')
        
        # Calculate Metrics
        sector_stats = self._calculate_group_metrics(exploded_df, 'sector', min_support=20)
        
        if sector_stats.empty:
            print("   No sectors with sufficient support found.")
            return

        
        output_path = os.path.join(self.output_dir, f'sector_performance{filename_suffix}.csv')
        sector_stats.to_csv(output_path, index=False)
        
        # Plotting (Top 20 by Support, then sorted by AUC-PR)
        # Select top 20 by support
        top_20_sectors = sector_stats.sort_values('support', ascending=False).head(20)
        # Sort by AUC-PR for display
        top_20_sectors = top_20_sectors.sort_values('auc_pr', ascending=False)
        
        self._plot_analysis(top_20_sectors, exploded_df, 'sector', metric='auc_pr', 
                           title_suffix=f"Sector ({k_str})", filename_suffix=filename_suffix)
        
        # Correlation Plot (All Sectors)
        self._plot_correlation(sector_stats, 'sector', metric='auc_pr', title_suffix=f"Sector ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(sector_stats, 'sector', metric='auc_pr', title_suffix=f"Sector ({k_str})", filename_suffix=filename_suffix)
        # Dual metric plot (top 20 by support for readability)
        top_sectors = sector_stats.sort_values('support', ascending=False).head(20)
        self._plot_dual_metric_correlation(top_sectors, 'sector', title_suffix=f"Sector (Top 20)", filename_suffix=filename_suffix)

        if wandb.run is not None:
            wandb.log({f"analysis/sector_performance{filename_suffix}": wandb.Table(dataframe=sector_stats)})

        # Comparative Precision @ k (Top 10 Sectors)
        # Select top 10 by support for comparative plots
        top_10_sectors_df = sector_stats.sort_values('support', ascending=False).head(10)
        top_10_sectors_list = top_10_sectors_df['sector'].values
        
        preds_by_sector = {}
        
        # --- 1. Global Baseline ---
        # Global uses the unique startups from top_df (before explode)
        global_preds_list = list(zip(top_df['org_uuid'], top_df['score'], top_df['gt_label']))
        preds_by_sector['All Sectors (Global)'] = global_preds_list
        
        # --- 2. Top 10 Combined ---
        # Union of startups in top 10 sectors
        top_10_combined_df = exploded_df[exploded_df['sector'].isin(top_10_sectors_list)]
        top_10_combined_unique = top_10_combined_df.drop_duplicates(subset='org_uuid')
        
        combined_preds_list = list(zip(top_10_combined_unique['org_uuid'], top_10_combined_unique['score'], top_10_combined_unique['gt_label']))
        preds_by_sector['Top 10 Combined'] = combined_preds_list

        # --- 3. Individual Top 10 Sectors ---
        for sec in top_10_sectors_list:
             sec_preds_df = exploded_df[exploded_df['sector'] == sec]
             preds_list = list(zip(sec_preds_df['org_uuid'], sec_preds_df['score'], sec_preds_df['gt_label']))
             preds_by_sector[sec] = preds_list
             
        self._plot_comparative_precision_at_k(preds_by_sector, filename_suffix=f"{filename_suffix}_Sector_Top10")

    def analyze_funding_stage(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by granular funding stage (Series A, B, etc.)."""
        k_str = f"Top {top_k}" if top_k else "All"
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
        
        # Map to Granular Stage using self.last_stage_map
        if hasattr(self, 'last_stage_map') and self.last_stage_map:
            top_df['stage'] = top_df['org_uuid'].map(self.last_stage_map).fillna('Unknown')
        # else:
        #     # Fallback to num_funding_rounds proxy if map not available
        #     def get_stage(n):
        #         if pd.isna(n): return 'Unknown'
        #         if n <= 1: return 'Seed/Angel'
        #         if n <= 3: return 'Early Stage (Series A/B)'
        #         return 'Late Stage (Series C+)'
        #     top_df['stage'] = top_df['num_funding_rounds'].apply(get_stage)
            
        # Clean up stage names if needed (e.g., normalize)
        # Custom Grouping and Ordering
        desired_stages = {
            'angel', 'pre_seed', 'seed', 
            'series_a', 'series_b', 'series_c', 'series_d', 'series_e', 'series_f'
        }
        
        # Group everything else into 'other'
        top_df['stage'] = top_df['stage'].apply(lambda s: s if s in desired_stages else 'other')
        
        # Calculate Metrics
        stage_stats = self._calculate_group_metrics(top_df, 'stage', min_support=5)
        
        if stage_stats.empty:
            print("   No stages with sufficient support found.")
            return

        
        output_path = os.path.join(self.output_dir, f'stage_performance{filename_suffix}.csv')
        stage_stats.to_csv(output_path, index=False)
        
        # Explicit Stage Order
        stage_order = [
            'angel', 'pre_seed', 'seed', 
            'series_a', 'series_b', 'series_c', 'series_d', 'series_e', 'series_f', 
            'other'
        ]
        
        # Filter order to only include present stages
        final_order = [s for s in stage_order if s in stage_stats['stage'].values]
        
        # Use AUC-PR explicitly
        analysis_metric = 'auc_pr' 
        
        self._plot_analysis(stage_stats, top_df, 'stage', metric=analysis_metric, 
                           title_suffix=f"Funding Stage ({k_str})", order=final_order, filename_suffix=filename_suffix)
        
        # Correlation Plots
        self._plot_correlation(stage_stats, 'stage', metric=analysis_metric, title_suffix=f"Funding Stage ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(stage_stats, 'stage', metric=analysis_metric, title_suffix=f"Funding Stage ({k_str})", filename_suffix=filename_suffix)
        self._plot_dual_metric_correlation(stage_stats, 'stage', title_suffix=f"Funding Stage", filename_suffix=filename_suffix)

        if wandb.run is not None:
            wandb.log({f"analysis/stage_performance{filename_suffix}": wandb.Table(dataframe=stage_stats)})

        # Comparative Precision @ k (All Stages in Order)
        preds_by_stage = {}
        for stage in final_order:
            stage_preds = top_df[top_df['stage'] == stage]
            preds_list = list(zip(stage_preds['org_uuid'], stage_preds['score'], stage_preds['gt_label']))
            preds_by_stage[stage] = preds_list
            
        self._plot_comparative_precision_at_k(preds_by_stage, filename_suffix=f"{filename_suffix}_Funding_Stage")

    def analyze_founding_year(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by founding year."""
        k_str = f"Top {top_k}" if top_k else "All"
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
        
        top_df['founded_year'] = top_df['founded_on'].dt.year
        
        # Filter for reasonable years (e.g., >= 2000) to keep plot readable
        top_df = top_df[top_df['founded_year'] >= 2000]
        top_df['founded_year'] = top_df['founded_year'].astype(int)
        
        # Calculate Metrics
        year_stats = self._calculate_group_metrics(top_df, 'founded_year', min_support=10)
        
        if year_stats.empty:
            print(f"   WARNING: Not enough data for Founding Year analysis (min_support=10)")
            return
        
        
        output_path = os.path.join(self.output_dir, f'founding_year_performance{filename_suffix}.csv')
        year_stats.to_csv(output_path, index=False)
        
        # Plotting (Ordered by Year)
        year_order = sorted(year_stats['founded_year'].unique())
        
        self._plot_analysis(year_stats, top_df, 'founded_year', metric='auc_pr', 
                           title_suffix=f"Founding Year ({k_str})", order=year_order, filename_suffix=filename_suffix)
        
        # Correlation Plots
        self._plot_correlation(year_stats, 'founded_year', metric='auc_pr', title_suffix=f"Founding Year ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(year_stats, 'founded_year', metric='auc_pr', title_suffix=f"Founding Year ({k_str})", filename_suffix=filename_suffix)
        self._plot_dual_metric_correlation(year_stats, 'founded_year', title_suffix=f"Founding Year", filename_suffix=filename_suffix)

        if wandb.run is not None:
            wandb.log({f"analysis/founding_year_performance{filename_suffix}": wandb.Table(dataframe=year_stats)})

        # Comparative Precision @ k by founding year cohort
        preds_by_year = {}

        # Global baseline
        global_preds = list(zip(top_df['org_uuid'], top_df['score'], top_df['gt_label']))
        preds_by_year['All Years (Global)'] = global_preds

        # Individual years (only those with sufficient support)
        valid_years = year_stats[year_stats['support'] >= 20]['founded_year'].values
        for year in sorted(valid_years):
            year_preds = top_df[top_df['founded_year'] == year]
            preds_list = list(zip(year_preds['org_uuid'], year_preds['score'], year_preds['gt_label']))
            preds_by_year[str(int(year))] = preds_list

        self._plot_comparative_precision_at_k(preds_by_year, filename_suffix=f"{filename_suffix}_Founding_Year")

    def analyze_geography(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by country."""
        k_str = f"Top {top_k}" if top_k else "All"
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
        
        # Calculate Metrics
        geo_stats = self._calculate_group_metrics(top_df, 'country_code', min_support=20)
        
        if geo_stats.empty:
             print(f"   WARNING: Not enough data for Geography analysis (min_support=20)")
             return
        
        
        output_path = os.path.join(self.output_dir, f'geo_performance{filename_suffix}.csv')
        geo_stats.to_csv(output_path, index=False)
        
        # Plotting (Top 20 by Support, then sorted by AUC-PR)
        top_20_geo = geo_stats.sort_values('support', ascending=False).head(20)
        top_20_geo = top_20_geo.sort_values('auc_pr', ascending=False)
        
        self._plot_analysis(top_20_geo, top_df, 'country_code', metric='auc_pr', 
                           title_suffix=f"Country ({k_str})", filename_suffix=filename_suffix)
        
        # Correlation Plot (All Countries)
        self._plot_correlation(geo_stats, 'country_code', metric='auc_pr', title_suffix=f"Country ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(geo_stats, 'country_code', metric='auc_pr', title_suffix=f"Country ({k_str})", filename_suffix=filename_suffix)
        
        if wandb.run is not None:
            wandb.log({f"analysis/geo_performance{filename_suffix}": wandb.Table(dataframe=geo_stats)})

        # Comparative Precision @ k (Top 10 Countries)
        # Select top 10 by support
        top_10_geo_df = geo_stats.sort_values('support', ascending=False).head(10)
        top_10_countries = top_10_geo_df['country_code'].values
        
        preds_by_country = {}
        
        # --- 1. Global Baseline ---
        global_preds_list = list(zip(top_df['org_uuid'], top_df['score'], top_df['gt_label']))
        preds_by_country['All Countries (Global)'] = global_preds_list
        
        # --- 2. Top 10 Combined ---
        top_10_combined_df = top_df[top_df['country_code'].isin(top_10_countries)]
        combined_preds_list = list(zip(top_10_combined_df['org_uuid'], top_10_combined_df['score'], top_10_combined_df['gt_label']))
        preds_by_country['Top 10 Combined'] = combined_preds_list
        
        # --- 3. Individual Top 10 Countries ---
        for country in top_10_countries:
            country_preds = top_df[top_df['country_code'] == country]
            preds_list = list(zip(country_preds['org_uuid'], country_preds['score'], country_preds['gt_label']))
            preds_by_country[country] = preds_list
            
        self._plot_comparative_precision_at_k(preds_by_country, filename_suffix=f"{filename_suffix}_Country_Top10")

    def analyze_continents(self, df: pd.DataFrame, top_k: Optional[int] = 1000, filename_suffix: str = ""):
        """Analyze performance by continent."""
        k_str = f"Top {top_k}" if top_k else "All"
        
        if top_k:
            top_df = df.sort_values('score', ascending=False).head(top_k).copy()
        else:
            top_df = df.copy()
            
        # Map Country to Continent
        top_df['continent'] = top_df['country_code'].apply(convert_to_continent)
        
        # Calculate Metrics
        cont_stats = self._calculate_group_metrics(top_df, 'continent', min_support=10)
        
        if cont_stats.empty:
             print(f"   WARNING: Not enough data for Continent analysis (min_support=10)")
             return
        
        
        output_path = os.path.join(self.output_dir, f'continent_performance{filename_suffix}.csv')
        cont_stats.to_csv(output_path, index=False)
        
        # Plotting
        self._plot_analysis(cont_stats, top_df, 'continent', metric='auc_pr', 
                           title_suffix=f"Continent ({k_str})", filename_suffix=filename_suffix)
        
        # Correlation Plots
        self._plot_correlation(cont_stats, 'continent', metric='auc_pr', title_suffix=f"Continent ({k_str})", filename_suffix=filename_suffix)
        self._plot_success_correlation(cont_stats, 'continent', metric='auc_pr', title_suffix=f"Continent ({k_str})", filename_suffix=filename_suffix)
        self._plot_dual_metric_correlation(cont_stats, 'continent', title_suffix=f"Continent", filename_suffix=filename_suffix)

        if wandb.run is not None:
            wandb.log({f"analysis/continent_performance{filename_suffix}": wandb.Table(dataframe=cont_stats)})

        # Comparative Precision @ k (All Continents)
        preds_by_continent = {}
        # Sort by support? or alphabetical? Let's use AUC-PR order for legend
        sorted_continents = cont_stats.sort_values('auc_pr', ascending=False)['continent'].values
        
        for cont in sorted_continents:
            cont_preds = top_df[top_df['continent'] == cont]
            preds_list = list(zip(cont_preds['org_uuid'], cont_preds['score'], cont_preds['gt_label']))
            preds_by_continent[cont] = preds_list
            
        self._plot_comparative_precision_at_k(preds_by_continent, filename_suffix=f"{filename_suffix}_Continent")

    def _plot_comparative_net_profit(self, group_data: Dict[str, Tuple], filename_suffix: str = ""):
        """
        Plot multiple Net Profit curves on the same chart.
        group_data: { 'Group Name': (ranks, costs, values, labels) }
        """
        if not group_data:
            return

        try:
            with plt.rc_context(_THESIS_RCPARAMS):
                fig, ax = plt.subplots(figsize=(7.5, 4.5))

                n = len(group_data)
                palette = plt.cm.tab10(np.linspace(0, 0.9, min(n, 10))) if n <= 10 else plt.cm.tab20(np.linspace(0, 0.95, n))

                for i, (name, (ranks, costs, values, labels)) in enumerate(group_data.items()):
                    if len(ranks) == 0: continue
                    profits = np.nan_to_num(values - costs)
                    ax.plot(ranks, profits, label=name, color=palette[i], linewidth=1.5, alpha=0.85)
                    peak_idx = np.argmax(profits)
                    ax.scatter(ranks[peak_idx], profits[peak_idx], color=palette[i], s=30, marker='o', zorder=5)

                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
                ax.set_xlabel('Portfolio Rank (Number of Startups)')
                ax.set_ylabel('Net Profit (USD)')
                ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.9)

                def currency_formatter(x, pos):
                    if abs(x) >= 1e9: return f'${x/1e9:.1f}B'
                    if abs(x) >= 1e6: return f'${x/1e6:.0f}M'
                    return f'${x:.0f}'
                ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
                _apply_thesis_style(ax)
                fig.tight_layout()

                filename = f'roi_comparative_net_profit{filename_suffix}.pdf'
                plot_path = os.path.join(self.output_dir, filename)
                fig.savefig(plot_path)
                plt.close(fig)

            if wandb.run is not None:
                wandb.log({f"analysis/roi_comparative_net_profit{filename_suffix}": wandb.Image(plot_path)})

        except Exception as e:
            print(f"Failed to plot Comparative Net Profit: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Downstream Analysis on Predictions CSV")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions CSV file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load Predictions
    print(f"Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions)
    
    # Process Predictions
    import ast
    
    print(f"   Parsing {len(pred_df)} predictions...")
    
    score_col = 'prediction' if 'prediction' in pred_df.columns else 'score'
    uuid_col = 'org_uuid' if 'org_uuid' in pred_df.columns else 'uuid'
    
    # Safe Eval Helper
    def safe_eval(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except (ValueError, SyntaxError):
            return float(x) if x else 0.0

    # Parse Columns
    pred_df[score_col] = pred_df[score_col].apply(safe_eval)
    pred_df['gt_label'] = pred_df['gt_label'].apply(safe_eval).fillna(0.0)

    predictions = list(zip(pred_df[uuid_col], pred_df[score_col], pred_df['gt_label']))

    print(f"Successfully parsed {len(predictions)} predictions.")

    # Initialize Analyzer
    analyzer = DownstreamAnalyzer(config)
    
    # Run Analysis
    analyzer.perform_downstream_analysis(predictions)
