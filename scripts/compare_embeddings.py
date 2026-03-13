"""Compare GNN-learned and text-based startup embeddings via clustering, visualization, and similarity analysis."""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import silhouette_score, pairwise_distances, adjusted_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("Script started!")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from scripts.case_study import load_graph_and_model
from src.ml.utils import load_config, get_maturity_mask

# --- HELPERS ---

def load_text_embeddings(csv_path, startup_uuids):
    """
    Load 384d text embeddings from CSV and align with graph startup UUIDs.
    Returns:
        valid_indices: Indices in the startup_uuids list that have text embeddings.
        text_embeddings: Numpy array of shape [N_valid, 384]
    """
    print(f"Loading text embeddings from {csv_path}...")
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Text embedding file not found at {csv_path}")

    # Read CSV (index_col=0 is usually UUID)
    df_emb = pd.read_csv(csv_path, index_col='uuid')
    
    # Reindex to match graph order (some will be NaN if missing)
    aligned_df = df_emb.reindex(startup_uuids)
    
    # Check which are valid
    valid_mask = aligned_df.iloc[:, 0].notna()
    valid_indices = np.where(valid_mask)[0]
    
    # Extract values
    emb_cols = [c for c in aligned_df.columns if c.startswith('emb_')]
    text_embeddings = aligned_df.iloc[valid_indices][emb_cols].values.astype(np.float32)
    
    print(f"   Aligning {len(startup_uuids)} graph nodes with {len(df_emb)} text embeddings...")
    print(f"   Found text embeddings for {len(valid_indices)} / {len(startup_uuids)} nodes ({len(valid_indices)/len(startup_uuids):.1%})")
    
    return valid_indices, text_embeddings

def compute_purity_at_k(embeddings, label_dict, k_values=(1, 5, 10, 20)):
    """Compute Neighborhood Purity@K for multiple label sets.

    For each node, checks what fraction of its K nearest neighbors
    (by cosine similarity) share the same label.

    Args:
        embeddings: L2-normalized numpy array (N, D)
        label_dict: dict of {name: labels_array} where labels_array is (N,)
        k_values: tuple of K values to evaluate

    Returns:
        dict of {name: {k: purity_score}}
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    sim_matrix = cos_sim(embeddings)
    np.fill_diagonal(sim_matrix, -1.0)

    results = {}
    for name, labels in label_dict.items():
        if labels is None:
            continue
        # Ensure labels is a numpy array (may be pandas Series from pd.qcut)
        labels = np.asarray(labels)
        n_classes = len(np.unique(labels))
        results[name] = {}
        for k in k_values:
            top_k_idx = np.argpartition(-sim_matrix, k, axis=1)[:, :k]
            neighbor_labels = labels[top_k_idx]
            self_labels = labels.reshape(-1, 1)
            purity = (neighbor_labels == self_labels).mean()
            results[name][k] = purity
        chance = 1.0 / n_classes
        results[name]['chance'] = chance
        results[name]['n_classes'] = n_classes
    return results


# ---------------------------------------------------------------------------
# Thesis figure style
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Retrieval overlap analysis (GNN vs Text)
# ---------------------------------------------------------------------------

def compute_retrieval_overlap(gnn_vecs, text_vecs, output_dir,
                              k_values=(10, 25, 50, 100, 250, 500, 1000)):
    """Compute Jaccard overlap between GNN and text top-K retrieval sets.

    For each query node, retrieves its K nearest neighbours in both embedding
    spaces and computes the Jaccard index |intersection| / |union|.  Produces
    a line plot of mean Jaccard overlap vs K.

    Args:
        gnn_vecs:  L2-normalised GNN embeddings  (N, D_gnn)
        text_vecs: L2-normalised text embeddings  (N, D_text)
        output_dir: pathlib.Path for saving outputs
        k_values:  tuple of K values for the overlap-vs-K curve
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    N = len(gnn_vecs)
    assert len(text_vecs) == N, "GNN and text must have the same number of nodes"
    print(f"\nComputing retrieval overlap (N={N}, K up to {max(k_values)})...")

    # Pre-compute full similarity matrices
    print("   Computing GNN similarity matrix...")
    gnn_sim = cos_sim(gnn_vecs)
    np.fill_diagonal(gnn_sim, -np.inf)  # exclude self

    print("   Computing text similarity matrix...")
    text_sim = cos_sim(text_vecs)
    np.fill_diagonal(text_sim, -np.inf)

    # Pre-compute top-K neighbour indices via argpartition (faster than argsort)
    max_k = max(k_values)
    print(f"   Ranking top-{max_k} neighbours...")
    gnn_topk_all = np.argpartition(-gnn_sim, max_k, axis=1)[:, :max_k]
    text_topk_all = np.argpartition(-text_sim, max_k, axis=1)[:, :max_k]

    # --- Overlap vs K ---
    results = []
    for k in k_values:
        jaccards = np.empty(N)
        for i in range(N):
            g_set = set(gnn_topk_all[i, :k])
            t_set = set(text_topk_all[i, :k])
            intersection = len(g_set & t_set)
            union = len(g_set | t_set)
            jaccards[i] = intersection / union if union > 0 else 0.0
        results.append({
            'k': k,
            'mean': jaccards.mean(),
            'median': np.median(jaccards),
            'std': jaccards.std(),
            'p25': np.percentile(jaccards, 25),
            'p75': np.percentile(jaccards, 75),
        })
        print(f"   K={k:>5d}  Jaccard: {jaccards.mean():.4f} ± {jaccards.std():.4f}  "
              f"(median {np.median(jaccards):.4f})")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'retrieval_overlap.csv', index=False)

    # ---- Figure: Overlap vs K curve ----
    thesis_fig_dir = output_dir / "figures"
    thesis_fig_dir.mkdir(exist_ok=True, parents=True)

    with plt.rc_context(_THESIS_RCPARAMS):
        fig, ax = plt.subplots(figsize=(6, 4))

        ks = df_results['k'].values
        means = df_results['mean'].values
        p25 = df_results['p25'].values
        p75 = df_results['p75'].values

        ax.plot(ks, means, marker='o', markersize=5, linewidth=1.5,
                color='#1f77b4', label='Mean Jaccard', zorder=3)
        ax.fill_between(ks, p25, p75, alpha=0.15, color='#1f77b4',
                        label='IQR (25th–75th)')

        # Reference: expected Jaccard for two random K-subsets of N items
        random_jaccard = ks / (2 * N - ks)
        ax.plot(ks, random_jaccard, linestyle=':', linewidth=1.2,
                color='0.5', label='Random baseline')

        ax.set_xlabel('K (number of retrieved neighbours)')
        ax.set_ylabel('Jaccard Overlap')
        ax.set_title('GNN vs. Text Retrieval Overlap')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        _apply_thesis_style(ax)
        fig.tight_layout()

        pdf_path = thesis_fig_dir / 'retrieval_overlap_curve.pdf'
        fig.savefig(pdf_path)
        fig.savefig(output_dir / 'retrieval_overlap_curve.png', dpi=150)
        plt.close(fig)
        print(f"   Saved overlap curve to {pdf_path}")

    return df_results


# --- ANALYZER CLASS ---

class EmbeddingAnalyzer:
    def __init__(self, name, embeddings, metadata_df, config, output_dir, method="umap", cluster_on_2d=True):
        """
        Generic Analyzer for any embedding set.
        params:
            name: Label for this analysis (e.g. "GNN", "Text")
            embeddings: Normalized numpy array (N, D)
            metadata_df: Aligned metadata DataFrame (N, ...)
            config: Config dict
            output_dir: Path object for saving outputs
        """
        self.name = name
        self.embeddings = embeddings
        self.meta = metadata_df
        self.config = config
        self.output_dir = output_dir
        self.method = method
        self.cluster_on_2d = cluster_on_2d
        
        # State
        self.proj_2d = None
        self.clusters = None
        self.k = None
        
        # Prepare Metadata Vectors (for fast metric calculation)
        self._prep_metadata_vectors()

    def _prep_metadata_vectors(self):
        """Prepare factorized/numeric vectors from metadata df."""
        self.meta_vecs = {}
        
        # Industry
        self.meta_vecs['Industry'], self.industry_levels = pd.factorize(self.meta['industry_top'])
        
        # City
        self.meta_vecs['City'], self.city_levels = pd.factorize(self.meta['city_top'])
        
        # Funding Bucket
        if len(self.meta['total_funding_usd'].unique()) > 5:
            self.meta_vecs['Funding'] = pd.qcut(self.meta['total_funding_usd'], q=5, labels=False, duplicates='drop')
        else:
            self.meta_vecs['Funding'] = None
            
        # Founded Year
        self.meta_vecs['Year'] = self.meta['founded_on_year'].values
        self.valid_year_mask = self.meta_vecs['Year'] > 1900
        
        # Stage
        self.meta_vecs['Stage'], self.stage_levels = pd.factorize(self.meta['last_funding_stage'])
        
        # Targets
        if 'status' in self.meta:
            self.meta_vecs['Status'], _ = pd.factorize(self.meta['status'])
        
        # Special handling for targets passed in meta (assumed to be added to meta_df before init)
        if 'momentum' in self.meta:
             self.meta_vecs['Momentum'] = self.meta['momentum'].values.astype(int)
        if 'liquidity' in self.meta:
             self.meta_vecs['Liquidity'] = self.meta['liquidity'].values.astype(int)
             
        # Maturity Mask
        mature_series = get_maturity_mask(self.meta, self.config)
        if mature_series is None: mature_series = pd.Series(True, index=self.meta.index)
        self.mature_mask = mature_series.values

    def run_dimensionality_reduction(self):
        print(f"\n[{self.name}] Reducing dimensions using {self.method.upper()}...")
        if self.method == "umap" and HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        else:
            reducer = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto', random_state=42)
        
        self.proj_2d = reducer.fit_transform(self.embeddings)
        return self.proj_2d

    def run_k_search(self, k_range=range(2, 11), subsample=10000):
        print(f"\n[{self.name}] Starting K-Search (Lowest within 5% of Max)...")
        
        # Decide what to cluster on
        X = self.proj_2d if self.cluster_on_2d else self.embeddings
        
        inertias = []
        silhouettes = []
        
        best_k = 0
        
        for k in k_range:
            print(f"   Testing K={k}...", end="\r")
            
            # For robustness, we should ideally use a train/test split.
            # But here we are just searching on the provided set (which IS the test split usually).
            # To match previous logic, we can just run on the full set provided.
            
            # Subsample for speed if needed
            if len(X) > subsample:
                idx = np.random.choice(len(X), subsample, replace=False)
                X_fit = X[idx]
            else:
                X_fit = X
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3).fit(X_fit)
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(kmeans.labels_)) < 2:
                sil = -1
            else:
                metric = 'euclidean' if self.cluster_on_2d else 'cosine'
                sil = silhouette_score(X_fit, kmeans.labels_, metric=metric)
            
            silhouettes.append(sil)
            
        # Select K logic
        max_sil = max(silhouettes)
        threshold = max_sil * 0.95
        
        for i, score in enumerate(silhouettes):
            if score >= threshold:
                best_k = k_range[i]
                break
                
        print(f"\n[{self.name}] K-Search Complete! Selected K={best_k} (Max: {max_sil:.4f}, Thr: {threshold:.4f})")
        
        # Save plot
        self._plot_k_search(k_range, inertias, silhouettes, best_k)
        
        return best_k

    def _plot_k_search(self, k_range, inertias, silhouettes, best_k):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color = 'tab:red'
        ax1.set_xlabel('K')
        ax1.set_ylabel('Inertia', color=color)
        ax1.plot(k_range, inertias, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Silhouette', color=color)
        ax2.plot(k_range, silhouettes, marker='s', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax2.axvline(best_k, color='green', linestyle='--', label=f'Best K={best_k}')
        plt.title(f'Optimal K Analysis ({self.name})')
        plt.savefig(self.output_dir / f"k_analysis_{self.name.lower()}.png")
        plt.close()

    def cluster(self, k):
        print(f"\n[{self.name}] Running K-Means (K={k})...")
        self.k = k
        X = self.proj_2d if self.cluster_on_2d else self.embeddings
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(X)
        return self.clusters

    def compute_metrics(self):
        print(f"\n[{self.name}] Calculating Metrics...")
        results = {}
        
        metric_vecs = self.proj_2d if self.cluster_on_2d else self.embeddings
        metric_dist = 'euclidean' if self.cluster_on_2d else 'cosine'
        
        for key, labels in self.meta_vecs.items():
            if labels is None: continue
            
            # Mask handling
            mask = None
            if key == 'Year': mask = self.valid_year_mask
            if key == 'Liquidity (Mature)': continue # Handled separately?
            
            # Apply mask
            curr_labels = labels
            curr_clusters = self.clusters
            curr_vecs = metric_vecs
            
            if mask is not None:
                curr_labels = curr_labels[mask]
                curr_clusters = curr_clusters[mask]
                curr_vecs = curr_vecs[mask]
                
            if len(curr_labels) == 0 or len(np.unique(curr_labels)) < 2: continue
            
            ami = adjusted_mutual_info_score(curr_labels, curr_clusters)
            sil = silhouette_score(curr_vecs, curr_labels, metric=metric_dist)
            
            results[f'AMI_{key}'] = ami
            results[f'Sil_{key}'] = sil
            print(f"      vs {key:15s} -> AMI: {ami:.4f} | Sil: {sil:.4f}")

        # Mature Liquidity special case
        if 'Liquidity' in self.meta_vecs:
             mask = self.mature_mask
             l_mature = self.meta_vecs['Liquidity'][mask]
             c_mature = self.clusters[mask]
             if len(l_mature) > 0:
                 ami_mat = adjusted_mutual_info_score(l_mature, c_mature)
                 results['AMI_Liquidity_Mature'] = ami_mat
                 print(f"      vs Liq(Mature)    -> AMI: {ami_mat:.4f}")

        # Neighborhood Purity@K (on full-dimensional embeddings)
        purity_labels = {k: v for k, v in self.meta_vecs.items() if v is not None and k not in ('Year', 'Status')}
        purity_results = compute_purity_at_k(self.embeddings, purity_labels)
        print(f"\n      Neighborhood Purity@K:")
        for feat, scores in purity_results.items():
            p10 = scores.get(10, 0)
            chance = scores.get('chance', 0)
            results[f'Purity@10_{feat}'] = p10
            print(f"        {feat:15s} -> Purity@10: {p10:.4f} (Chance: {chance:.4f})")

        pd.DataFrame([results]).to_csv(self.output_dir / f"metrics_{self.name.lower()}.csv", index=False)

    def profile_clusters(self):
        print(f"\n[{self.name}] Profiling Clusters...")
        data = []
        for c_id in range(self.k):
            mask = self.clusters == c_id
            count = mask.sum()
            if count == 0: continue
            
            sub_meta = self.meta[mask]
            
            # Stats
            avg_funding = sub_meta['total_funding_usd'].mean()
            
            mom_rate, liq_rate = 0, 0
            if 'Momentum' in self.meta_vecs: mom_rate = self.meta_vecs['Momentum'][mask].mean()
            if 'Liquidity' in self.meta_vecs: liq_rate = self.meta_vecs['Liquidity'][mask].mean()
            
            # Top Stage
            top_stage = "Unknown"
            if 'last_funding_stage' in sub_meta.columns:
                counts = sub_meta['last_funding_stage'].value_counts()
                if not counts.empty:
                    top_stage = f"{counts.index[0]} ({counts.iloc[0]/count:.0%})"
            
            # City Purity
            top_city_s = sub_meta['city'].value_counts()
            top_city = top_city_s.index[0] if not top_city_s.empty else "Unknown"
            purity = top_city_s.iloc[0]/count if not top_city_s.empty else 0
            
            data.append({
                "Cluster": c_id,
                "Count": count,
                "Avg Funding": f"${avg_funding/1e6:,.1f}M",
                "Top Stage": top_stage,
                "Momentum %": f"{mom_rate:.1%}",
                "Liquidity %": f"{liq_rate:.1%}",
                "Top City (Purity)": f"{top_city} ({purity:.1%})"
            })
            
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        df.to_csv(self.output_dir / f"profile_{self.name.lower()}.csv", index=False)

    # Numeric stage code → human-readable name
    _STAGE_NAMES = {
        0: 'Unknown/Other',
        1: 'Angel/Grant',
        2: 'Pre-Seed',
        3: 'Seed',
        4: 'Series A',
        5: 'Series B',
        6: 'Series C',
        7: 'Series D',
        8: 'Series E',
        9: 'Series F',
        10: 'Series G', 11: 'Series H', 12: 'Series I', 13: 'Series J',
        14: 'Private Equity',
        15: 'Post-IPO',
    }

    def plot_dashboard(self):
        print(f"\n[{self.name}] Generating Dashboard...")

        with plt.rc_context(_THESIS_RCPARAMS):
            fig, axes = plt.subplots(3, 3, figsize=(22, 20))
            fig.suptitle(f"{self.name} Embedding Space (K-means K={self.k})",
                         fontsize=14, fontweight='bold', y=0.995)

            # --- helpers ---------------------------------------------------
            def _style(ax, title):
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                _apply_thesis_style(ax)

            def plot_categorical(ax, labels, label_map, title, cmap='tab10'):
                """Scatter with per-category colour and legend."""
                unique_ids = np.unique(labels)
                if len(unique_ids) > 15:
                    unique_ids = unique_ids[:15]
                colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(unique_ids)))
                for i, uid in enumerate(unique_ids):
                    mask = labels == uid
                    if isinstance(label_map, (pd.Index, np.ndarray, list)):
                        name = label_map[uid] if uid < len(label_map) else str(uid)
                    else:
                        name = str(uid)
                    alpha = 0.3 if name in ('Other', 'Unknown/Other') else 0.6
                    size = 3 if name in ('Other', 'Unknown/Other') else 6
                    ax.scatter(self.proj_2d[mask, 0], self.proj_2d[mask, 1],
                               c=[colors[i]], s=size, alpha=alpha, label=name)
                ax.legend(markerscale=2.5, fontsize=7, loc='upper right',
                          ncol=1, framealpha=0.8)
                _style(ax, title)

            def plot_continuous(ax, values, title, cmap='viridis',
                               cbar_label=None):
                sc = ax.scatter(self.proj_2d[:, 0], self.proj_2d[:, 1],
                                c=values, cmap=cmap, s=5, alpha=0.6)
                cb = plt.colorbar(sc, ax=ax, shrink=0.8)
                if cbar_label:
                    cb.set_label(cbar_label, fontsize=8)
                _style(ax, title)

            def plot_binary(ax, values, title, labels_01=('No', 'Yes'),
                            colors_01=('#3274A1', '#E1812C'), grey_mask=None):
                """Binary scatter with legend. Optionally grey-out non-mask."""
                if grey_mask is not None:
                    n_grey = (~grey_mask).sum()
                    ax.scatter(self.proj_2d[~grey_mask, 0],
                               self.proj_2d[~grey_mask, 1],
                               c='lightgrey', s=2, alpha=0.1,
                               label=f'Non-mature ({n_grey:,})')
                    vals = values[grey_mask]
                    pts = self.proj_2d[grey_mask]
                else:
                    vals = values
                    pts = self.proj_2d
                for val, color, label in [(0, colors_01[0], labels_01[0]),
                                          (1, colors_01[1], labels_01[1])]:
                    m = vals == val
                    ax.scatter(pts[m, 0], pts[m, 1], c=color, s=6,
                               alpha=0.6, label=f'{label} ({m.sum():,})')
                ax.legend(markerscale=3, fontsize=8, loc='upper right',
                          framealpha=0.8)
                _style(ax, title)

            # --- Row 1: Clusters, Industry, City -------------------------
            plot_categorical(axes[0, 0], self.clusters,
                             [f"Cluster {i}" for i in range(self.k)],
                             f"K-means Clusters (K={self.k})")
            plot_categorical(axes[0, 1], self.meta_vecs['Industry'],
                             self.industry_levels, "Industry (Top 10)",
                             cmap='tab20')
            plot_categorical(axes[0, 2], self.meta_vecs['City'],
                             self.city_levels, "City (Top 10)", cmap='tab20')

            # --- Row 2: Next Funding Round, Exit (mature), Funding Stage -
            if 'Momentum' in self.meta_vecs:
                plot_binary(axes[1, 0], self.meta_vecs['Momentum'],
                            "Next Funding Round",
                            labels_01=('No', 'Yes'))

            if 'Liquidity' in self.meta_vecs:
                plot_binary(axes[1, 1], self.meta_vecs['Liquidity'],
                            "Exit (Mature Only)",
                            labels_01=('No Exit', 'Exit'),
                            grey_mask=self.mature_mask)

            # Stage with human-readable names, chronological order
            _STAGE_ORDER = [
                'Angel/Grant', 'Pre-Seed', 'Seed',
                'Series A', 'Series B', 'Series C',
                'Series D', 'Series E', 'Private Equity',
                'Post-IPO',
            ]
            raw_stages = self.meta['last_funding_stage']
            stage_names = raw_stages.apply(
                lambda x: self._STAGE_NAMES.get(x, x)
                          if isinstance(x, (int, float, np.integer))
                          else str(x))
            # Keep top 10 chronological stages, rest → Other
            stage_display = stage_names.apply(
                lambda x: x if x in _STAGE_ORDER else 'Other')
            order = [s for s in _STAGE_ORDER
                     if (stage_display == s).any()] + ['Other']
            stage_codes = stage_display.map(
                {n: i for i, n in enumerate(order)}).values
            plot_categorical(axes[1, 2], stage_codes, order,
                             "Funding Stage", cmap='tab20')

            # --- Row 3: Total Funding, Funding Rounds, Founded Year -----
            if 'total_funding_log' in self.meta.columns:
                plot_continuous(axes[2, 0], self.meta['total_funding_log'].values,
                                "Total Funding", cbar_label='log(1 + USD)')

            if 'num_funding_rounds' in self.meta.columns:
                plot_continuous(axes[2, 1],
                                np.clip(self.meta['num_funding_rounds'].values, 0, 10),
                                "Number of Funding Rounds",
                                cmap='YlOrRd', cbar_label='Rounds')
            else:
                axes[2, 1].axis('off')

            plot_continuous(axes[2, 2], self.meta['founded_on_year'].values,
                            "Founded Year", cmap='plasma',
                            cbar_label='Year')

            plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save
        plt.savefig(self.output_dir / f"dashboard_{self.name.lower()}.png",
                    dpi=150, bbox_inches='tight')
        thesis_fig_dir = self.output_dir / "figures"
        thesis_fig_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(thesis_fig_dir / f"embedding_dashboard_{self.name.lower()}.pdf",
                    dpi=300, bbox_inches='tight')
        print(f"   Saved thesis figure to "
              f"{thesis_fig_dir / f'embedding_dashboard_{self.name.lower()}.pdf'}")
        plt.close()

    def run(self, k=None, find_k=False):
        print(f"\nStarting Analysis for {self.name}...")
        self.run_dimensionality_reduction()
        
        if find_k or k is None:
            k = self.run_k_search()
        
        self.cluster(k)
        self.compute_metrics()
        self.profile_clusters()
        self.plot_dashboard()

def main():
    print("Starting embedding comparison...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--find_k", action="store_true")
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--cluster_on_2d", action="store_true", default=True)
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--state-dir", type=str, default=None, help="Path to pipeline state (graph_data.pt + models/)")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to save outputs")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--graph-path", type=str, default=None, help="Path to graph_data.pt")
    parser.add_argument("--max-metapaths", type=int, default=None, help="Override max_metapaths to match checkpoint")
    parser.add_argument("--num-hops", type=int, default=None, help="Override SeHGNN num_hops to match checkpoint")
    args = parser.parse_args()

    config = load_config()
    if args.output_dir:
        output_dir = Path(args.output_dir) / "embedding_comparison"
    else:
        output_dir = Path("outputs/embedding_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_overrides = {}
    if args.max_metapaths is not None:
        config_overrides["metapath_discovery.automatic.max_metapaths"] = args.max_metapaths
    if args.num_hops is not None:
        config_overrides["models.SeHGNN.num_hops"] = args.num_hops

    # 1. Load Data
    print("Loading Data...")
    state_dir = getattr(args, 'state_dir', None)
    trainer = load_graph_and_model(
        state_dir=state_dir,
        model_path=args.model_path,
        graph_path=args.graph_path,
        config_overrides=config_overrides or None,
    )    
    graph = trainer.data
    model = trainer.model
    
    model.eval()
    with torch.no_grad():
        output = model(graph.x_dict, graph.edge_index_dict)
        gnn_raw = output["embedding"]["startup"].cpu().numpy()
        
    # Identifiers
    node_df = graph.node_names['startup']
    startup_uuids = node_df['startup_uuid'].tolist() if 'startup_uuid' in node_df.columns else node_df.index.tolist()
    
    # Text
    csv_path = "data/embeddings_csv/org_embeddings_original_384d.csv"
    valid_text_indices, text_raw_full = load_text_embeddings(csv_path, startup_uuids)
    
    # Masks
    test_mask = graph['startup'].test_mask.cpu().numpy() if 'test_mask' in graph['startup'] else np.ones(len(startup_uuids), dtype=bool)
    
    # Intersect: Test Split AND Text Available
    combined_mask = np.zeros(len(startup_uuids), dtype=bool)
    combined_mask[valid_text_indices] = True
    
    # Use test split for consistency with K-search results
    final_mask = combined_mask & test_mask

    valid_indices = np.where(final_mask)[0]
    print(f"Final Dataset Size (Test Split): {len(valid_indices)} startups")

    gnn_vecs = normalize(gnn_raw[valid_indices])
    text_vecs = normalize(text_raw_full[final_mask[valid_text_indices]])
    
    # Load Metadata
    print("Loading Metadata...")
    raw_df = pd.read_csv("data/crunchbase/2023/organizations.csv", 
                         usecols=['uuid', 'total_funding_usd', 'status', 'city', 'founded_on', 'category_groups_list'])
    raw_df.set_index('uuid', inplace=True)
    final_uuids = [startup_uuids[i] for i in valid_indices]
    meta_df = raw_df.reindex(final_uuids)
    
    # Preprocess Metadata
    meta_df['total_funding_usd'] = meta_df['total_funding_usd'].fillna(0)
    meta_df['total_funding_log'] = np.log1p(meta_df['total_funding_usd'])
    meta_df['founded_on_year'] = pd.to_datetime(meta_df['founded_on'], errors='coerce').dt.year.fillna(0).astype(int)
    meta_df['city'] = meta_df['city'].fillna('Unknown')
    meta_df['status'] = meta_df['status'].fillna('Unknown')
    meta_df['primary_industry'] = meta_df['category_groups_list'].fillna('Unknown').apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')
    
    # Top 10 Ind/City
    top_ind = meta_df['primary_industry'].value_counts().nlargest(10).index
    meta_df['industry_top'] = meta_df['primary_industry'].apply(lambda x: x if x in top_ind else 'Other')
    
    top_city = meta_df['city'].value_counts().nlargest(10).index
    meta_df['city_top'] = meta_df['city'].apply(lambda x: x if x in top_city else 'Other')

    # Add Stage + Num Funding Rounds
    try:
        nodes_df = pd.read_csv("data/graph/startup_nodes.csv",
                               usecols=['startup_uuid', 'last_funding_stage', 'num_funding_rounds'])
        nodes_df.set_index('startup_uuid', inplace=True)
        aligned = nodes_df.reindex(meta_df.index)
        meta_df['last_funding_stage'] = aligned['last_funding_stage'].fillna('Unknown')
        meta_df['num_funding_rounds'] = aligned['num_funding_rounds'].fillna(0).astype(int)
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        meta_df['last_funding_stage'] = 'Unknown'
        meta_df['num_funding_rounds'] = 0

    # Add Targets
    y = graph['startup'].y.cpu().numpy()[valid_indices]
    meta_df['momentum'] = y[:, 0]
    meta_df['liquidity'] = y[:, 1]
    
    # 2. Run Analyzers
    
    # GNN
    analyzer_gnn = EmbeddingAnalyzer("GNN", gnn_vecs, meta_df, config, output_dir, method=args.method, cluster_on_2d=args.cluster_on_2d)
    analyzer_gnn.run(k=args.n_clusters, find_k=args.find_k)
    
    # Text
    analyzer_text = EmbeddingAnalyzer("Text", text_vecs, meta_df, config, output_dir, method=args.method, cluster_on_2d=args.cluster_on_2d)
    analyzer_text.run(k=args.n_clusters, find_k=args.find_k)

    # 3. Retrieval Overlap Analysis (GNN vs Text)
    compute_retrieval_overlap(gnn_vecs, text_vecs, output_dir)

    print("\nComparison Complete. Check 'outputs/embedding_comparison/'")

if __name__ == "__main__":
    main()
