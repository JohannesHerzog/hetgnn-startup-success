"""
Analyze GNN and text embeddings using the same loading logic as competitor_retrieval.py
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import sys
import os
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use the same loading logic as competitor_retrieval.py
from scripts.case_study import load_graph_and_model
from scripts.compare_embeddings import compute_purity_at_k, load_text_embeddings

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


def plot_pairwise_similarity(similarities, dim_stds, label, output_dir,
                             color_hist='steelblue', color_dim='coral',
                             color_zoom='crimson'):
    """Render the 3-panel pairwise similarity figure in thesis style.

    Panels:
      (a) Similarity distribution with mean/median markers
      (b) Per-dimension standard deviation histogram
      (c) Zoom on high-similarity region (>0.95)
    """
    with plt.rc_context(_THESIS_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

        # --- (a) Similarity distribution ---
        ax = axes[0]
        ax.hist(similarities, bins=100, alpha=0.75, edgecolor='white',
                linewidth=0.3, color=color_hist)
        ax.axvline(similarities.mean(), color='#d62728', linestyle='--',
                   linewidth=1.5, label=f'Mean: {similarities.mean():.3f}')
        ax.axvline(np.median(similarities), color='#ff7f0e', linestyle='--',
                   linewidth=1.5, label=f'Median: {np.median(similarities):.3f}')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title(f'(a) Pairwise Cosine Similarity ({label})')
        ax.legend(fontsize=8)
        ax.set_xlim([-1, 1])
        _apply_thesis_style(ax)

        # --- (b) Per-dimension std ---
        ax = axes[1]
        ax.hist(dim_stds, bins=30, edgecolor='white', linewidth=0.3,
                color=color_dim, alpha=0.75)
        ax.axvline(dim_stds.mean(), color='#d62728', linestyle='--',
                   linewidth=1.5, label=f'Mean: {dim_stds.mean():.3f}')
        ax.set_xlabel('Standard Deviation')
        ax.set_ylabel('Number of Dimensions')
        ax.set_title('(b) Per-Dimension Std Dev')
        ax.legend(fontsize=8)
        _apply_thesis_style(ax)

        # --- (c) High-similarity zoom ---
        ax = axes[2]
        high_sim = similarities[similarities > 0.95]
        pct = 100 * len(high_sim) / len(similarities)
        ax.hist(high_sim, bins=50, edgecolor='white', linewidth=0.3,
                color=color_zoom, alpha=0.75)
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title(
            f'(c) High Similarity (>0.95)\n'
            f'{len(high_sim):,} / {len(similarities):,} pairs ({pct:.1f}%)')
        ax.set_xlim([0.95, 1.0])
        _apply_thesis_style(ax)

        fig.tight_layout()

        # Save
        thesis_fig_dir = output_dir / "figures"
        thesis_fig_dir.mkdir(exist_ok=True, parents=True)
        slug = label.lower().replace(' ', '_').replace('-', '_')
        # Keep backward-compatible filename for GNN (thesis references it)
        suffix = '' if slug == 'gnn' else f'_{slug}'
        png_path = output_dir / f'embedding_similarity_{slug}.png'
        pdf_path = thesis_fig_dir / f'embedding_similarity_analysis{suffix}.pdf'
        fig.savefig(png_path, dpi=150)
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"Saved {label} similarity figure to {pdf_path}")
        return pdf_path

# Parse CLI args
parser = argparse.ArgumentParser(description="Analyze GNN embeddings")
parser.add_argument("--state-dir", type=str, default=None, help="Path to pipeline state (graph_data.pt + models/)")
parser.add_argument("--output-dir", type=str, default=None, help="Path to save outputs")
parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--graph-path", type=str, default=None, help="Path to graph_data.pt")
parser.add_argument("--max-metapaths", type=int, default=None, help="Override max_metapaths to match checkpoint")
parser.add_argument("--num-hops", type=int, default=None, help="Override SeHGNN num_hops to match checkpoint")
_args = parser.parse_args()

state_dir = _args.state_dir or "outputs/pipeline_state"

config_overrides = {}
if _args.max_metapaths is not None:
    config_overrides["metapath_discovery.automatic.max_metapaths"] = _args.max_metapaths
if _args.num_hops is not None:
    config_overrides["models.SeHGNN.num_hops"] = _args.num_hops

print("Loading graph and model...")
trainer = load_graph_and_model(
    state_dir=state_dir,
    model_path=_args.model_path,
    graph_path=_args.graph_path,
    config_overrides=config_overrides or None,
)
graph = trainer.data
model = trainer.model
config = trainer.config

print("Running inference to get embeddings...")
model.eval()
device = next(model.parameters()).device

with torch.no_grad():
    output = model(graph.x_dict, graph.edge_index_dict)
    embeddings = output["embedding"]["startup"].cpu().numpy()

print("=" * 80)
print("EMBEDDING STATISTICS")
print("=" * 80)
print(f"Shape: {embeddings.shape}")
print(f"Mean: {embeddings.mean():.6f}")
print(f"Std: {embeddings.std():.6f}")
print(f"Min: {embeddings.min():.6f}")
print(f"Max: {embeddings.max():.6f}")
print()

# Per-dimension statistics
print("PER-DIMENSION STATISTICS:")
dim_means = embeddings.mean(axis=0)
dim_stds = embeddings.std(axis=0)
print(f"Mean of dimension means: {dim_means.mean():.6f}")
print(f"Std of dimension means: {dim_means.std():.6f}")
print(f"Mean of dimension stds: {dim_stds.mean():.6f}")
print(f"Std of dimension stds: {dim_stds.std():.6f}")
print()

# Check for collapsed dimensions
collapsed_dims = (dim_stds < 0.01).sum()
print(f"Collapsed dimensions (std < 0.01): {collapsed_dims} / {embeddings.shape[1]}")
print()

# Pairwise cosine similarity sample
sample_size = min(5000, embeddings.shape[0])
np.random.seed(42)
sample_idx = np.random.choice(embeddings.shape[0], sample_size, replace=False)
sample_emb = embeddings[sample_idx]

# Normalize
sample_norm = normalize(sample_emb, axis=1)

# Compute pairwise similarities
print("Computing pairwise similarities...")
sim_matrix = cosine_similarity(sample_norm)

# Get upper triangle (excluding diagonal)
upper_tri = np.triu_indices_from(sim_matrix, k=1)
similarities = sim_matrix[upper_tri]

print()
print(f"PAIRWISE COSINE SIMILARITY (sample of {sample_size} startups):")
print(f"Mean: {similarities.mean():.6f}")
print(f"Std: {similarities.std():.6f}")
print(f"Min: {similarities.min():.6f}")
print(f"Max: {similarities.max():.6f}")
print(f"Median: {np.median(similarities):.6f}")
print()

# Distribution percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("Similarity Percentiles:")
for p in percentiles:
    val = np.percentile(similarities, p)
    print(f"  {p:3d}%: {val:.6f}")
print()

# Check for mode collapse
high_sim_count = (similarities > 0.99).sum()
very_high_sim_count = (similarities > 0.999).sum()
perfect_sim_count = (similarities >= 0.9999).sum()
total_pairs = len(similarities)
print(f"Pairs with similarity > 0.99:   {high_sim_count:6d} / {total_pairs} ({100*high_sim_count/total_pairs:5.2f}%)")
print(f"Pairs with similarity > 0.999:  {very_high_sim_count:6d} / {total_pairs} ({100*very_high_sim_count/total_pairs:5.2f}%)")
print(f"Pairs with similarity >= 0.9999: {perfect_sim_count:6d} / {total_pairs} ({100*perfect_sim_count/total_pairs:5.2f}%)")
print()

# Robust path handling
project_root = Path(__file__).resolve().parent.parent
if _args.output_dir:
    output_dir = Path(_args.output_dir)
else:
    output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True, parents=True)

# --- Text pairwise similarity figure (same layout) ---
print("\n" + "=" * 80)
print("TEXT EMBEDDING PAIRWISE SIMILARITY")
print("=" * 80)

node_df = graph.node_names['startup']
startup_uuids = node_df['startup_uuid'].tolist() if 'startup_uuid' in node_df.columns else node_df.index.tolist()

csv_path = "data/embeddings_csv/org_embeddings_original_384d.csv"
valid_text_indices, text_raw = load_text_embeddings(csv_path, startup_uuids)

# Sample from nodes that have text embeddings
text_sample_size = min(5000, len(valid_text_indices))
np.random.seed(42)
text_sample_pos = np.random.choice(len(valid_text_indices), text_sample_size, replace=False)
text_sample_emb = text_raw[text_sample_pos]
text_sample_norm = normalize(text_sample_emb, axis=1)

# Pairwise similarities
print(f"Computing text pairwise similarities ({text_sample_size} nodes)...")
text_sim_matrix = cosine_similarity(text_sample_norm)
text_upper_tri = np.triu_indices_from(text_sim_matrix, k=1)
text_similarities = text_sim_matrix[text_upper_tri]

print(f"TEXT PAIRWISE COSINE SIMILARITY (sample of {text_sample_size} startups):")
print(f"Mean: {text_similarities.mean():.6f}")
print(f"Std: {text_similarities.std():.6f}")
print(f"Min: {text_similarities.min():.6f}")
print(f"Max: {text_similarities.max():.6f}")
print(f"Median: {np.median(text_similarities):.6f}")

# Per-dimension stats
text_dim_stds = text_sample_emb.std(axis=0)
text_collapsed = (text_dim_stds < 0.01).sum()
print(f"Collapsed dimensions (std < 0.01): {text_collapsed} / {text_sample_emb.shape[1]}")

# Percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("Text Similarity Percentiles:")
for p in percentiles:
    val = np.percentile(text_similarities, p)
    print(f"  {p:3d}%: {val:.6f}")

# High-similarity counts
high_sim_count = (text_similarities > 0.99).sum()
total_pairs = len(text_similarities)
print(f"Pairs with similarity > 0.99:  {high_sim_count:6d} / {total_pairs} ({100*high_sim_count/total_pairs:5.2f}%)")

# --- GNN pairwise similarity figure (thesis-styled) ---
plot_pairwise_similarity(similarities, dim_stds, "GNN", output_dir)
plot_pairwise_similarity(text_similarities, text_dim_stds, "Text", output_dir)

# ------------------------------------------------------------------------------
# CLUSTER VISUALIZATION (UMAP) — 2x3 Multi-Panel
# ------------------------------------------------------------------------------
try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False

print("-" * 80)
method_name = 'UMAP' if HAS_UMAP else 't-SNE'
print(f"CLUSTER ANALYSIS ({method_name})")
print("-" * 80)

# --- Load Metadata for Coloring ---
print("Loading metadata for visualization...")

# Node UUIDs
node_df = graph.node_names['startup']
startup_uuids = node_df['startup_uuid'].tolist() if 'startup_uuid' in node_df.columns else node_df.index.tolist()
sample_uuids = [startup_uuids[i] for i in sample_idx]

# Organizations CSV → funding, city, industry
raw_df = pd.read_csv("data/crunchbase/2023/organizations.csv",
                     usecols=['uuid', 'total_funding_usd', 'city', 'category_groups_list'])
raw_df.set_index('uuid', inplace=True)
meta_df = raw_df.reindex(sample_uuids)

meta_df['total_funding_usd'] = meta_df['total_funding_usd'].fillna(0)
meta_df['total_funding_log'] = np.log1p(meta_df['total_funding_usd'])
meta_df['city'] = meta_df['city'].fillna('Unknown')
meta_df['primary_industry'] = meta_df['category_groups_list'].fillna('Unknown').apply(
    lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')

# Top-10 filtering for industry and city
top_ind = meta_df['primary_industry'].value_counts().nlargest(10).index
meta_df['industry_top'] = meta_df['primary_industry'].apply(lambda x: x if x in top_ind else 'Other')

top_city = meta_df['city'].value_counts().nlargest(10).index
meta_df['city_top'] = meta_df['city'].apply(lambda x: x if x in top_city else 'Other')

# Startup nodes CSV → funding stage
try:
    nodes_df = pd.read_csv("data/graph/startup_nodes.csv", usecols=['startup_uuid', 'last_funding_stage'])
    nodes_df.set_index('startup_uuid', inplace=True)
    meta_df['last_funding_stage'] = nodes_df.reindex(meta_df.index)['last_funding_stage'].fillna('Unknown')
except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
    meta_df['last_funding_stage'] = 'Unknown'

# Targets → momentum, liquidity
y_all = graph['startup'].y.cpu().numpy()
meta_df['momentum'] = y_all[sample_idx, 0].astype(int)
meta_df['liquidity'] = y_all[sample_idx, 1].astype(int)

print(f"  Metadata loaded for {len(meta_df)} samples.")

# --- Dimensionality Reduction ---
if HAS_UMAP:
    print(f"Running UMAP (n_neighbors=15, min_dist=0.1, metric='cosine') on {sample_size} samples...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
else:
    print(f"Running t-SNE (perplexity=30, metric='cosine') on {sample_size} samples...")
    reducer = TSNE(n_components=2, random_state=42, metric='cosine', perplexity=30, init='pca', learning_rate='auto')

proj_2d = reducer.fit_transform(sample_norm)

# --- Stage name mapping (numeric code → human-readable) ---
_STAGE_NAMES = {
    0: 'Unknown/Other', 1: 'Angel/Grant', 2: 'Pre-Seed', 3: 'Seed',
    4: 'Series A', 5: 'Series B', 6: 'Series C', 7: 'Series D',
    8: 'Series E', 9: 'Series F', 10: 'Series G', 11: 'Series H',
    12: 'Series I', 13: 'Series J', 14: 'Private Equity', 15: 'Post-IPO',
}

# Map numeric stage codes to names
meta_df['stage_named'] = meta_df['last_funding_stage'].apply(
    lambda x: _STAGE_NAMES.get(x, x) if isinstance(x, (int, float, np.integer)) else str(x))
# Group rare stages (<1%) into "Other"
_stage_counts = meta_df['stage_named'].value_counts(normalize=True)
_rare_stages = _stage_counts[_stage_counts < 0.01].index
meta_df['stage_display'] = meta_df['stage_named'].apply(
    lambda x: 'Other' if x in _rare_stages else x)

# --- 2x3 Multi-Panel Figure (thesis-styled) ---
with plt.rc_context(_THESIS_RCPARAMS):
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(f'{method_name} of GNN Embeddings (n={sample_size:,})',
                 fontsize=14, fontweight='bold', y=0.995)

    def _style_umap(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        _apply_thesis_style(ax)

    def plot_categorical(ax, labels, title, cmap='tab10'):
        """Plot categorical feature with legend."""
        unique_labels = sorted(labels.unique(), key=lambda x: (x in ('Other', 'Unknown/Other'), x))
        colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            mask = labels.values == label
            alpha = 0.3 if label in ('Other', 'Unknown/Other') else 0.6
            size = 3 if label in ('Other', 'Unknown/Other') else 6
            ax.scatter(proj_2d[mask, 0], proj_2d[mask, 1],
                       c=[colors[i]], s=size, alpha=alpha, label=label)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(markerscale=2.5, fontsize=7, loc='upper right',
                  ncol=1, framealpha=0.8)
        _style_umap(ax)

    def plot_continuous(ax, values, title, cmap='viridis'):
        """Plot continuous feature with colorbar."""
        sc = ax.scatter(proj_2d[:, 0], proj_2d[:, 1],
                        c=values, cmap=cmap, s=5, alpha=0.6)
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        _style_umap(ax)

    def plot_binary(ax, values, title, labels_01=('No', 'Yes')):
        """Plot binary feature with two colors and legend."""
        for val, color, label in [(0, '#3274A1', labels_01[0]),
                                  (1, '#E1812C', labels_01[1])]:
            mask = values == val
            ax.scatter(proj_2d[mask, 0], proj_2d[mask, 1],
                       c=color, s=5, alpha=0.6,
                       label=f'{label} ({mask.sum():,})')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(markerscale=3, fontsize=8, loc='upper right', framealpha=0.8)
        _style_umap(ax)

    # Row 1: Funding, Stage (named), Industry
    plot_continuous(axes[0, 0], meta_df['total_funding_log'].values,
                    'Total Funding (Log USD)')
    plot_categorical(axes[0, 1], meta_df['stage_display'],
                     'Funding Stage', cmap='tab20')
    plot_categorical(axes[0, 2], meta_df['industry_top'],
                     'Industry (Top 10)', cmap='tab20')

    # Row 2: City, Next Funding Round, Exit
    plot_categorical(axes[1, 0], meta_df['city_top'],
                     'City (Top 10)', cmap='tab20')
    plot_binary(axes[1, 1], meta_df['momentum'].values,
                'Next Funding Round', labels_01=('No', 'Yes'))
    plot_binary(axes[1, 2], meta_df['liquidity'].values,
                'Exit', labels_01=('No Exit', 'Exit'))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

thesis_fig_dir2 = output_dir / "figures"
thesis_fig_dir2.mkdir(exist_ok=True, parents=True)
umap_path = output_dir / 'embedding_umap_multipanel.png'
plt.savefig(umap_path, dpi=150, bbox_inches='tight')
plt.savefig(thesis_fig_dir2 / 'embedding_umap_multipanel.pdf', dpi=300, bbox_inches='tight')
print(f"Saved multi-panel UMAP to {umap_path}")
print(f"Saved thesis figure to {thesis_fig_dir2 / 'embedding_umap_multipanel.pdf'}")
plt.close()
print("-" * 80)

# ------------------------------------------------------------------------------
# CLUSTERING METRICS (QUANTITATIVE)
# ------------------------------------------------------------------------------
from sklearn.metrics import silhouette_score

print("\nQUANTITATIVE CLUSTERING METRICS:")

# Use funding stage as the primary label for clustering quality
stage_labels, stage_levels = pd.factorize(meta_df['last_funding_stage'])
num_stage_classes = len(stage_levels)

try:
    sil_score = silhouette_score(sample_norm, stage_labels, metric='cosine')
    print(f"  Silhouette Score (by Stage): {sil_score:.4f} (Range -1 to 1, higher is better)")
    print("    > 0.5: Strong structure | > 0.2: Weak structure | < 0.1: No structure")
except Exception as e:
    print(f"  Could not compute Silhouette: {e}")

# Neighborhood Purity@K using shared function
label_sets = {
    'Stage': stage_labels,
    'Industry': pd.factorize(meta_df['industry_top'])[0],
    'City': pd.factorize(meta_df['city_top'])[0],
    'Momentum': meta_df['momentum'].values,
    'Liquidity': meta_df['liquidity'].values,
}

purity_results = compute_purity_at_k(sample_norm, label_sets)
print("\n  Neighborhood Purity@K:")
for feat_name, scores in purity_results.items():
    n_classes = scores['n_classes']
    chance = scores['chance']
    print(f"\n    {feat_name} ({n_classes} classes):")
    for k in (1, 5, 10, 20):
        print(f"      Purity@{k}: {scores[k]:.4f} (Chance: {chance:.4f})")

print("-" * 80)
