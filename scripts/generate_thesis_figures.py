"""
Generate publication-quality figures for thesis Section 5.1 (Graph Data Analysis).

Figures:
  1. Homophily Spectrum — metapath lollipop chart
  2. Degree-Success Nexus — dual-panel degree vs. outcome
  3. Feature Sparsity Funnel — grouped bar chart of startup feature coverage
  4. Graph Schema — heterogeneous graph structure diagram

Usage:
  python scripts/generate_thesis_figures.py [--skip-degree]
"""

import sys
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "HGNN_Thesis" / "figures"
STATS_DIR = PROJECT_ROOT / "outputs" / "graph_statistics"

# Publication style
plt.rcParams.update({
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
})


# ============================================================
# Figure 1: Homophily Spectrum
# ============================================================

# Semantic categories for metapath names
METAPATH_CATEGORIES = {
    "city_peers": "Geographic",
    "sector_peers": "Industry",
    "alumni_network": "Academic",
    "co_working_network": "Professional",
}

CATEGORY_COLORS = {
    "Geographic": "#9467bd",
    "Industry": "#7f7f7f",
    "Academic": "#17becf",
    "Professional": "#bcbd22",
    "Early-Stage": "#2ca02c",
    "Mid-Stage": "#ff7f0e",
    "Late-Stage": "#d62728",
}

# Pretty labels for metapaths
METAPATH_LABELS = {
    "city_peers": "City Peers",
    "sector_peers": "Sector Peers",
    "alumni_network": "Alumni Network",
    "co_working_network": "Co-Working Network",
    "early_portfolio_siblings": "Early Portfolio Siblings",
    "early_founder_vc_employment": "Early Founder-VC Employment",
    "early_founder_coworking_investor": "Early Founder Co-Working → Inv.",
    "early_investor_alumni": "Early Investor Alumni",
    "early_alumni_investor_network": "Early Alumni-Investor Net.",
    "early_investor_founder_coworking": "Early Investor-Founder Co-Work.",
    "early_founder_coworking_syndicate": "Early Founder Co-Work. Syndicate",
    "mid_portfolio_siblings": "Mid Portfolio Siblings",
    "mid_founder_vc_employment": "Mid Founder-VC Employment",
    "mid_founder_coworking_investor": "Mid Founder Co-Working → Inv.",
    "mid_investor_alumni": "Mid Investor Alumni",
    "mid_alumni_investor_network": "Mid Alumni-Investor Net.",
    "mid_investor_founder_coworking": "Mid Investor-Founder Co-Work.",
    "mid_founder_coworking_syndicate": "Mid Founder Co-Work. Syndicate",
    "late_portfolio_siblings": "Late Portfolio Siblings",
    "late_founder_vc_employment": "Late Founder-VC Employment",
    "late_founder_coworking_investor": "Late Founder Co-Working → Inv.",
    "late_investor_alumni": "Late Investor Alumni",
    "late_alumni_investor_network": "Late Alumni-Investor Net.",
    "late_investor_founder_coworking": "Late Investor-Founder Co-Work.",
    "late_founder_coworking_syndicate": "Late Founder Co-Work. Syndicate",
}


def _categorize(name):
    if name in METAPATH_CATEGORIES:
        return METAPATH_CATEGORIES[name]
    if name.startswith("early_"):
        return "Early-Stage"
    if name.startswith("mid_"):
        return "Mid-Stage"
    if name.startswith("late_"):
        return "Late-Stage"
    return "Other"


def figure_homophily_spectrum():
    """Figure 1: horizontal lollipop chart of metapath homophily."""
    csv_path = STATS_DIR / "homophily_metrics.csv"
    df = pd.read_csv(csv_path)

    # Sort by homophily descending
    df = df.sort_values("MSHR (Edge Homophily)", ascending=True).reset_index(drop=True)

    # Categorize and label
    df["category"] = df["Relation"].map(_categorize)
    df["label"] = df["Relation"].map(lambda x: METAPATH_LABELS.get(x, x))
    df["color"] = df["category"].map(CATEGORY_COLORS)
    df["h"] = df["MSHR (Edge Homophily)"]
    df["log_edges"] = np.log10(df["Num Edges"].clip(lower=1))

    # Random baseline: h = p_pos^2 + p_neg^2 with p_pos = 0.127
    h_random = 0.127**2 + 0.873**2  # ≈ 0.778

    fig, ax = plt.subplots(figsize=(7, 8))

    # Lollipop stems
    for i, row in df.iterrows():
        ax.hlines(y=i, xmin=h_random, xmax=row["h"], color=row["color"],
                  alpha=0.5, linewidth=1.2)

    # Dots (size proportional to log edge count)
    sizes = 20 + (df["log_edges"] - df["log_edges"].min()) / \
            (df["log_edges"].max() - df["log_edges"].min()) * 120
    ax.scatter(df["h"], range(len(df)), c=df["color"], s=sizes,
               zorder=5, edgecolors="white", linewidth=0.5)

    # Random baseline
    ax.axvline(x=h_random, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(h_random - 0.008, len(df) + 0.5,
            f"Random baseline ($h = {h_random:.3f}$)",
            fontsize=8, va="bottom", ha="right", style="italic")

    # Shaded regions
    ax.axvspan(0.3, h_random, alpha=0.04, color="red")
    ax.axvspan(h_random, 0.85, alpha=0.04, color="green")
    ax.text(0.42, -1.8, "← Heterophilic", fontsize=8, color="#b22222",
            ha="center", style="italic")
    ax.text(0.82, -1.8, "Homophilic →", fontsize=8, color="#228b22",
            ha="center", style="italic")

    # Y-axis labels
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=8)

    ax.set_xlabel("Edge Homophily Ratio ($h$)")
    ax.set_xlim(0.3, 0.85)
    ax.set_ylim(-2.5, len(df) + 1.5)

    # Legend for categories
    legend_handles = [mpatches.Patch(color=c, label=cat)
                      for cat, c in CATEGORY_COLORS.items()
                      if cat in df["category"].values]
    # Size legend
    size_small = mlines.Line2D([], [], color="gray", marker="o", linestyle="",
                                markersize=5, label="~100 edges")
    size_large = mlines.Line2D([], [], color="gray", marker="o", linestyle="",
                                markersize=11, label="~1M edges")
    legend_handles.extend([size_small, size_large])
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7,
              framealpha=0.9, ncol=2, bbox_to_anchor=(0.0, 1.0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / "homophily_spectrum.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Figure 2: Degree-Success Nexus
# ============================================================

def figure_degree_success():
    """Figure 2: dual-panel — KDE of degree by class + success rate by decile."""
    import torch
    import gc

    graph_path = STATS_DIR / "graph_data_full.pt"
    print("  Loading graph (extracting degrees only)...")
    data = torch.load(graph_path, weights_only=False)

    y = data["startup"].y[:, 0].numpy().copy()
    n_startups = len(y)

    # Compute total degree across all edge types
    total_degree = np.zeros(n_startups, dtype=np.int64)
    for etype in data.edge_types:
        src_type, rel, dst_type = etype
        ei = data[etype].edge_index
        if ei.shape[1] == 0:
            continue
        if src_type == "startup":
            np.add.at(total_degree, ei[0].numpy(), 1)
        if dst_type == "startup":
            np.add.at(total_degree, ei[1].numpy(), 1)

    del data
    gc.collect()

    pos_deg = total_degree[y == 1]
    neg_deg = total_degree[y == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # --- Left panel: KDE ---
    from scipy.stats import gaussian_kde

    # Use log-transformed degrees for KDE
    log_neg = np.log10(neg_deg.clip(min=1).astype(float))
    log_pos = np.log10(pos_deg.clip(min=1).astype(float))

    x_range = np.linspace(0, np.log10(500), 300)
    kde_neg = gaussian_kde(log_neg, bw_method=0.15)
    kde_pos = gaussian_kde(log_pos, bw_method=0.15)

    ax1.fill_between(x_range, kde_neg(x_range), alpha=0.35, color="#4878cf",
                     label=f"Negative (n={len(neg_deg):,})")
    ax1.fill_between(x_range, kde_pos(x_range), alpha=0.35, color="#e24a33",
                     label=f"Positive (n={len(pos_deg):,})")
    ax1.plot(x_range, kde_neg(x_range), color="#4878cf", linewidth=1.5)
    ax1.plot(x_range, kde_pos(x_range), color="#e24a33", linewidth=1.5)

    # Mean lines
    ax1.axvline(np.log10(neg_deg.mean()), color="#4878cf", linestyle="--",
                linewidth=1, alpha=0.8)
    ax1.axvline(np.log10(pos_deg.mean()), color="#e24a33", linestyle="--",
                linewidth=1, alpha=0.8)
    ax1.text(np.log10(neg_deg.mean()) - 0.03, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1.5,
             f"μ={neg_deg.mean():.1f}", fontsize=7, color="#4878cf", ha="right",
             va="top", rotation=90)
    ax1.text(np.log10(pos_deg.mean()) + 0.03, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1.5,
             f"μ={pos_deg.mean():.1f}", fontsize=7, color="#e24a33", ha="left",
             va="top", rotation=90)

    # Custom x-axis ticks (log scale labels)
    tick_vals = [1, 2, 5, 10, 20, 50, 100, 200, 400]
    ax1.set_xticks([np.log10(v) for v in tick_vals])
    ax1.set_xticklabels(tick_vals)
    ax1.set_xlabel("Total Node Degree")
    ax1.set_ylabel("Relative Frequency")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("(a) Degree Distribution by Class", fontsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right panel: Success rate by degree decile ---
    deciles = pd.qcut(total_degree, 10, labels=False, duplicates="drop")
    n_bins = deciles.max() + 1

    bin_rates = []
    bin_labels = []
    for d in range(n_bins):
        mask = deciles == d
        rate = y[mask].mean() * 100
        deg_min = total_degree[mask].min()
        deg_max = total_degree[mask].max()
        bin_rates.append(rate)
        bin_labels.append(f"{deg_min}-{deg_max}")

    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_bins))
    bars = ax2.bar(range(n_bins), bin_rates, color=colors, edgecolor="white",
                   linewidth=0.5)

    # Base rate reference
    base_rate = y.mean() * 100
    ax2.axhline(base_rate, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax2.text(n_bins - 0.5, base_rate + 0.5, f"Base rate {base_rate:.1f}%",
             fontsize=7, ha="right", style="italic")

    # Annotate top/bottom bars
    ax2.text(0, bin_rates[0] + 0.3, f"{bin_rates[0]:.1f}%", ha="center",
             fontsize=7, fontweight="bold")
    ax2.text(n_bins - 1, bin_rates[-1] + 0.3, f"{bin_rates[-1]:.1f}%",
             ha="center", fontsize=7, fontweight="bold")

    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f"D{i+1}" for i in range(n_bins)], fontsize=8)
    ax2.set_xlabel("Degree Decile (D1=lowest)")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("(b) Success Rate by Degree Decile", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=3)
    out = OUTPUT_DIR / "degree_success_nexus.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Figure 3: Feature Sparsity Funnel
# ============================================================

# Semantic groupings for startup features
FEATURE_GROUPS = {
    "Funding Rounds": [
        ("Seed", "seed_round"),
        ("Angel", "angel_round"),
        ("Venture", "venture_round"),
        ("Series A", "series_a_round"),
        ("Series B", "series_b_round"),
        ("Series C", "series_c_round"),
        ("Series D", "series_d_round"),
        ("Series E", "series_e_round"),
        ("Series F", "series_f_round"),
        ("Series G", "series_g_round"),
        ("Priv. Equity", "private_equity_round"),
    ],
    "Funding Amounts": [
        ("Seed $", "money_seed"),
        ("Angel $", "money_angel"),
        ("Venture $", "money_venture_round"),
        ("Series A $", "money_series_a"),
        ("Series B $", "money_series_b"),
        ("Series C $", "money_series_c"),
        ("Series D $", "money_series_d"),
    ],
    "Team & Education": [
        ("# Founders", "founder_count"),
        ("Serial Founders", "serial_founder_count"),
        ("Bachelor's", "bachelor_count"),
        ("Master's", "master_count"),
        ("Business Deg.", "business_degree_count"),
        ("PhD", "phd_count"),
        ("STEM Degree", "stem_degree_count"),
        ("Ivy League", "ivy_league_plus_count"),
    ],
    "Financial Quality": [
        ("Total Funding", "total_funding_usd"),
        ("# Rounds", "num_funding_rounds"),
        ("Employees", "employee_count"),
        ("Top-10% Inv.", "top_10_percent_investor_count"),
        ("# Exits", "num_exits"),
        ("Acquirer", "acquirer"),
    ],
    "Online Presence": [
        ("Email", "has_email"),
        ("Twitter", "has_twitter_url"),
        ("Facebook", "has_facebook_url"),
        ("LinkedIn", "has_linkedin_url"),
        ("Homepage", "has_domain"),
    ],
}


def figure_feature_sparsity():
    """Figure 3: grouped horizontal bar chart of startup feature coverage."""
    json_path = STATS_DIR / "graph_statistics_full.json"
    with open(json_path) as f:
        stats = json.load(f)

    per_feature = stats["feature_coverage"]["startup"]["per_feature"]

    fig, ax = plt.subplots(figsize=(7, 9))

    y_pos = 0
    y_ticks = []
    y_labels = []
    group_positions = []
    colors_list = []

    group_colors = {
        "Funding Rounds": "#e24a33",
        "Funding Amounts": "#e8855e",
        "Team & Education": "#4878cf",
        "Financial Quality": "#2ca02c",
        "Online Presence": "#9467bd",
    }

    for group_name, features in FEATURE_GROUPS.items():
        group_start = y_pos
        for display_name, feat_key in features:
            coverage = per_feature.get(feat_key, {}).get("pct_non_zero", 0)
            color = group_colors[group_name]
            ax.barh(y_pos, coverage, height=0.7, color=color, alpha=0.8,
                    edgecolor="white", linewidth=0.3)

            # Annotate sparse features
            if coverage < 10:
                ax.text(coverage + 0.5, y_pos, f"{coverage:.1f}%",
                        va="center", fontsize=6.5, color="#555")
            elif coverage > 0:
                ax.text(coverage + 0.5, y_pos, f"{coverage:.0f}%",
                        va="center", fontsize=6.5, color="#555")

            y_ticks.append(y_pos)
            y_labels.append(display_name)
            colors_list.append(color)
            y_pos += 1

        group_end = y_pos - 1
        group_positions.append((group_name, (group_start + group_end) / 2,
                                group_start, group_end))
        y_pos += 0.8  # Gap between groups

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("% Non-Zero (after imputation)")
    ax.set_xlim(0, 105)

    # Group labels on right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([pos for _, pos, _, _ in group_positions])
    ax2.set_yticklabels([name for name, _, _, _ in group_positions],
                        fontsize=8, fontweight="bold")

    # Vertical reference lines
    ax.axvline(x=50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(x=10, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    out = OUTPUT_DIR / "feature_sparsity_funnel.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Figure 4: Graph Schema
# ============================================================

def figure_graph_schema():
    """Figure 4: heterogeneous graph structure diagram."""
    json_path = STATS_DIR / "graph_statistics_full.json"
    with open(json_path) as f:
        stats = json.load(f)

    node_stats = stats["node_stats"]

    # Node layout (manually positioned for clean diagram)
    node_layout = {
        "startup":    (0.50, 0.50),
        "investor":   (0.10, 0.85),
        "founder":    (0.90, 0.85),
        "sector":     (0.10, 0.15),
        "city":       (0.50, 0.00),
        "university": (0.90, 0.15),
    }

    node_colors = {
        "startup":    "#d62728",
        "investor":   "#1f77b4",
        "founder":    "#ff7f0e",
        "city":       "#2ca02c",
        "university": "#9467bd",
        "sector":     "#7f7f7f",
    }

    # Edge definitions (semantic aggregations, one direction)
    edges = [
        ("startup", "investor", "funded_by\n330K", 3.5),
        ("startup", "founder", "team\n288K", 3.0),
        ("startup", "sector", "in_sector\n566K", 4.0),
        ("founder", "investor", "professional\n189K", 2.0),
        ("founder", "university", "studied_at\n335K", 3.0),
        ("university", "city", "based_in\n25K", 1.0),
    ]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.12, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges first (behind nodes)
    for src, dst, label, width in edges:
        x1, y1 = node_layout[src]
        x2, y2 = node_layout[dst]
        # Midpoint for label
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", color="#888",
                                    lw=width * 0.6, alpha=0.6,
                                    connectionstyle="arc3,rad=0.05"))

        # Edge label with white background
        ax.text(mx, my, label, fontsize=6.5, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.85))

    # Draw nodes
    for ntype, (x, y) in node_layout.items():
        count = node_stats[ntype]["count"]
        n_feat = node_stats[ntype]["features"]

        # Node size proportional to log(count)
        radius = 0.04 + 0.02 * (np.log10(count) - 2)

        circle = plt.Circle((x, y), radius, color=node_colors[ntype],
                             alpha=0.85, zorder=10)
        ax.add_patch(circle)

        # Node label (bold, white text inside circle)
        ax.text(x, y + 0.005, ntype.capitalize(), fontsize=8,
                fontweight="bold", ha="center", va="center", color="white",
                zorder=11)

        # Stats below
        if count >= 1000:
            count_str = f"{count / 1000:.0f}K" if count < 1_000_000 else f"{count / 1_000_000:.1f}M"
        else:
            count_str = str(count)

        ax.text(x, y - radius - 0.025, f"{count_str} nodes · {n_feat} feat.",
                fontsize=7, ha="center", va="top", color="#444")

    # Startup self-loop annotation (similarity edges)
    sx, sy = node_layout["startup"]
    ax.annotate("", xy=(sx + 0.06, sy - 0.04), xytext=(sx + 0.06, sy + 0.04),
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8,
                                connectionstyle="arc3,rad=-1.5"))
    ax.text(sx + 0.14, sy, "similarity\n1.5K", fontsize=6.5,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85))

    # Title annotation
    total_nodes = sum(v["count"] for v in node_stats.values())
    ax.text(0.5, 0.97, f"Heterogeneous Venture Graph — {total_nodes:,} nodes, ~2M edges",
            fontsize=10, ha="center", va="top", fontweight="bold",
            transform=ax.transAxes)

    out = OUTPUT_DIR / "graph_schema.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures for Section 5.1")
    parser.add_argument("--skip-degree", action="store_true",
                        help="Skip Figure 2 (requires loading 2.2GB graph)")
    parser.add_argument("--only", type=int, choices=[1, 2, 3, 4],
                        help="Generate only a specific figure")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figures = {
        1: ("Homophily Spectrum", figure_homophily_spectrum),
        2: ("Degree-Success Nexus", figure_degree_success),
        3: ("Feature Sparsity Funnel", figure_feature_sparsity),
        4: ("Graph Schema", figure_graph_schema),
    }

    for num, (name, func) in figures.items():
        if args.only and args.only != num:
            continue
        if num == 2 and args.skip_degree:
            print(f"Figure {num}: {name} — SKIPPED (--skip-degree)")
            continue
        print(f"Figure {num}: {name}")
        func()

    print("\nDone. Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
