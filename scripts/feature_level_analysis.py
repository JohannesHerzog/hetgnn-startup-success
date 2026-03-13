#!/usr/bin/env python3
"""
Feature Information Level Ablation — Analysis.

Compares model performance across two feature levels:
  - "intrinsic": only entity-intrinsic features (~84 dims)
  - "full":      all features including relational aggregations (~184 dims)

Loads results from JSON exports (outputs/results/) and generates:
  1. Summary table (mean +/- std across seeds)
  2. Bar chart comparing models across levels
  3. LaTeX table for thesis

Usage:
    python scripts/feature_level_analysis.py
    python scripts/feature_level_analysis.py --results-dir outputs/results --latex
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication style (matches generate_thesis_figures.py)
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

# Model order: tabular → homogeneous GNN → heterogeneous GNN
# This ordering visually reveals the robustness gradient
ALL_MODELS = ["XGBoost", "MLP", "GCN", "SageGNN", "HAN", "SeHGNN"]

# Display names for thesis
MODEL_DISPLAY = {
    "XGBoost": "XGBoost",
    "MLP": "MLP",
    "GCN": "GCN",
    "SageGNN": "GraphSAGE",
    "HAN": "HAN",
    "SeHGNN": "SeHGNN",
}


def load_all_results(results_dir: str = "outputs/results") -> pd.DataFrame:
    """Load all JSON result files into a DataFrame."""
    records = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if not f.endswith(".json") or f.endswith("_error.json"):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r") as fp:
                    data = json.load(fp)
                record = {}
                for k, v in data.get("metadata", {}).items():
                    record[f"meta.{k}"] = v
                record.update(_flatten_dict(data.get("config", {}), prefix="config"))
                for k, v in data.get("metrics", {}).items():
                    if isinstance(v, (int, float)):
                        record[f"metric.{k}"] = v
                record["_source_file"] = path
                records.append(record)
            except (json.JSONDecodeError, KeyError):
                pass
    return pd.DataFrame(records) if records else pd.DataFrame()


def _flatten_dict(d, prefix="", sep="."):
    """Flatten nested dict with dot-separated keys."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key, sep))
        else:
            items[key] = v
    return items


def get_feature_level(row):
    """Extract feature level from config columns."""
    level = row.get("config.data_processing.ablation.feature_information_level")
    if level == "intrinsic":
        return "intrinsic"
    return "full"


def analyze(df: pd.DataFrame, metric: str = "metric.test_auc_pr_mom",
            models: list = None):
    """Analyze results by model and feature level."""
    df = df.copy()
    df["feature_level"] = df.apply(get_feature_level, axis=1)
    df["model"] = df.get("config.train.model", df.get("meta.model"))

    # Filter to relevant models
    if models is None:
        models = ALL_MODELS
    df = df[df["model"].isin(models)]

    if metric not in df.columns:
        # Try alternative metric names
        alternatives = [c for c in df.columns if "auc_pr" in c and "mom" in c]
        if alternatives:
            metric = alternatives[0]
            print(f"  Using metric: {metric}")
        else:
            print(f"  ERROR: Metric '{metric}' not found. Available:")
            print(f"    {[c for c in df.columns if c.startswith('metric.')]}")
            return None

    # Group by model and level
    grouped = df.groupby(["model", "feature_level"])[metric].agg(["mean", "std", "count"])
    grouped = grouped.reset_index()

    print(f"\n{'='*72}")
    print(f"  Feature Level Ablation — {metric}")
    print(f"{'='*72}")
    print(f"\n{'Model':<12} {'Level':<12} {'Mean':>8} {'Std':>8} {'Seeds':>6}")
    print("-" * 50)

    for _, row in grouped.iterrows():
        print(f"{row['model']:<12} {row['feature_level']:<12} "
              f"{row['mean']:>8.4f} {row['std']:>8.4f} {int(row['count']):>6}")

    # Compute deltas
    print(f"\n{'Model':<12} {'Full':>8} {'Intrinsic':>10} {'Delta':>8} {'Relative':>10}")
    print("-" * 52)

    for model in models:
        full_vals = df[(df["model"] == model) & (df["feature_level"] == "full")][metric]
        intr_vals = df[(df["model"] == model) & (df["feature_level"] == "intrinsic")][metric]
        if len(full_vals) > 0 and len(intr_vals) > 0:
            full_mean = full_vals.mean()
            intr_mean = intr_vals.mean()
            delta = full_mean - intr_mean
            rel = delta / intr_mean * 100 if intr_mean > 0 else float("inf")
            print(f"{model:<12} {full_mean:>8.4f} {intr_mean:>10.4f} "
                  f"{delta:>+8.4f} {rel:>+9.1f}%")

    return grouped


# Hardcoded results from completed experiments (5 seeds each).
# Sources: XGBoost/MLP/GraphSAGE/SeHGNN from original 4-model sweep;
#          GCN/HAN back-computed from thesis text (pp deltas + percentages).
HARDCODED_RESULTS = {
    #            intrinsic   full
    "XGBoost":  (0.369,      0.415),
    "MLP":      (0.356,      0.385),
    "GCN":      (0.376,      0.308),
    "SageGNN":  (0.363,      0.377),
    "HAN":      (0.365,      0.374),
    "SeHGNN":   (0.401,      0.391),
}


def plot_hardcoded(output_path: str = "outputs/figures/feature_level_ablation.pdf",
                   models: list = None):
    """Generate figure from hardcoded results (no JSON loading needed)."""
    if models is None:
        models = ALL_MODELS
    models = [m for m in models if m in HARDCODED_RESULTS]
    display_names = [MODEL_DISPLAY.get(m, m) for m in models]

    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(12, 4.5),
                                            gridspec_kw={"width_ratios": [3, 2]})

    # --- Left panel: grouped bar chart ---
    x = np.arange(len(models))
    width = 0.35
    colors = {"intrinsic": "#4878CF", "full": "#D65F5F"}

    for i, level in enumerate(["intrinsic", "full"]):
        means = []
        for model in models:
            intr, full = HARDCODED_RESULTS[model]
            means.append(intr if level == "intrinsic" else full)

        label = ("Intrinsic (~85 dims)" if level == "intrinsic"
                 else "Full (~178 dims)")
        bars = ax_abs.bar(x + i * width - width / 2, means, width,
                          label=label, capsize=3, alpha=0.85, color=colors[level],
                          edgecolor="white", linewidth=0.5)

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax_abs.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f".{int(mean*1000)}", ha="center", va="bottom",
                            fontsize=7.5)

    ax_abs.set_ylabel("AUC-PR (Next Funding Round)")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(display_names)
    ax_abs.legend(loc="lower left")
    # Set y-axis to start near the data range for better visibility
    all_vals = [v for pair in HARDCODED_RESULTS.values() for v in pair]
    ymin = max(0, min(all_vals) - 0.05)
    ymax = max(all_vals) + 0.03
    ax_abs.set_ylim(ymin, ymax)
    ax_abs.grid(axis="y", alpha=0.3)
    ax_abs.set_title("(a) Absolute Performance")

    # Annotate model families
    ax_abs.axvline(x=1.5, color="gray", linestyle=":", alpha=0.4)
    ax_abs.text(0.5, ax_abs.get_ylim()[0] + 0.002, "Tabular", ha="center",
                fontsize=8, color="gray", style="italic")
    ax_abs.text(3.5, ax_abs.get_ylim()[0] + 0.002, "Graph Neural Networks",
                ha="center", fontsize=8, color="gray", style="italic")

    # --- Right panel: relative change (%) ---
    deltas = []
    for model in models:
        intr, full = HARDCODED_RESULTS[model]
        rel = (full - intr) / intr * 100
        deltas.append(rel)

    bar_colors = ["#D65F5F" if d > 0 else "#4878CF" for d in deltas]
    bars = ax_delta.barh(x, deltas, color=bar_colors, alpha=0.85,
                         edgecolor="white", linewidth=0.5)
    ax_delta.set_yticks(x)
    ax_delta.set_yticklabels(display_names)
    ax_delta.set_xlabel("Relative Change: Intrinsic → Full (%)")
    ax_delta.axvline(x=0, color="black", linewidth=0.8)
    ax_delta.grid(axis="x", alpha=0.3)
    ax_delta.set_title("(b) Impact of Relational Features")
    ax_delta.invert_yaxis()

    for bar, delta in zip(bars, deltas):
        offset = 0.3 if delta >= 0 else -0.3
        ha = "left" if delta >= 0 else "right"
        ax_delta.text(delta + offset, bar.get_y() + bar.get_height() / 2,
                      f"{delta:+.1f}%", ha=ha, va="center", fontsize=8.5,
                      fontweight="bold")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")
    plt.close(fig)


def plot_comparison(df: pd.DataFrame, metric: str = "metric.test_auc_pr_mom",
                    output_path: str = "outputs/figures/feature_level_ablation.pdf",
                    models: list = None):
    """Generate grouped bar chart showing feature-level ablation across models.

    Models are ordered tabular → homogeneous GNN → heterogeneous GNN so the
    robustness gradient is visually apparent.
    """
    df = df.copy()
    df["feature_level"] = df.apply(get_feature_level, axis=1)
    df["model"] = df.get("config.train.model", df.get("meta.model"))
    if models is None:
        models = ALL_MODELS
    # Only keep models that have data
    available = df["model"].unique()
    models = [m for m in models if m in available]
    df = df[df["model"].isin(models)]

    if metric not in df.columns:
        alternatives = [c for c in df.columns if "auc_pr" in c and "mom" in c]
        if alternatives:
            metric = alternatives[0]

    display_names = [MODEL_DISPLAY.get(m, m) for m in models]

    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(12, 4.5),
                                            gridspec_kw={"width_ratios": [3, 2]})

    # --- Left panel: grouped bar chart ---
    x = np.arange(len(models))
    width = 0.35
    colors = {"intrinsic": "#4878CF", "full": "#D65F5F"}

    for i, level in enumerate(["intrinsic", "full"]):
        means, stds = [], []
        for model in models:
            vals = df[(df["model"] == model) & (df["feature_level"] == level)][metric]
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)

        label = ("Intrinsic (~85 dims)" if level == "intrinsic"
                 else "Full (~178 dims)")
        bars = ax_abs.bar(x + i * width - width / 2, means, width, yerr=stds,
                          label=label, capsize=3, alpha=0.85, color=colors[level],
                          edgecolor="white", linewidth=0.5)

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax_abs.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f".{int(mean*1000)}", ha="center", va="bottom",
                            fontsize=7.5)

    ax_abs.set_ylabel("AUC-PR (Next Funding Round)")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(display_names)
    ax_abs.legend(loc="lower left")
    # Set y-axis to start near the data range for better visibility
    all_vals = df[metric].dropna()
    if len(all_vals) > 0:
        ymin = max(0, all_vals.min() - 0.05)
        ymax = all_vals.max() + 0.03
        ax_abs.set_ylim(ymin, ymax)
    ax_abs.grid(axis="y", alpha=0.3)
    ax_abs.set_title("(a) Absolute Performance")

    # Annotate model families
    ax_abs.axvline(x=1.5, color="gray", linestyle=":", alpha=0.4)
    ax_abs.text(0.5, ax_abs.get_ylim()[0] + 0.002, "Tabular", ha="center",
                fontsize=8, color="gray", style="italic")
    ax_abs.text(3.5, ax_abs.get_ylim()[0] + 0.002, "Graph Neural Networks",
                ha="center", fontsize=8, color="gray", style="italic")

    # --- Right panel: relative change (%) ---
    deltas = []
    for model in models:
        full = df[(df["model"] == model) & (df["feature_level"] == "full")][metric]
        intr = df[(df["model"] == model) & (df["feature_level"] == "intrinsic")][metric]
        if len(full) > 0 and len(intr) > 0:
            rel = (full.mean() - intr.mean()) / intr.mean() * 100
        else:
            rel = 0
        deltas.append(rel)

    bar_colors = ["#D65F5F" if d > 0 else "#4878CF" for d in deltas]
    bars = ax_delta.barh(x, deltas, color=bar_colors, alpha=0.85,
                         edgecolor="white", linewidth=0.5)
    ax_delta.set_yticks(x)
    ax_delta.set_yticklabels(display_names)
    ax_delta.set_xlabel("Relative Change: Intrinsic → Full (%)")
    ax_delta.axvline(x=0, color="black", linewidth=0.8)
    ax_delta.grid(axis="x", alpha=0.3)
    ax_delta.set_title("(b) Impact of Relational Features")
    ax_delta.invert_yaxis()

    for bar, delta in zip(bars, deltas):
        offset = 0.3 if delta >= 0 else -0.3
        ha = "left" if delta >= 0 else "right"
        ax_delta.text(delta + offset, bar.get_y() + bar.get_height() / 2,
                      f"{delta:+.1f}%", ha=ha, va="center", fontsize=8.5,
                      fontweight="bold")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"\nSaved figure: {output_path}")
    plt.close(fig)


def latex_table(df: pd.DataFrame, metric: str = "metric.test_auc_pr_mom",
                models: list = None) -> str:
    """Generate LaTeX table for thesis."""
    df = df.copy()
    df["feature_level"] = df.apply(get_feature_level, axis=1)
    df["model"] = df.get("config.train.model", df.get("meta.model"))
    if models is None:
        models = ALL_MODELS
    available = df["model"].unique()
    models = [m for m in models if m in available]
    df = df[df["model"].isin(models)]

    if metric not in df.columns:
        alternatives = [c for c in df.columns if "auc_pr" in c and "mom" in c]
        if alternatives:
            metric = alternatives[0]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Feature information level ablation. \textit{Intrinsic} retains only "
        r"entity-intrinsic attributes (temporal, online presence, organization, text "
        r"embeddings; ${\sim}84$ dimensions). \textit{Full} adds relational aggregations "
        r"from funding rounds, investors, founders, and graph structure "
        r"(${\sim}184$ dimensions). SeHGNN always has access to the full graph "
        r"structure via message passing.}",
        r"\label{tab:feature_level_ablation}",
        r"\begin{tabular}{l cc c}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Intrinsic} & \textbf{Full} & \textbf{$\Delta$} \\",
        r"\midrule",
    ]

    for model in models:
        full = df[(df["model"] == model) & (df["feature_level"] == "full")][metric]
        intr = df[(df["model"] == model) & (df["feature_level"] == "intrinsic")][metric]
        display = MODEL_DISPLAY.get(model, model)

        if len(full) > 0 and len(intr) > 0:
            f_str = f"${full.mean():.3f} \\pm {full.std():.3f}$"
            i_str = f"${intr.mean():.3f} \\pm {intr.std():.3f}$"
            d_str = f"${full.mean() - intr.mean():+.3f}$"
        else:
            f_str = i_str = d_str = "---"

        lines.append(f"  {display} & {i_str} & {f_str} & {d_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Feature level ablation analysis")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--metric", default="metric.test_auc_pr_mom")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    if args.plot:
        print("Generating figure from hardcoded results...")
        plot_hardcoded(os.path.join(args.output_dir, "feature_level_ablation.pdf"),
                       models=ALL_MODELS)

    # Optional: load JSON results for analysis / LaTeX table
    df = None
    if args.latex or not args.plot:
        print("Loading results...")
        df = load_all_results(args.results_dir)
        if df.empty:
            print("No results found.")
            return
        print(f"Loaded {len(df)} records.")
        analyze(df, args.metric, models=ALL_MODELS)

    if args.latex and df is not None:
        tex = latex_table(df, args.metric, models=ALL_MODELS)
        tex_path = os.path.join(args.output_dir, "feature_level_ablation.tex")
        Path(tex_path).parent.mkdir(parents=True, exist_ok=True)
        with open(tex_path, "w") as f:
            f.write(tex)
        print(f"\nSaved LaTeX: {tex_path}")
        print(f"\n{tex}")


if __name__ == "__main__":
    main()
