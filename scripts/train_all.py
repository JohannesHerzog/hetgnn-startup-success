"""Train models K, M, G with shared test set (Option B) + comparison plots.

Shared benchmark = K's held-out test set (20% of filter_K population).
All three models are evaluated on this identical test set.
M and G are trained on their filtered populations with K's test UUIDs excluded.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --config path/to/config.yaml
"""
import sys
import os
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_mask(idx, mask):
    return idx[mask[idx] == 1]


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_on_shared_test(model_label, bundles, shared_X, shared_targets, shared_test_idx):
    """Evaluate a model's bundles on the shared K test set. Returns metrics dict."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score,
    )

    results = {}
    print(f"\n  [{model_label}] on shared K test set:")
    for task_name, task_data in shared_targets.items():
        if task_name not in bundles:
            continue
        model     = bundles[task_name]["model"]
        threshold = bundles[task_name]["threshold"]
        mask      = task_data["mask"]
        y         = task_data["y"]

        s_idx  = _apply_mask(shared_test_idx, mask)
        X_test = shared_X[s_idx]
        y_test = y[s_idx]

        if len(y_test) == 0 or y_test.sum() == 0:
            print(f"    {task_name}: skipped (no positive samples)")
            continue

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= threshold).astype(int)

        metrics = {
            "auc_roc":   roc_auc_score(y_test, prob),
            "auc_pr":    average_precision_score(y_test, prob),
            "prec":      precision_score(y_test, pred, zero_division=0),
            "recall":    recall_score(y_test, pred, zero_division=0),
            "f1":        f1_score(y_test, pred, zero_division=0),
            "n":         len(y_test),
            "pos_rate":  y_test.mean(),
            "prob":      prob,
            "y":         y_test,
            "threshold": threshold,
        }
        results[task_name] = metrics
        print(
            f"    {task_name:12s}  AUC-ROC={metrics['auc_roc']:.4f}  "
            f"AUC-PR={metrics['auc_pr']:.4f}  "
            f"Prec={metrics['prec']:.4f}  Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"n={metrics['n']:,}  pos={metrics['pos_rate']:.2%}  thresh={threshold:.2f}"
        )
    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_comparisons(all_results, out_dir):
    """Generate PR curves and a summary bar chart.

    all_results: {model_label: {task_name: metrics_dict}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    os.makedirs(out_dir, exist_ok=True)
    colors = {"K": "#e63946", "M": "#457b9d", "G": "#2a9d8f"}
    tasks  = ["momentum", "liquidity"]

    # ── 1. PR curves (one plot per task) ──────────────────────────────────────
    for task in tasks:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, task_results in all_results.items():
            if task not in task_results:
                continue
            m = task_results[task]
            prec_c, rec_c, _ = precision_recall_curve(m["y"], m["prob"])
            ax.plot(rec_c, prec_c, label=f"Model {label}  (AUC-PR={m['auc_pr']:.3f})",
                    color=colors.get(label, None), linewidth=2)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{task.capitalize()} — Precision-Recall Curve\n(shared K test set)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"{task}_pr_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved PR curve → {path}")

    # ── 2. ROC curves ──────────────────────────────────────────────────────────
    from sklearn.metrics import roc_curve
    for task in tasks:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        for label, task_results in all_results.items():
            if task not in task_results:
                continue
            m = task_results[task]
            fpr, tpr, _ = roc_curve(m["y"], m["prob"])
            ax.plot(fpr, tpr, label=f"Model {label}  (AUC-ROC={m['auc_roc']:.3f})",
                    color=colors.get(label, None), linewidth=2)

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{task.capitalize()} — ROC Curve\n(shared K test set)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, f"{task}_roc_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved ROC curve  → {path}")

    # ── 3. Summary bar chart ───────────────────────────────────────────────────
    model_labels = list(all_results.keys())
    metric_keys  = ["auc_pr", "auc_roc", "prec", "recall", "f1"]
    metric_names = ["AUC-PR", "AUC-ROC", "Precision", "Recall", "F1"]

    for task in tasks:
        fig, axes = plt.subplots(1, len(metric_keys), figsize=(14, 4), sharey=False)
        fig.suptitle(f"{task.capitalize()} — Metric Comparison (shared K test set)", fontsize=12)

        for ax, mk, mn in zip(axes, metric_keys, metric_names):
            vals = [all_results[lbl][task][mk] if task in all_results[lbl] else 0
                    for lbl in model_labels]
            bars = ax.bar(model_labels, vals,
                          color=[colors.get(l, "#999") for l in model_labels],
                          edgecolor="white", width=0.5)
            ax.set_title(mn, fontsize=10)
            ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
            ax.set_xticks(range(len(model_labels)))
            ax.set_xticklabels(model_labels)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        path = os.path.join(out_dir, f"{task}_summary_bars.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved bar chart  → {path}")

    # ── 4. Threshold sensitivity: Precision & Recall vs threshold ─────────────
    for task in tasks:
        fig, axes = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 4),
                                 sharey=True)
        if len(all_results) == 1:
            axes = [axes]
        fig.suptitle(f"{task.capitalize()} — Threshold Sensitivity\n(shared K test set)", fontsize=12)

        thresholds = np.arange(0.05, 0.96, 0.01)
        for ax, (label, task_results) in zip(axes, all_results.items()):
            if task not in task_results:
                ax.set_visible(False)
                continue
            m = task_results[task]
            from sklearn.metrics import precision_score as ps, recall_score as rs
            precs   = [ps(m["y"], (m["prob"] >= t).astype(int), zero_division=0) for t in thresholds]
            recalls = [rs(m["y"], (m["prob"] >= t).astype(int), zero_division=0) for t in thresholds]
            ax.plot(thresholds, precs,   label="Precision", color="#e63946", linewidth=2)
            ax.plot(thresholds, recalls, label="Recall",    color="#457b9d", linewidth=2)
            ax.axvline(x=task_results[task]["threshold"],
                       color="gray", linestyle="--", alpha=0.6, label="Current threshold")
            ax.set_title(f"Model {label}")
            ax.set_xlabel("Threshold")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        path = os.path.join(out_dir, f"{task}_threshold_sensitivity.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved threshold  → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])

    from src.ml.preprocessing import preprocess
    from src.ml.train import train
    from src.ml.filters import filter_M, filter_K

    all_results = {}

    from src.ml.train import _optimize_threshold
    import joblib

    def recalibrate_threshold(bundles, model_label, models_dir):
        """Re-optimise momentum threshold on K's val set and update saved bundle."""
        print(f"\n  Re-calibrating {model_label} momentum threshold on K val set...")
        val_idx   = data_K["splits"]["val"]
        mask      = data_K["targets"]["momentum"]["mask"]
        v_idx     = _apply_mask(val_idx, mask)
        X_val_K   = data_K["X"][v_idx]
        y_val_K   = data_K["targets"]["momentum"]["y"][v_idx]
        new_thresh = _optimize_threshold(
            bundles["momentum"]["model"], X_val_K, y_val_K,
            metric=config["training"].get("threshold_metric", "f1"),
            recall_target=config["training"].get("threshold_recall_target"),
        )
        bundles["momentum"]["threshold"] = new_thresh
        path   = os.path.join(models_dir, "xgboost_momentum.pkl")
        bundle = joblib.load(path)
        bundle["threshold"] = new_thresh
        joblib.dump(bundle, path)
        print(f"  Updated threshold → {path}")

    # ── Model K ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODEL K  (Europe + founded>=2016 + ≥1 round)")
    print("=" * 60)
    data_K    = preprocess(config, filter_fn=filter_K)
    bundles_K = train(data_K, config, models_dir="outputs/models_K")

    shared_X          = data_K["X"]
    shared_targets    = data_K["targets"]
    shared_test_idx   = data_K["splits"]["test"]
    shared_test_uuids = data_K["test_uuids"]
    # K's val UUIDs are excluded from M/G training so we can use them
    # for threshold calibration without leakage
    val_idx_K      = data_K["splits"]["val"]
    val_uuids_K    = set(data_K["uuids"][val_idx_K].tolist())
    exclude_from_M = shared_test_uuids | val_uuids_K
    print(f"\n  Shared test set:  {len(shared_test_uuids):,} startups")
    print(f"  K val set (for threshold calibration): {len(val_uuids_K):,} startups")

    all_results["K"] = evaluate_on_shared_test(
        "K", bundles_K, shared_X, shared_targets, shared_test_idx
    )

    # ── Model M ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODEL M  (operating + founded≥2014 + founder>0)")
    print("=" * 60)
    data_M    = preprocess(config, filter_fn=filter_M, exclude_uuids=exclude_from_M)
    bundles_M = train(data_M, config, models_dir="outputs/models_M")
    print(f"  M training set: {len(data_M['splits']['train']):,} startups")
    recalibrate_threshold(bundles_M, "M", "outputs/models_M")

    all_results["M"] = evaluate_on_shared_test(
        "M", bundles_M, shared_X, shared_targets, shared_test_idx
    )

    # ── Model G ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODEL G  (no filter — all startups)")
    print("=" * 60)
    data_G    = preprocess(config, exclude_uuids=exclude_from_M)
    bundles_G = train(data_G, config, models_dir="outputs/models_G")
    print(f"  G training set: {len(data_G['splits']['train']):,} startups")
    recalibrate_threshold(bundles_G, "G", "outputs/models_G")

    all_results["G"] = evaluate_on_shared_test(
        "G", bundles_G, shared_X, shared_targets, shared_test_idx
    )

    # ── Comparison plots ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)
    plot_comparisons(all_results, out_dir="outputs/comparison")

    print("\n" + "=" * 60)
    print("DONE")
    print("  K → outputs/models_K/")
    print("  M → outputs/models_M/")
    print("  G → outputs/models_G/")
    print("  Plots → outputs/comparison/")
    print("=" * 60)


if __name__ == "__main__":
    main()
