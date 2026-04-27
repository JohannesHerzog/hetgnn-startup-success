"""Train Model M with Optuna hyperparameter tuning, evaluate on K's test set.

Steps:
  1. Preprocess K  → extract shared test set + val set (no training)
  2. Preprocess M  → exclude K test + val UUIDs to avoid leakage
  3. Tune M        → Optuna TPE search on M's val set (maximise AUC-PR)
  4. Train M       → final model with best params
  5. Recalibrate   → threshold optimised on K's val set (85% recall on K population)
  6. Evaluate      → M on K's shared test set
  7. Plots         → PR curve, ROC, bar chart, threshold sensitivity

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --config path/to/config.yaml --trials 50
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


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_on_shared_test(model_label, bundles, shared_X, shared_targets, shared_test_idx):
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

def plot_results(results, label, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    os.makedirs(out_dir, exist_ok=True)
    tasks = [t for t in ["momentum", "liquidity"] if t in results]

    for task in tasks:
        m = results[task]

        # PR curve
        fig, ax = plt.subplots(figsize=(7, 5))
        prec_c, rec_c, _ = precision_recall_curve(m["y"], m["prob"])
        ax.plot(rec_c, prec_c, linewidth=2, color="#457b9d",
                label=f"{label}  (AUC-PR={m['auc_pr']:.3f})")
        ax.axvline(m["recall"], color="gray", linestyle="--", alpha=0.6,
                   label=f"Operating point  (Prec={m['prec']:.2f} Rec={m['recall']:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{task.capitalize()} — Precision-Recall Curve\n(K test set)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{task}_pr_curve.png"), dpi=150)
        plt.close(fig)

        # ROC curve
        fig, ax = plt.subplots(figsize=(7, 5))
        fpr, tpr, _ = roc_curve(m["y"], m["prob"])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        ax.plot(fpr, tpr, linewidth=2, color="#457b9d",
                label=f"{label}  (AUC-ROC={m['auc_roc']:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{task.capitalize()} — ROC Curve\n(K test set)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{task}_roc_curve.png"), dpi=150)
        plt.close(fig)

        # Threshold sensitivity
        thresholds = np.arange(0.05, 0.96, 0.01)
        from sklearn.metrics import precision_score as ps, recall_score as rs
        precs   = [ps(m["y"], (m["prob"] >= t).astype(int), zero_division=0) for t in thresholds]
        recalls = [rs(m["y"], (m["prob"] >= t).astype(int), zero_division=0) for t in thresholds]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(thresholds, precs,   label="Precision", color="#e63946", linewidth=2)
        ax.plot(thresholds, recalls, label="Recall",    color="#457b9d", linewidth=2)
        ax.axvline(m["threshold"], color="gray", linestyle="--", alpha=0.7,
                   label=f"Current threshold ({m['threshold']:.2f})")
        ax.set_xlabel("Threshold")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{task.capitalize()} — Threshold Sensitivity  [{label}]\n(K test set)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{task}_threshold_sensitivity.png"), dpi=150)
        plt.close(fig)

        print(f"  Saved plots for {task} → {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])

    from src.ml.preprocessing import preprocess
    from src.ml.train import tune_hyperparams, train_task, save_bundle, _apply_mask, _optimize_threshold
    from src.ml.filters import filter_M, filter_K
    import joblib

    # ── Step 1: Preprocess K (no training — only for test/val split) ───────────
    print("\n" + "=" * 60)
    print("PREPROCESSING K  (extracting shared test + val sets)")
    print("=" * 60)
    data_K = preprocess(config, filter_fn=filter_K)

    shared_X          = data_K["X"]
    shared_targets    = data_K["targets"]
    shared_test_idx   = data_K["splits"]["test"]
    shared_test_uuids = data_K["test_uuids"]
    val_idx_K         = data_K["splits"]["val"]
    val_uuids_K       = set(data_K["uuids"][val_idx_K].tolist())
    exclude_uuids     = shared_test_uuids | val_uuids_K

    print(f"  Shared test set : {len(shared_test_uuids):,} startups")
    print(f"  K val set       : {len(val_uuids_K):,} startups  (for threshold calibration)")

    # ── Step 2: Preprocess M ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREPROCESSING M  (operating + founded≥2014 + founder>0)")
    print("=" * 60)
    data_M = preprocess(config, filter_fn=filter_M, exclude_uuids=exclude_uuids)
    print(f"  M dataset: {len(data_M['splits']['train']):,} train  "
          f"{len(data_M['splits']['val']):,} val  "
          f"{len(data_M['splits']['test']):,} test")

    models_dir = "outputs/models"
    os.makedirs(models_dir, exist_ok=True)

    # ── Steps 3–4: Tune + train each task ─────────────────────────────────────
    bundles = {}
    for task_name, task_data in data_M["targets"].items():
        print(f"\n{'=' * 60}")
        print(f"TUNING + TRAINING: {task_name.upper()}")
        print("=" * 60)

        t_idx = _apply_mask(data_M["splits"]["train"], task_data["mask"])
        v_idx = _apply_mask(data_M["splits"]["val"],   task_data["mask"])

        X_train, y_train = data_M["X"][t_idx], task_data["y"][t_idx]
        X_val,   y_val   = data_M["X"][v_idx], task_data["y"][v_idx]

        print(f"  Train: {len(X_train):,}  ({y_train.mean():.2%} pos)")
        print(f"  Val:   {len(X_val):,}  ({y_val.mean():.2%} pos)")
        print(f"\n  Running Optuna ({args.trials} trials)...")

        best_params = tune_hyperparams(
            X_train, y_train, X_val, y_val,
            base_params={"objective": "binary:logistic", "tree_method": "hist"},
            n_trials=args.trials,
            seed=config["seed"],
        )

        model, threshold = train_task(
            task_name,
            data_M["X"],
            task_data["y"],
            task_data["mask"],
            data_M["splits"],
            best_params,
            config,
        )
        save_bundle(model, data_M["feature_names"], task_name, threshold,
                    models_dir, test_uuids=shared_test_uuids)
        bundles[task_name] = {"model": model, "threshold": threshold}

    # ── Step 5: Recalibrate momentum threshold on K's val set ─────────────────
    print("\n" + "=" * 60)
    print("RECALIBRATING MOMENTUM THRESHOLD ON K VAL SET")
    print("=" * 60)
    mask_mom = shared_targets["momentum"]["mask"]
    v_idx_K  = _apply_mask(val_idx_K, mask_mom)
    new_thresh = _optimize_threshold(
        bundles["momentum"]["model"],
        shared_X[v_idx_K],
        shared_targets["momentum"]["y"][v_idx_K],
        metric=config["training"].get("threshold_metric", "f1"),
        recall_target=config["training"].get("threshold_recall_target"),
    )
    bundles["momentum"]["threshold"] = new_thresh
    b = joblib.load(os.path.join(models_dir, "xgboost_momentum.pkl"))
    b["threshold"] = new_thresh
    joblib.dump(b, os.path.join(models_dir, "xgboost_momentum.pkl"))
    print(f"  Saved updated threshold → {models_dir}/xgboost_momentum.pkl")

    # ── Step 6: Evaluate on K's shared test set ────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION ON K SHARED TEST SET")
    print("=" * 60)
    results = evaluate_on_shared_test(
        "M (tuned)", bundles, shared_X, shared_targets, shared_test_idx
    )

    # ── Step 7: Plots ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_results(results, label="Model M (tuned)", out_dir="outputs/comparison")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Model  → {models_dir}/")
    print("  Plots  → outputs/comparison/")
    print("=" * 60)


if __name__ == "__main__":
    main()
