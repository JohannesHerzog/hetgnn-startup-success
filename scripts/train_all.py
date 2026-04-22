"""Train models K, M, G sequentially with test-set isolation and cross-evaluation.

Order: K (most filtered) → M (pipeline filter) → G (no filter)
Each model's test set is excluded from all subsequent training runs.

Cross-evaluation:
  Model M is additionally evaluated on K's test set.
  Model G is additionally evaluated on K's and M's test sets.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --config path/to/config.yaml
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_mask(idx, mask):
    return idx[mask[idx] == 1]


def cross_evaluate(bundles, eval_data, eval_label):
    """Evaluate trained models on the test split of a different dataset.

    Args:
        bundles:    {task_name: {"model": ..., "threshold": ...}}
        eval_data:  preprocess() output dict (X, targets, splits)
        eval_label: string label for logging (e.g. "K test set")
    """
    X        = eval_data["X"]
    targets  = eval_data["targets"]
    test_idx = eval_data["splits"]["test"]

    print(f"\n  -- Cross-evaluation on {eval_label} --")
    for task_name, task_data in targets.items():
        if task_name not in bundles:
            continue
        model     = bundles[task_name]["model"]
        threshold = bundles[task_name]["threshold"]
        mask      = task_data["mask"]
        y         = task_data["y"]

        s_idx  = _apply_mask(test_idx, mask)
        X_test = X[s_idx]
        y_test = y[s_idx]

        if len(y_test) == 0 or y_test.sum() == 0:
            print(f"    {task_name}: skipped (no positive samples in {eval_label})")
            continue

        prob  = model.predict_proba(X_test)[:, 1]
        pred  = (prob >= threshold).astype(int)
        auc   = roc_auc_score(y_test, prob)
        ap    = average_precision_score(y_test, prob)
        prec  = precision_score(y_test, pred, zero_division=0)
        rec   = recall_score(y_test, pred, zero_division=0)
        f1    = f1_score(y_test, pred, zero_division=0)
        print(
            f"    {task_name:12s}  AUC-ROC={auc:.4f}  AUC-PR={ap:.4f}  "
            f"Prec={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  "
            f"n={len(y_test):,}  pos={y_test.mean():.2%}  thresh={threshold:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])

    from src.ml.preprocessing import preprocess
    from src.ml.train import train
    from src.ml.filters import filter_M, filter_K

    # ── Model K (most filtered) ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODEL K  (Europe, founded>2020, ≥1 round)")
    print("=" * 60)
    data_K    = preprocess(config, filter_fn=filter_K)
    bundles_K = train(data_K, config, models_dir="outputs/models_K")
    test_uuids_K = data_K["test_uuids"]
    print(f"\n  K test set: {len(test_uuids_K):,} startups held out")

    # ── Model M (pipeline filter, excludes K test set) ─────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODEL M  (operating, founded≥2014, founder>0)")
    print("=" * 60)
    data_M    = preprocess(config, filter_fn=filter_M, exclude_uuids=test_uuids_K)
    bundles_M = train(data_M, config, models_dir="outputs/models_M")
    test_uuids_M = data_M["test_uuids"]
    print(f"\n  M test set: {len(test_uuids_M):,} startups held out")

    # Cross-eval M on K's test set
    cross_evaluate(bundles_M, data_K, "K test set (Europe/2020/funded)")

    # ── Model G (no filter, excludes K and M test sets) ────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODEL G  (no filter, all startups)")
    print("=" * 60)
    data_G    = preprocess(config, exclude_uuids=test_uuids_K | test_uuids_M)
    bundles_G = train(data_G, config, models_dir="outputs/models_G")
    test_uuids_G = data_G["test_uuids"]
    print(f"\n  G test set: {len(test_uuids_G):,} startups held out")

    # Cross-eval G on K's and M's test sets
    cross_evaluate(bundles_G, data_K, "K test set (Europe/2020/funded)")
    cross_evaluate(bundles_G, data_M, "M test set (operating/2014/founder)")

    print("\n" + "=" * 60)
    print("DONE — models saved to:")
    print("  K → outputs/models_K/")
    print("  M → outputs/models_M/")
    print("  G → outputs/models_G/")
    print("=" * 60)


if __name__ == "__main__":
    main()
