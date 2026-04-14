"""XGBoost training for Momentum (next funding round) and Liquidity (exit) tasks."""
import os
import numpy as np
import joblib


def _apply_mask(idx, mask):
    """Return only the elements of idx where mask == 1."""
    return idx[mask[idx] == 1]


def _evaluate(model, X, y, label):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    if len(y) == 0 or y.sum() == 0:
        print(f"  {label}: skipped (no positive samples)")
        return

    prob = model.predict_proba(X)[:, 1]
    auc  = roc_auc_score(y, prob)
    ap   = average_precision_score(y, prob)
    print(f"  {label}: AUC-ROC={auc:.4f}  AUC-PR={ap:.4f}  n={len(y)}  pos={y.mean():.2%}")


def _optimize_threshold(model, X_val, y_val, metric="f1", recall_target=None):
    """Find the decision threshold that maximises `metric` on the validation set.

    If metric='min_recall', finds the highest threshold (most precise) that still
    achieves recall >= recall_target on the positive class.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    prob = model.predict_proba(X_val)[:, 1]

    def _report(thresh):
        pred = (prob >= thresh).astype(int)
        p  = precision_score(y_val, pred, zero_division=0)
        r  = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        print(f"  Threshold={thresh:.2f}  →  Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    if metric == "min_recall":
        target = recall_target if recall_target is not None else 0.85
        best_thresh, best_precision = None, -1.0
        for t in np.arange(0.05, 0.95, 0.01):
            pred = (prob >= t).astype(int)
            r = recall_score(y_val, pred, zero_division=0)
            p = precision_score(y_val, pred, zero_division=0)
            if r >= target and p > best_precision:
                best_precision, best_thresh = p, t
        if best_thresh is None:
            print(f"  WARNING: recall target {target:.0%} unreachable — falling back to F1")
            return _optimize_threshold(model, X_val, y_val, metric="f1")
        print(f"  Threshold optimised for recall≥{target:.0%}:")
        _report(best_thresh)
        return float(best_thresh)

    metric_fns = {
        "f1":        lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
        "precision": lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
        "recall":    lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
    }
    fn = metric_fns.get(metric, metric_fns["f1"])

    best_thresh, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        pred  = (prob >= t).astype(int)
        score = fn(y_val, pred)
        if score > best_score:
            best_score, best_thresh = score, t

    print(f"  Threshold optimised for {metric} ({metric}={best_score:.4f}):")
    _report(best_thresh)
    return float(best_thresh)


def train_task(task_name, X, y, mask, splits, model_params, config):
    """Train one XGBoost model; return (model, threshold)."""
    import xgboost as xgb

    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]

    t_idx = _apply_mask(train_idx, mask)
    v_idx = _apply_mask(val_idx,   mask)
    s_idx = _apply_mask(test_idx,  mask)

    X_train, y_train = X[t_idx], y[t_idx]
    X_val,   y_val   = X[v_idx], y[v_idx]
    X_test,  y_test  = X[s_idx], y[s_idx]

    print(f"\n{'='*40}")
    print(f"Task: {task_name}")
    print(f"  Train: {len(X_train):,}  ({y_train.mean():.2%} pos)")
    print(f"  Val:   {len(X_val):,}  ({y_val.mean():.2%} pos)")
    print(f"  Test:  {len(X_test):,}  ({y_test.mean():.2%} pos)")

    # Compute scale_pos_weight from actual training distribution after masking.
    # Overrides the static config value, which may not match the masked class ratio.
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    dynamic_spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
    params = {**model_params, "scale_pos_weight": dynamic_spw}
    print(f"  scale_pos_weight: {dynamic_spw:.2f}  (neg={n_neg:,} pos={n_pos:,})")

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    _evaluate(model, X_val,  y_val,  "Val")
    _evaluate(model, X_test, y_test, "Test")

    threshold = 0.5
    if config["training"].get("optimize_threshold", False):
        threshold = _optimize_threshold(
            model, X_val, y_val,
            metric=config["training"].get("threshold_metric", "f1"),
            recall_target=config["training"].get("threshold_recall_target"),
        )

    return model, threshold


def save_bundle(model, feature_names, task_name, threshold, output_dir):
    """Save a joblib bundle compatible with scripts/predict.py."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"xgboost_{task_name.lower()}.pkl")
    bundle = {
        "model":         model,
        "feature_names": feature_names,
        "task":          task_name,
        "num_classes":   2,
        "threshold":     threshold,
    }
    joblib.dump(bundle, path)
    print(f"  Saved → {path}")
    return path


def train(data, config):
    """Train Momentum and Liquidity models and save bundles."""
    X             = data["X"]
    feature_names = data["feature_names"]
    targets       = data["targets"]
    splits        = data["splits"]
    model_params  = config["model"]
    models_dir    = config["paths"]["models_dir"]

    for task_name, task_data in targets.items():
        model, threshold = train_task(
            task_name,
            X,
            task_data["y"],
            task_data["mask"],
            splits,
            model_params,
            config,
        )
        save_bundle(model, feature_names, task_name, threshold, models_dir)

    print("\nAll models saved.")
