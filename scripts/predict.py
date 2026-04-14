"""Inference script: load trained XGBoost models and predict startup success from custom inputs."""
import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PCA_PATH = "data/embeddings_csv/org_pca_64d.pkl"
EMBEDDING_DIM = 64


def embed_description(text):
    """Encode a description string to 64-dim vector using Sentence-Transformer + saved PCA."""
    import joblib
    from sentence_transformers import SentenceTransformer

    if not os.path.exists(PCA_PATH):
        print(f"Warning: PCA model not found at {PCA_PATH}. Description will be ignored (all zeros).")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    pca = joblib.load(PCA_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode([text], batch_size=1, show_progress_bar=False)  # (1, 384)
    reduced = pca.transform(embedding)  # (1, 64)
    return reduced[0].astype(np.float32)


# Only features a human can meaningfully provide — everything else defaults to 0
HUMAN_FEATURES = {
    "founded_on_year": "Year founded (e.g. 2020)",
    "has_domain": "Has website domain? (0=No, 1=Yes)",
    "has_linkedin_url": "Has LinkedIn? (0=No, 1=Yes)",
    "has_description": "Has company description? (0=No, 1=Yes)",
    "description_length": "Description length in characters (e.g. 300)",
    "founder_count": "Number of founders",
    "total_funding_usd": "Total funding raised in USD (e.g. 5000000)",
    "num_funding_rounds": "Number of funding rounds so far",
    "total_investors": "Total number of investors",
    "last_funding_stage": "Last funding stage (0=none, 1=Pre-Seed, 2=Seed, 3=Series A, 4=Series B, 5=Series C+)",
    "months_since_last_funding": "Months since last funding round (0 if never funded)",
    "female_count": "Number of female founders",
    "serial_founder_count": "Number of founders who founded a company before",
    "ivy_league_plus_count": "Founders from top universities (Ivy League / equivalent)",
    "bachelor_count": "Founders with bachelor's degree",
    "master_count": "Founders with master's degree",
    "phd_count": "Founders with PhD",
}


def load_bundle(path):
    import joblib
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def predict_proba(bundle, feature_vector):
    model = bundle["model"]
    X = np.array([feature_vector], dtype=np.float32)
    proba = model.predict_proba(X)
    return float(proba[0][1]) if proba.shape[1] == 2 else float(proba[0].max())


def plot_waterfall(bundle, feature_vector, title):
    """Plot a SHAP waterfall chart, grouping desc_emb_* into one 'description' bar."""
    import shap
    import matplotlib.pyplot as plt

    model = bundle["model"]
    feature_names = bundle.get("feature_names", [])
    X = np.array([feature_vector], dtype=np.float32)

    explainer = shap.Explainer(model, feature_perturbation="interventional", link=shap.links.logit)
    shap_values = explainer(X)

    # Aggregate all desc_emb_* into a single "description" feature
    desc_indices = [i for i, n in enumerate(feature_names) if n.startswith("desc_emb_")]
    other_indices = [i for i, n in enumerate(feature_names) if not n.startswith("desc_emb_")]

    if desc_indices:
        agg_values = shap_values.values[0][other_indices].tolist()
        agg_values.append(shap_values.values[0][desc_indices].sum())
        agg_data = X[0][other_indices].tolist()
        agg_data.append(0.0)  # raw value not meaningful for aggregated embedding
        agg_names = [feature_names[i] for i in other_indices] + ["description (embedding)"]

        new_shap = shap.Explanation(
            values=np.array(agg_values),
            base_values=shap_values.base_values[0],
            data=np.array(agg_data),
            feature_names=agg_names,
        )
    else:
        new_shap = shap_values[0]

    shap.plots.waterfall(new_shap, show=False)
    plt.title(title, pad=12)
    plt.tight_layout()
    plt.savefig(f"waterfall_{title.lower().replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved waterfall_{title.lower().replace(' ', '_')}.png")


def prompt_features(feature_names):
    """Ask the user only for human-readable features; fill the rest with 0."""
    print("\nEnter startup features below.")
    print("Press Enter to use 0 for unknown / not applicable values.\n")

    # Ask for description text first
    description = input("  description (Short company description, e.g. 'AI-powered logistics platform' — press Enter to skip): ").strip()

    # Collect values for human-provided features
    human_values = {}
    for feat, desc in HUMAN_FEATURES.items():
        while True:
            raw = input(f"  {feat} ({desc}): ").strip()
            if raw == "":
                human_values[feat] = 0.0
                break
            try:
                human_values[feat] = float(raw)
                break
            except ValueError:
                print("    Please enter a number (or press Enter for 0).")

    # Compute description embeddings if provided
    desc_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    if description:
        print("\nEncoding description...")
        desc_emb = embed_description(description)
        print("Done.")

    # Build full feature vector in the correct order
    values = []
    emb_idx = 0
    for feat in feature_names:
        if feat.startswith("desc_emb_"):
            values.append(float(desc_emb[emb_idx]))
            emb_idx += 1
        else:
            values.append(human_values.get(feat, 0.0))
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Predict startup success probability using trained XGBoost models."
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/pipeline_state",
        help="Directory containing trained model files (default: outputs/pipeline_state)",
    )
    args = parser.parse_args()

    models_dir = os.path.join(args.model_dir, "models")

    # Try to load momentum and liquidity models
    mom_bundle = load_bundle(os.path.join(models_dir, "xgboost_momentum.pkl"))
    liq_bundle = load_bundle(os.path.join(models_dir, "xgboost_liquidity.pkl"))
    default_bundle = load_bundle(os.path.join(models_dir, "xgboost_default.pkl"))

    if mom_bundle is None and liq_bundle is None and default_bundle is None:
        print(f"No XGBoost model files found in {models_dir}")
        print("Run training first:  python src/main.py")
        sys.exit(1)

    # Pick feature names from the first available bundle
    reference_bundle = mom_bundle or liq_bundle or default_bundle
    feature_names = reference_bundle.get("feature_names", [])

    if not feature_names:
        print("Warning: feature names not found in model bundle. Cannot prompt for inputs.")
        sys.exit(1)

    print(f"Loaded model(s) from {models_dir}")
    print(f"Model expects {len(feature_names)} features.")

    feature_values = prompt_features(feature_names)

    print("\n" + "=" * 50)
    print("PREDICTIONS")
    print("=" * 50)

    if mom_bundle is not None:
        prob = predict_proba(mom_bundle, feature_values)
        print(f"Next Funding Round (Momentum):  {prob:.1%}")
        plot_waterfall(mom_bundle, feature_values, "Momentum")

    if liq_bundle is not None:
        prob = predict_proba(liq_bundle, feature_values)
        print(f"Exit Event - IPO / Acquisition: {prob:.1%}")
        plot_waterfall(liq_bundle, feature_values, "Liquidity")

    if default_bundle is not None:
        prob = predict_proba(default_bundle, feature_values)
        print(f"Prediction:  {prob:.1%}")
        plot_waterfall(default_bundle, feature_values, "Prediction")

    print("=" * 50)


if __name__ == "__main__":
    main()
