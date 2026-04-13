"""FastAPI inference server for startup success prediction."""
import os
import io
import base64
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')  # Verhindert GUI-Abstürze auf dem Server!
import matplotlib.pyplot as plt

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

MODELS_DIR = "models"
PCA_PATH   = "data/embeddings_csv/org_pca_64d.pkl"
EMB_DIM    = 64

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name in ("momentum", "liquidity"):
        path = os.path.join(MODELS_DIR, f"xgboost_{name}.pkl")
        if os.path.exists(path):
            state[name] = joblib.load(path)
            print(f"Loaded model: {name}")

    if os.path.exists(PCA_PATH):
        state["pca"] = joblib.load(PCA_PATH)

    from sentence_transformers import SentenceTransformer
    state["st"] = SentenceTransformer("all-MiniLM-L6-v2")
    print("Ready.")
    yield
    state.clear()


app = FastAPI(title="Startup Success Predictor", lifespan=lifespan)


class StartupInput(BaseModel):
    founder_count:             Optional[float] = 0
    female_count:              Optional[float] = 0
    serial_founder_count:      Optional[float] = 0
    ivy_league_plus_count:     Optional[float] = 0
    bachelor_count:            Optional[float] = 0
    master_count:              Optional[float] = 0
    phd_count:                 Optional[float] = 0
    founded_on_year:           Optional[float] = 0
    has_domain:                Optional[float] = 0
    has_linkedin_url:          Optional[float] = 0
    has_description:           Optional[float] = 0
    description_length:        Optional[float] = 0
    description:               Optional[str]   = ""
    total_funding_usd:         Optional[float] = 0
    num_funding_rounds:        Optional[float] = 0
    total_investors:           Optional[float] = 0
    last_funding_stage:        Optional[float] = 0
    months_since_last_funding: Optional[float] = 0


def _build_vector(data: StartupInput, feature_names: list) -> np.ndarray:
    desc_emb = np.zeros(EMB_DIM, dtype=np.float32)
    if data.description and "pca" in state:
        raw      = state["st"].encode([data.description], show_progress_bar=False)
        desc_emb = state["pca"].transform(raw)[0].astype(np.float32)

    human = data.model_dump(exclude={"description"})
    values, idx = [], 0
    for feat in feature_names:
        if feat.startswith("desc_emb_"):
            values.append(float(desc_emb[idx])); idx += 1
        else:
            values.append(float(human.get(feat, 0.0)))
    return np.array([values], dtype=np.float32)


def get_shap_waterfall_base64(model, feature_names, X, title):
    """Generates a SHAP waterfall plot and returns it as a base64 string."""
    explainer = shap.Explainer(model, feature_perturbation="interventional", link=shap.links.logit)
    shap_values = explainer(X)

    desc_indices = [i for i, n in enumerate(feature_names) if n.startswith("desc_emb_")]
    other_indices = [i for i, n in enumerate(feature_names) if not n.startswith("desc_emb_")]

    if desc_indices:
        agg_values = shap_values.values[0][other_indices].tolist()
        agg_values.append(shap_values.values[0][desc_indices].sum())
        agg_data = X[0][other_indices].tolist()
        agg_data.append(0.0) 
        agg_names = [feature_names[i] for i in other_indices] + ["description (embedding)"]

        new_shap = shap.Explanation(
            values=np.array(agg_values),
            base_values=shap_values.base_values[0],
            data=np.array(agg_data),
            feature_names=agg_names,
        )
    else:
        new_shap = shap_values[0]

    plt.figure()
    shap.plots.waterfall(new_shap, show=False)
    plt.title(title, pad=12)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close('all')
    
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/predict")
def predict(data: StartupInput):
    ref = state.get("momentum") or state.get("liquidity")
    if not ref:
        return {"error": "No models loaded"}

    X = _build_vector(data, ref["feature_names"])
    
    results = {}
    for name in ("momentum", "liquidity"):
        if name in state:
            model = state[name]["model"]
            # Score für BEIDE berechnen (Momentum & Liquidity)
            results[name] = round(float(model.predict_proba(X)[0][1]), 4)
            
            # SHAP Plot ABER NUR für Momentum generieren
            if name == "momentum":
                results["plot_momentum_base64"] = get_shap_waterfall_base64(
                    model, ref["feature_names"], X, title="Momentum Waterfall"
                )
            
    return results


@app.get("/health")
def health():
    return {"status": "ok", "models": {k: k in state for k in ("momentum", "liquidity")}}