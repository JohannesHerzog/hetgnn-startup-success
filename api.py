"""FastAPI inference server for startup success prediction."""
import os
import numpy as np
import joblib
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


@app.post("/predict")
def predict(data: StartupInput):
    ref = state.get("momentum") or state.get("liquidity")
    if not ref:
        return {"error": "No models loaded"}

    X = _build_vector(data, ref["feature_names"])
    return {
        name: round(float(state[name]["model"].predict_proba(X)[0][1]), 4)
        for name in ("momentum", "liquidity") if name in state
    }


@app.get("/health")
def health():
    return {"status": "ok", "models": {k: k in state for k in ("momentum", "liquidity")}}
