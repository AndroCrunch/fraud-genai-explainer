from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import Config
from src.enrich import enrich_transactions
from src.features import add_derived_features, apply_rates

app = FastAPI(title="Fraud GenAI Explainer API", version="1.0.0")

MODEL_PATH = Path("artifacts/model.pkl")
ENC_PATH = Path("artifacts/encoders.pkl")
RATES_PATH = Path("artifacts/rates.pkl")

cfg = Config()
model = None
enc = None
rates = None


class PredictRequest(BaseModel):
    features: dict


@app.on_event("startup")
def load_artifacts():
    global model, enc, rates

    if not MODEL_PATH.exists():
        raise RuntimeError("Missing artifacts/model.pkl (run training first).")
    if not ENC_PATH.exists():
        raise RuntimeError("Missing artifacts/encoders.pkl (run training first).")
    if not RATES_PATH.exists():
        raise RuntimeError("Missing artifacts/rates.pkl (run training first).")

    model = joblib.load(MODEL_PATH)
    enc = joblib.load(ENC_PATH)
    rates = joblib.load(RATES_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "enc_loaded": enc is not None,
        "rates_loaded": rates is not None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or enc is None or rates is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    df = pd.DataFrame([req.features])

    # SAME preprocessing as generate_reports.py
    df, _ = enrich_transactions(df, cfg)
    df = add_derived_features(df)
    df = apply_rates(df, rates)

    # SAME features list as training
    feat_cols = enc["features"]
    X = df[feat_cols]

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= cfg.alert_threshold)

    return {"prediction": pred, "proba_fraud": proba}