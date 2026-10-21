"""
FastAPI Prediction Service
===========================
Serves ML model predictions via REST endpoints.
Called internally by Streamlit (localhost:8000).

Run:  uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src/ is on path when run from project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
PROC_DIR   = ROOT / "data" / "processed"

app = FastAPI(
    title="Product Quality Intelligence API",
    description="ML prediction endpoints for defect risk, return forecasting, and anomaly detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Cache (loaded once on startup) ─────────────────────────────────────

_models = {}

def load_models():
    """Lazy-load models from disk."""
    global _models
    defect_path  = MODELS_DIR / "defect_predictor.joblib"
    return_path  = MODELS_DIR / "return_forecaster.joblib"
    anomaly_path = MODELS_DIR / "anomaly_detector.joblib"

    if defect_path.exists() and "defect" not in _models:
        _models["defect"] = joblib.load(defect_path)
        log.info("Loaded defect predictor")

    if return_path.exists() and "return" not in _models:
        _models["return"] = joblib.load(return_path)
        log.info("Loaded return forecaster")

    if anomaly_path.exists() and "anomaly" not in _models:
        _models["anomaly"] = joblib.load(anomaly_path)
        log.info("Loaded anomaly detector")


@app.on_event("startup")
async def startup():
    load_models()


# ── Request / Response Schemas ────────────────────────────────────────────────

class DefectPredictRequest(BaseModel):
    supplier_enc:        float = Field(0, description="Encoded supplier index (0–9)")
    production_line_enc: float = Field(0, description="Encoded production line index")
    category_enc:        float = Field(0, description="Encoded product category")
    avg_lead_time:       float = Field(10.0)
    n_batches:           float = Field(5.0)
    avg_discount:        float = Field(0.1)
    avg_sentiment:       float = Field(0.0)
    pct_negative:        float = Field(0.1)
    rolling_3m_return:   float = Field(0.05)
    return_rate:         float = Field(0.05)
    n_reviews:           float = Field(50.0)


class ReturnForecastRequest(BaseModel):
    supplier_enc:        float = Field(0)
    production_line_enc: float = Field(0)
    category_enc:        float = Field(0)
    total_units_sold:    float = Field(500.0)
    avg_rating:          float = Field(3.5)
    avg_sentiment:       float = Field(0.0)
    pct_negative:        float = Field(0.1)
    avg_defect_rate:     float = Field(0.03)
    defect_risk_label:   float = Field(0.0)
    rolling_3m_return:   float = Field(0.05)
    rolling_6m_return:   float = Field(0.05)
    return_momentum:     float = Field(0.0)
    avg_discount:        float = Field(0.1)
    avg_lead_time:       float = Field(10.0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(_models.keys()),
    }


@app.post("/predict/defect")
def predict_defect(req: DefectPredictRequest):
    if "defect" not in _models:
        raise HTTPException(503, "Defect model not loaded. Run the training pipeline first.")

    bundle   = _models["defect"]
    model    = bundle["model"]
    features = bundle["features"]

    X = pd.DataFrame([req.dict()])[features]
    proba     = float(model.predict_proba(X)[0][1])
    label     = "High Risk" if proba > 0.5 else "Low Risk"
    importances = dict(zip(features, model.feature_importances_.tolist()))
    top_factors = sorted(importances, key=importances.get, reverse=True)[:3]

    return {
        "defect_probability": round(proba, 4),
        "risk_label":         label,
        "confidence":         round(max(proba, 1 - proba), 4),
        "top_risk_factors":   top_factors,
    }


@app.post("/predict/returns")
def predict_returns(req: ReturnForecastRequest):
    if "return" not in _models:
        raise HTTPException(503, "Return model not loaded.")

    bundle   = _models["return"]
    model    = bundle["model"]
    features = bundle["features"]

    X = pd.DataFrame([req.dict()])[features]
    predicted_qty = max(0.0, float(model.predict(X)[0]))

    return {
        "predicted_return_qty":    round(predicted_qty, 1),
        "predicted_return_rate_pct": round(
            predicted_qty / max(req.total_units_sold, 1) * 100, 2
        ),
        "risk_level": (
            "High"   if predicted_qty > 50 else
            "Medium" if predicted_qty > 20 else
            "Low"
        ),
    }


@app.get("/anomalies")
def get_anomalies(limit: int = 50):
    path = PROC_DIR / "anomalies.parquet"
    if not path.exists():
        raise HTTPException(404, "Anomaly data not found. Run anomaly_detector.py first.")
    df = pd.read_parquet(path).head(limit)
    return df.to_dict(orient="records")


@app.get("/products/summary")
def product_summary(n: int = 20):
    path = PROC_DIR / "features_df.parquet"
    if not path.exists():
        raise HTTPException(404, "Feature data not found.")
    df = pd.read_parquet(path).head(n)
    cols = ["product_id", "product_name", "category", "avg_defect_rate",
            "return_rate", "avg_rating", "quality_score"]
    avail = [c for c in cols if c in df.columns]
    return df[avail].to_dict(orient="records")
