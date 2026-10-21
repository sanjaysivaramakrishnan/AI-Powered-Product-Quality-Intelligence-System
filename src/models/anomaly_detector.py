"""
Anomaly Detector — Proactive Warning System
============================================
Uses Isolation Forest to detect products with anomalous
quality signals across defect rate, return rate, and sentiment.

Outputs: data/processed/anomalies.parquet
         (loaded directly by the Alerts tab in Streamlit)

Run:  python src/models/anomaly_detector.py
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

ANOMALY_FEATURES = [
    "avg_defect_rate",
    "return_rate",
    "avg_sentiment",
    "pct_negative",
    "rolling_3m_return",
    "defect_return_rate",
]

RISK_THRESHOLDS = {
    "critical": -0.15,   # anomaly score below this → Critical
    "high":     -0.05,   # below this → High Risk
    "medium":    0.00,   # below this → Medium
}


def load_data() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "features_df.parquet")
    df = pd.read_parquet(path)
    for col in ANOMALY_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def severity_label(score: float) -> str:
    if score < RISK_THRESHOLDS["critical"]:
        return "🔴 Critical"
    elif score < RISK_THRESHOLDS["high"]:
        return "🟠 High"
    elif score < RISK_THRESHOLDS["medium"]:
        return "🟡 Medium"
    return "🟢 Normal"


def train_and_score(df: pd.DataFrame):
    os.makedirs(MODELS_DIR, exist_ok=True)

    feat_cols = [c for c in ANOMALY_FEATURES if c in df.columns]
    X = df[feat_cols].astype(float)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=200,
            contamination=0.12,   # ~12% flagged as anomalous
            random_state=42,
            n_jobs=-1,
        )),
    ])

    pipe.fit(X)

    # Isolation Forest: +1 = normal, -1 = anomaly
    df["anomaly_flag"]   = pipe.predict(X)           # -1 or +1
    df["anomaly_score"]  = pipe.named_steps["iforest"].decision_function(
        pipe.named_steps["scaler"].transform(X)
    ).round(4)                                        # continuous score
    df["is_anomaly"]     = (df["anomaly_flag"] == -1).astype(int)
    df["risk_severity"]  = df["anomaly_score"].apply(severity_label)

    n_anomalies = df["is_anomaly"].sum()
    log.info("Anomalies detected: %d / %d (%.1f%%)",
             n_anomalies, len(df), 100 * n_anomalies / len(df))

    # Save model
    model_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
    joblib.dump({"pipeline": pipe, "features": feat_cols}, model_path)
    log.info("✅ Anomaly model saved → %s", model_path)

    return df, pipe


def build_alert_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a clean alert table for the dashboard:
    only anomalous products with key metrics displayed.
    """
    alerts = df[df["is_anomaly"] == 1].copy()
    display_cols = [
        "product_id", "product_name", "category", "supplier",
        "avg_defect_rate", "return_rate", "avg_sentiment",
        "avg_rating", "quality_score",
        "anomaly_score", "risk_severity",
    ]
    avail = [c for c in display_cols if c in alerts.columns]
    alerts = alerts[avail].sort_values("anomaly_score").reset_index(drop=True)

    # Human-readable formatting
    for pct_col in ["avg_defect_rate", "return_rate"]:
        if pct_col in alerts.columns:
            alerts[pct_col] = (alerts[pct_col] * 100).round(2).astype(str) + "%"

    alerts.rename(columns={
        "product_id":     "Product ID",
        "product_name":   "Product",
        "category":       "Category",
        "supplier":       "Supplier",
        "avg_defect_rate":"Defect Rate",
        "return_rate":    "Return Rate",
        "avg_sentiment":  "Avg Sentiment",
        "avg_rating":     "Avg Rating",
        "quality_score":  "Quality Score",
        "anomaly_score":  "Anomaly Score",
        "risk_severity":  "Risk Level",
    }, inplace=True)

    return alerts


def main():
    df = load_data()
    df_scored, _ = train_and_score(df)

    # Save full scored dataset
    df_scored.to_parquet(
        os.path.join(PROC_DIR, "anomaly_scores.parquet"), index=False
    )
    log.info("✅ anomaly_scores.parquet saved")

    # Save alert table
    alert_table = build_alert_table(df_scored)
    alert_table.to_parquet(
        os.path.join(PROC_DIR, "anomalies.parquet"), index=False
    )
    log.info("✅ anomalies.parquet saved → %d alerts", len(alert_table))

    print("\n🚨 Top 10 Anomalous Products:")
    print(alert_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
