"""
Feature Engineering
====================
Builds the final ML-ready feature matrix by joining:
  - master_df (product-level aggregations)
  - sentiment_per_product (VADER scores)
  - lag / rolling features from timeseries_df

Output: data/processed/features_df.parquet
        (used directly by ML models)

Run:  python src/features/engineering.py
"""

import os
import logging
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


# ── Rolling / Lag Features from Time-Series ───────────────────────────────────

def build_lag_features(ts: pd.DataFrame) -> pd.DataFrame:
    """
    For each product, compute rolling statistics across months:
      - 3-month & 6-month rolling return rate
      - momentum (change in return rate)
    """
    ts = ts.copy()
    ts["month_dt"] = pd.to_datetime(ts["month"])
    ts.sort_values(["product_id", "month_dt"], inplace=True)

    grp = ts.groupby("product_id")["return_rate_monthly"]

    ts["rolling_3m_return"]  = grp.transform(lambda x: x.rolling(3,  min_periods=1).mean())
    ts["rolling_6m_return"]  = grp.transform(lambda x: x.rolling(6,  min_periods=1).mean())
    ts["return_momentum"]    = grp.transform(lambda x: x.diff(1).fillna(0))

    # Take the LAST observation per product as summary
    latest = ts.sort_values("month_dt").groupby("product_id").last().reset_index()
    out = latest[[
        "product_id",
        "rolling_3m_return", "rolling_6m_return", "return_momentum",
    ]]
    log.info("Lag features built for %d products", len(out))
    return out


# ── Correlation Analysis ──────────────────────────────────────────────────────

def build_correlation_summary(master: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation of defect_rate, return_rate, avg_sentiment, avg_rating.
    Returns a tidy DataFrame for display in the dashboard.
    """
    cols = ["avg_defect_rate", "return_rate", "avg_sentiment",
            "avg_rating", "quality_score"]
    sub = master[cols].dropna()
    corr = sub.corr().round(3)
    return corr


# ── Supplier Risk Encoding ────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode supplier, production_line, category."""
    df = df.copy()
    for col in ["supplier", "production_line", "category"]:
        if col in df.columns:
            df[col + "_enc"] = pd.factorize(df[col].fillna("Unknown"))[0]
    return df


# ── Main Feature Matrix Builder ───────────────────────────────────────────────

def build_features() -> pd.DataFrame:
    master      = pd.read_parquet(os.path.join(PROC_DIR, "master_df.parquet"))
    ts          = pd.read_parquet(os.path.join(PROC_DIR, "timeseries_df.parquet"))
    sent_agg    = pd.read_parquet(os.path.join(PROC_DIR, "sentiment_per_product.parquet"))

    # Merge sentiment
    master = master.merge(sent_agg, on="product_id", how="left")
    master["avg_sentiment"].fillna(0, inplace=True)
    master["pct_positive"].fillna(0, inplace=True)
    master["pct_negative"].fillna(0, inplace=True)

    # Merge lag features
    lag = build_lag_features(ts)
    master = master.merge(lag, on="product_id", how="left")
    master["rolling_3m_return"].fillna(master["return_rate"], inplace=True)
    master["rolling_6m_return"].fillna(master["return_rate"], inplace=True)
    master["return_momentum"].fillna(0, inplace=True)

    # Encode categoricals
    master = encode_categoricals(master)

    # ── ML target labels ───────────────────────────────────────────────────────
    # Defect target: 1 if avg_defect_rate > 5%
    master["defect_risk_label"] = (master["avg_defect_rate"] > 0.05).astype(int)

    # Return target: continuous (total_return_qty) for regression
    master["return_qty_target"] = master["total_return_qty"]

    log.info("Feature matrix: %d rows × %d columns", master.shape[0], master.shape[1])
    log.info("  High-defect products: %d / %d",
             master["defect_risk_label"].sum(), len(master))

    return master


def main():
    feat = build_features()
    out_path = os.path.join(PROC_DIR, "features_df.parquet")
    feat.to_parquet(out_path, index=False)
    log.info("✅ features_df.parquet saved → %s", out_path)

    # Print correlation summary
    corr = build_correlation_summary(feat)
    print("\n📊 Cross-metric Correlation Matrix:")
    print(corr.to_string())

    print("\n📋 Feature Matrix Preview:")
    display_cols = [
        "product_id", "category", "avg_defect_rate", "return_rate",
        "avg_rating", "avg_sentiment", "quality_score", "defect_risk_label"
    ]
    print(feat[[c for c in display_cols if c in feat.columns]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
