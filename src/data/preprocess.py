"""
Data Preprocessing Pipeline
============================
Reads raw CSVs, cleans and merges them into a single master DataFrame,
then saves it as a Parquet file for downstream use.

Run:  python src/data/preprocess.py
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


# ── Individual Loaders ────────────────────────────────────────────────────────

def load_products() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "products.csv"), parse_dates=["launch_date"])
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df.dropna(subset=["product_id", "category"], inplace=True)
    log.info("Products loaded: %d rows", len(df))
    return df


def load_sales() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "sales.csv"), parse_dates=["date"])
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).astype(int)
    df["revenue"]    = pd.to_numeric(df["revenue"],    errors="coerce").fillna(0.0)
    df.rename(columns={"date": "sale_date"}, inplace=True)
    log.info("Sales loaded: %d rows", len(df))
    return df


def load_returns() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "returns.csv"), parse_dates=["date"])
    df["return_qty"] = pd.to_numeric(df["return_qty"], errors="coerce").fillna(0).astype(int)
    df["defect_type"] = df["defect_type"].fillna("None")
    df["is_defect_return"] = (df["defect_type"] != "None").astype(int)
    df.rename(columns={"date": "return_date"}, inplace=True)
    log.info("Returns loaded: %d rows", len(df))
    return df


def load_reviews() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "reviews.csv"), parse_dates=["date"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(1, 5)
    df["review_text"] = df["review_text"].fillna("")
    df.rename(columns={"date": "review_date"}, inplace=True)
    log.info("Reviews loaded: %d rows", len(df))
    return df


def load_manufacturing() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "manufacturing.csv"), parse_dates=["production_date"])
    df["defect_rate"] = pd.to_numeric(df["defect_rate"], errors="coerce").clip(0, 1).fillna(0)
    log.info("Manufacturing loaded: %d rows", len(df))
    return df


# ── Product-level Aggregations ────────────────────────────────────────────────

def aggregate_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales per product."""
    agg = sales.groupby("product_id").agg(
        total_units_sold=("units_sold", "sum"),
        total_revenue=("revenue", "sum"),
        avg_discount=("discount_pct", "mean"),
        n_sales_records=("sale_id", "count"),
    ).reset_index()
    return agg


def aggregate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate returns per product."""
    agg = returns.groupby("product_id").agg(
        total_return_qty=("return_qty", "sum"),
        n_return_records=("return_id", "count"),
        total_refund=("refund_amt", "sum"),
        defect_return_count=("is_defect_return", "sum"),
    ).reset_index()
    return agg


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reviews per product."""
    agg = reviews.groupby("product_id").agg(
        avg_rating=("rating", "mean"),
        n_reviews=("review_id", "count"),
        pct_verified=("verified_purchase", "mean"),
    ).reset_index()
    agg["avg_rating"] = agg["avg_rating"].round(2)
    return agg


def aggregate_manufacturing(mfg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate manufacturing per product (latest supplier, avg defect rate)."""
    # Latest batch per product → representative snapshot
    latest = mfg.sort_values("production_date").groupby("product_id").last().reset_index()
    agg = mfg.groupby("product_id").agg(
        avg_defect_rate=("defect_rate", "mean"),
        n_batches=("batch_id", "count"),
        total_batch_size=("batch_size", "sum"),
        avg_lead_time=("lead_time_days", "mean"),
        defect_flag_count=("high_defect_flag", "sum"),
    ).reset_index()
    agg = agg.merge(
        latest[["product_id", "supplier", "production_line"]],
        on="product_id", how="left"
    )
    return agg


# ── Master Merge ──────────────────────────────────────────────────────────────

def build_master(
    products:      pd.DataFrame,
    sales_agg:     pd.DataFrame,
    returns_agg:   pd.DataFrame,
    reviews_agg:   pd.DataFrame,
    mfg_agg:       pd.DataFrame,
) -> pd.DataFrame:
    master = products.copy()
    master = master.merge(sales_agg,   on="product_id", how="left")
    master = master.merge(returns_agg, on="product_id", how="left")
    master = master.merge(reviews_agg, on="product_id", how="left")
    master = master.merge(mfg_agg,     on="product_id", how="left")

    # Fill products with no sales / returns / reviews with 0
    fill_zero = [
        "total_units_sold", "total_revenue", "n_sales_records",
        "total_return_qty", "n_return_records", "total_refund", "defect_return_count",
        "n_reviews", "avg_rating",
        "avg_defect_rate", "n_batches", "total_batch_size", "defect_flag_count",
    ]
    for col in fill_zero:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    # Derived KPIs
    master["return_rate"] = np.where(
        master["total_units_sold"] > 0,
        master["total_return_qty"] / master["total_units_sold"],
        0.0
    ).round(4)

    master["defect_return_rate"] = np.where(
        master["n_return_records"] > 0,
        master["defect_return_count"] / master["n_return_records"],
        0.0
    ).round(4)

    master["quality_score"] = (
        master["avg_rating"] * 20          # 0-100
        - master["return_rate"] * 100      # penalise returns
        - master["avg_defect_rate"] * 100  # penalise defects
    ).clip(0, 100).round(2)

    log.info("Master DataFrame built: %d products × %d features",
             len(master), master.shape[1])
    return master


# ── Time-Series Dataset ───────────────────────────────────────────────────────

def build_timeseries(
    sales: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Monthly aggregation per product for time-series analysis."""
    sales["month"] = sales["sale_date"].dt.to_period("M")
    returns["month"] = returns["return_date"].dt.to_period("M")

    s_monthly = sales.groupby(["product_id", "month"]).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()

    r_monthly = returns.groupby(["product_id", "month"]).agg(
        return_qty=("return_qty", "sum"),
        n_returns=("return_id", "count"),
    ).reset_index()

    ts = s_monthly.merge(r_monthly, on=["product_id", "month"], how="outer")
    ts.fillna(0, inplace=True)
    ts["return_rate_monthly"] = np.where(
        ts["units_sold"] > 0,
        ts["return_qty"] / ts["units_sold"], 0.0
    ).round(4)
    ts["month"] = ts["month"].astype(str)
    return ts


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    products = load_products()
    sales    = load_sales()
    returns  = load_returns()
    reviews  = load_reviews()
    mfg      = load_manufacturing()

    sales_agg   = aggregate_sales(sales)
    returns_agg = aggregate_returns(returns)
    reviews_agg = aggregate_reviews(reviews)
    mfg_agg     = aggregate_manufacturing(mfg)

    master = build_master(products, sales_agg, returns_agg, reviews_agg, mfg_agg)
    master.to_parquet(os.path.join(PROC_DIR, "master_df.parquet"), index=False)
    log.info("✅ master_df.parquet saved")

    ts = build_timeseries(sales, returns)
    ts.to_parquet(os.path.join(PROC_DIR, "timeseries_df.parquet"), index=False)
    log.info("✅ timeseries_df.parquet saved")

    # Save raw cleaned CSVs for Streamlit display
    reviews.to_parquet(os.path.join(PROC_DIR, "reviews_clean.parquet"), index=False)
    mfg.to_parquet(os.path.join(PROC_DIR, "manufacturing_clean.parquet"), index=False)

    print("\n🎉 Preprocessing complete! Files in:", os.path.abspath(PROC_DIR))
    print(master.describe().to_string())


if __name__ == "__main__":
    main()
