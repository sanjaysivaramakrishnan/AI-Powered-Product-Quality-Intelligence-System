"""
EDA & Insights Page
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from components.charts import (
    defect_rate_bar, return_rate_heatmap, sentiment_distribution,
    sentiment_vs_rating, correlation_heatmap, monthly_trend,
    quality_score_histogram, category_quality_box,
)

PROC_DIR = ROOT / "data" / "processed"


@st.cache_data(ttl=600)
def load_data():
    master = pd.read_parquet(PROC_DIR / "features_df.parquet")
    ts     = pd.read_parquet(PROC_DIR / "timeseries_df.parquet")
    try:
        reviews = pd.read_parquet(PROC_DIR / "reviews_sentiment.parquet")
    except FileNotFoundError:
        reviews = pd.DataFrame()
    return master, ts, reviews


def render():
    st.markdown("## 📊 EDA & Quality Insights")
    st.markdown("Explore product quality metrics, correlations, and trends across all data sources.")

    try:
        master, ts, reviews = load_data()
    except FileNotFoundError:
        st.error("⚠️ Processed data not found. Run the pipeline first:\n```\npython run_pipeline.py\n```")
        return
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Products",    f"{len(master):,}")
    c2.metric("Avg Defect Rate",   f"{master['avg_defect_rate'].mean()*100:.2f}%")
    c3.metric("Avg Return Rate",   f"{master['return_rate'].mean()*100:.2f}%")
    c4.metric("Avg Rating",        f"{master['avg_rating'].mean():.2f} ⭐")
    c5.metric("Avg Quality Score", f"{master['quality_score'].mean():.1f}")

    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        top_n = st.slider("Top N Products", 5, 30, 15, key="defect_topn")
        st.plotly_chart(defect_rate_bar(master, top_n), use_container_width=True)
    with col2:
        st.plotly_chart(quality_score_histogram(master), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(return_rate_heatmap(master), use_container_width=True)
    with col4:
        st.plotly_chart(category_quality_box(master), use_container_width=True)

    col5, col6 = st.columns([1, 2])
    with col5:
        if not reviews.empty:
            st.plotly_chart(sentiment_distribution(reviews), use_container_width=True)
        else:
            st.info("Run sentiment scoring to see this chart.")
    with col6:
        st.plotly_chart(sentiment_vs_rating(master), use_container_width=True)

    st.plotly_chart(correlation_heatmap(master), use_container_width=True)

    st.markdown("### 📈 Monthly Trend Explorer")
    all_pids = ["All Products"] + sorted(master["product_id"].unique().tolist())
    chosen = st.selectbox("Select Product", all_pids, key="ts_product")
    pid = None if chosen == "All Products" else chosen
    st.plotly_chart(monthly_trend(ts, pid), use_container_width=True)

    with st.expander("📋 View Raw Product Data"):
        display_cols = [c for c in [
            "product_id", "product_name", "category", "supplier",
            "avg_defect_rate", "return_rate", "avg_rating",
            "avg_sentiment", "quality_score",
        ] if c in master.columns]
        fmt = {}
        if "avg_defect_rate" in display_cols: fmt["avg_defect_rate"] = "{:.2%}"
        if "return_rate"     in display_cols: fmt["return_rate"]     = "{:.2%}"
        if "avg_rating"      in display_cols: fmt["avg_rating"]      = "{:.2f}"
        if "avg_sentiment"   in display_cols: fmt["avg_sentiment"]   = "{:.3f}"
        if "quality_score"   in display_cols: fmt["quality_score"]   = "{:.1f}"
        st.dataframe(
            master[display_cols].sort_values("quality_score").style
            .background_gradient(subset=[c for c in ["avg_defect_rate","return_rate"] if c in display_cols], cmap="RdYlGn_r")
            .background_gradient(subset=[c for c in ["quality_score","avg_rating"] if c in display_cols], cmap="RdYlGn")
            .format(fmt),
            use_container_width=True, height=400,
        )
