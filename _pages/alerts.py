"""
Proactive Alerts Page — Anomaly Detection Results
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from components.charts import anomaly_score_gauge

PROC_DIR = ROOT / "data" / "processed"

SEVERITY_ORDER  = {"🔴 Critical": 0, "🟠 High": 1, "🟡 Medium": 2, "🟢 Normal": 3}
SEVERITY_COLORS = {"🔴 Critical": "#EF4444", "🟠 High": "#F59E0B", "🟡 Medium": "#FBBF24", "🟢 Normal": "#22D3A5"}


@st.cache_data(ttl=300)
def load_anomalies():
    path = PROC_DIR / "anomaly_scores.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def render():
    st.markdown("## 🚨 Proactive Quality Alerts")
    st.markdown(
        "Products flagged by **Isolation Forest** anomaly detection. "
        "Anomalies represent unusual combinations of defect rate, return rate, and customer sentiment."
    )

    full_df = load_anomalies()

    if full_df.empty:
        st.error("⚠️ Anomaly data not found. Run:\n```\npython src/models/anomaly_detector.py\n```")
        return

    total     = len(full_df)
    n_anomaly = int(full_df["is_anomaly"].sum()) if "is_anomaly" in full_df.columns else 0
    critical  = int((full_df.get("risk_severity", pd.Series()) == "🔴 Critical").sum())
    high      = int((full_df.get("risk_severity", pd.Series()) == "🟠 High").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products Monitored", f"{total:,}")
    c2.metric("🚨 Anomalies Detected",    f"{n_anomaly:,}", delta=f"{n_anomaly/total*100:.1f}% of portfolio", delta_color="inverse")
    c3.metric("🔴 Critical Risk",         f"{critical:,}", delta_color="inverse")
    c4.metric("🟠 High Risk",             f"{high:,}",     delta_color="inverse")

    st.divider()

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        chosen_sev = st.selectbox("Filter by Severity", ["All"] + list(SEVERITY_ORDER.keys()), key="alert_sev")
    with col_f2:
        cats = ["All"] + sorted(full_df["category"].dropna().unique().tolist()) if "category" in full_df.columns else ["All"]
        chosen_cat = st.selectbox("Filter by Category", cats, key="alert_cat")
    with col_f3:
        sups = ["All"] + sorted(full_df["supplier"].dropna().unique().tolist()) if "supplier" in full_df.columns else ["All"]
        chosen_sup = st.selectbox("Filter by Supplier", sups, key="alert_sup")

    disp = full_df[full_df["is_anomaly"] == 1].copy() if "is_anomaly" in full_df.columns else full_df.copy()
    if chosen_sev != "All" and "risk_severity" in disp.columns:
        disp = disp[disp["risk_severity"] == chosen_sev]
    if chosen_cat != "All" and "category" in disp.columns:
        disp = disp[disp["category"] == chosen_cat]
    if chosen_sup != "All" and "supplier" in disp.columns:
        disp = disp[disp["supplier"] == chosen_sup]
    if "risk_severity" in disp.columns:
        disp["_ord"] = disp["risk_severity"].map(SEVERITY_ORDER).fillna(9)
        disp = disp.sort_values("_ord").drop(columns=["_ord"])

    st.markdown(f"**Showing {len(disp)} alert(s)**")

    if "anomaly_score" in disp.columns and len(disp) > 0:
        top5 = disp.head(5)
        cols = st.columns(len(top5))
        for i, (_, row) in enumerate(top5.iterrows()):
            with cols[i]:
                name = str(row.get("product_name", row.get("product_id", "")))[:20]
                st.plotly_chart(anomaly_score_gauge(float(row["anomaly_score"]), name), use_container_width=True)

    st.divider()
    st.markdown("### 📋 Full Alert List")

    table_cols = [c for c in [
        "product_id", "product_name", "category", "supplier",
        "avg_defect_rate", "return_rate", "avg_sentiment",
        "avg_rating", "quality_score", "anomaly_score", "risk_severity",
    ] if c in disp.columns]

    if len(disp) > 0:
        fmt = {}
        if "avg_defect_rate" in table_cols: fmt["avg_defect_rate"] = "{:.2%}"
        if "return_rate"     in table_cols: fmt["return_rate"]     = "{:.2%}"
        if "avg_sentiment"   in table_cols: fmt["avg_sentiment"]   = "{:.3f}"
        if "avg_rating"      in table_cols: fmt["avg_rating"]      = "{:.2f}"
        if "quality_score"   in table_cols: fmt["quality_score"]   = "{:.1f}"
        if "anomaly_score"   in table_cols: fmt["anomaly_score"]   = "{:.4f}"

        def color_sev(val):
            c = SEVERITY_COLORS.get(str(val), "")
            return f"color: {c}; font-weight: bold" if c else ""

        styled = disp[table_cols].style.format(fmt)
        if "risk_severity" in table_cols:
            styled = styled.map(color_sev, subset=["risk_severity"])
        if "quality_score" in table_cols:
            styled = styled.background_gradient(subset=["quality_score"], cmap="RdYlGn")
        st.dataframe(styled, use_container_width=True, height=500)
    else:
        st.info("No anomalies match the selected filters.")

    if critical > 0:
        st.markdown("---")
        st.error(
            f"### ⚠️ Recommended Actions for {critical} Critical Product(s)\n"
            "1. **Immediate supplier audit** — Initiate quality review with flagged suppliers.\n"
            "2. **Hold new shipments** — Pause distribution until root cause is identified.\n"
            "3. **Customer outreach** — Proactively contact customers who purchased these products.\n"
            "4. **Production line inspection** — Review manufacturing logs for the affected batches."
        )
