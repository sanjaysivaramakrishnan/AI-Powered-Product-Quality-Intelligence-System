"""
Predictions Page — Defect Risk & Return Forecasting
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from components.charts import defect_probability_gauge, BG, TEXT, ACCENT, WARNING, DANGER

MODELS_DIR = ROOT / "models"
API_URL    = "http://127.0.0.1:8000"

SUPPLIERS  = [f"Supplier_{chr(65+i)}" for i in range(10)]
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Sports", "Automotive"]
PROD_LINES = [f"Line_{i:02d}" for i in range(1, 9)]


def call_api(endpoint, payload):
    if not HAS_HTTPX:
        return None, "httpx not available"
    try:
        r = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=5)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def predict_local_defect(payload):
    path = MODELS_DIR / "defect_predictor.joblib"
    if not path.exists():
        return None, "defect_predictor.joblib not found — run training pipeline first."
    bundle = joblib.load(path)
    model, features = bundle["model"], bundle["features"]
    X = pd.DataFrame([{f: payload.get(f, 0.0) for f in features}])[features].astype(float)
    proba = float(model.predict_proba(X)[0][1])
    return {"defect_probability": round(proba, 4), "risk_label": "High Risk" if proba > 0.5 else "Low Risk",
            "confidence": round(max(proba, 1 - proba), 4), "top_risk_factors": features[:3]}, None


def predict_local_returns(payload):
    path = MODELS_DIR / "return_forecaster.joblib"
    if not path.exists():
        return None, "return_forecaster.joblib not found — run training pipeline first."
    bundle = joblib.load(path)
    model, features = bundle["model"], bundle["features"]
    X = pd.DataFrame([{f: payload.get(f, 0.0) for f in features}])[features].astype(float)
    pred_qty = float(max(0, model.predict(X)[0]))
    units = payload.get("total_units_sold", 1)
    return {"predicted_return_qty": round(pred_qty, 1),
            "predicted_return_rate_pct": round(pred_qty / max(units, 1) * 100, 2),
            "risk_level": "High" if pred_qty > 50 else ("Medium" if pred_qty > 20 else "Low")}, None


def return_forecast_chart(predicted_qty, units_sold):
    fig = go.Figure(go.Bar(
        x=["Units Sold", "Predicted Returns"],
        y=[units_sold, predicted_qty],
        marker_color=[ACCENT, DANGER if predicted_qty / max(units_sold, 1) > 0.1 else WARNING],
        text=[f"{units_sold:,.0f}", f"{predicted_qty:,.1f}"],
        textposition="outside",
    ))
    fig.update_layout(title="📦 Predicted Return Volume vs Sales", yaxis_title="Quantity",
                      paper_bgcolor=BG, plot_bgcolor="#1A1D27", font=dict(color=TEXT),
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render():
    st.markdown("## 📈 ML Predictions")
    st.markdown("Use the trained **XGBoost** models to predict defect risk and forecast return volumes.")

    tab_defect, tab_return = st.tabs(["🔴 Defect Risk Predictor", "📦 Return Volume Forecaster"])

    with tab_defect:
        st.markdown("### Configure Product Scenario")
        c1, c2, c3 = st.columns(3)
        with c1:
            supplier  = st.selectbox("Supplier",        SUPPLIERS,  key="d_sup")
            category  = st.selectbox("Category",        CATEGORIES, key="d_cat")
            prod_line = st.selectbox("Production Line", PROD_LINES, key="d_line")
        with c2:
            avg_lead     = st.slider("Avg Lead Time (days)", 3,   90,  15,  key="d_lead")
            n_batches    = st.slider("Number of Batches",    1,   50,  10,  key="d_batch")
            avg_discount = st.slider("Avg Discount (%)",     0,   40,  10,  key="d_disc") / 100.0
        with c3:
            avg_sent    = st.slider("Avg Sentiment Score", -1.0, 1.0,  0.1, 0.01, key="d_sent")
            pct_neg     = st.slider("% Negative Reviews",   0,   50,  10,   key="d_neg")  / 100.0
            return_rate = st.slider("Return Rate (%)",       0,   30,   5,   key="d_ret")  / 100.0
            n_reviews   = st.slider("Number of Reviews",    1,  500,  50,   key="d_nrev")

        payload = {
            "supplier_enc": float(SUPPLIERS.index(supplier)),
            "production_line_enc": float(PROD_LINES.index(prod_line)),
            "category_enc": float(CATEGORIES.index(category)),
            "avg_lead_time": float(avg_lead), "n_batches": float(n_batches),
            "avg_discount": avg_discount, "avg_sentiment": avg_sent,
            "pct_negative": pct_neg, "rolling_3m_return": return_rate,
            "return_rate": return_rate, "n_reviews": float(n_reviews),
        }

        if st.button("🔮 Predict Defect Risk", type="primary", key="btn_defect"):
            with st.spinner("Running prediction…"):
                result, err = call_api("/predict/defect", payload)
                if err:
                    result, err = predict_local_defect(payload)
                if err:
                    st.error(err); st.stop()

            col_g, col_i = st.columns([2, 3])
            with col_g:
                st.plotly_chart(defect_probability_gauge(result["defect_probability"]), use_container_width=True)
            with col_i:
                label = result["risk_label"]
                color = "red" if "High" in label else "green"
                st.markdown(f"### Risk Label: :{color}[**{label}**]")
                st.metric("Defect Probability", f"{result['defect_probability']*100:.1f}%")
                st.metric("Model Confidence",   f"{result['confidence']*100:.1f}%")
                if result.get("top_risk_factors"):
                    st.markdown("**📌 Top Contributing Factors:**")
                    for f in result["top_risk_factors"]:
                        st.markdown(f"- `{f.replace('_', ' ').title()}`")
                if "High" in label:
                    st.error("**⚠️ Recommended Actions:**\n- Audit supplier\n- Increase QC inspection\n- Review production line\n- Reduce lead time")

    with tab_return:
        st.markdown("### Configure Product Scenario")
        c1, c2, c3 = st.columns(3)
        with c1:
            r_supplier = st.selectbox("Supplier",        SUPPLIERS,  key="r_sup")
            r_category = st.selectbox("Category",        CATEGORIES, key="r_cat")
            r_line     = st.selectbox("Production Line", PROD_LINES, key="r_line")
        with c2:
            units_sold  = st.number_input("Total Units Sold", 50, 10000, 500, step=50, key="r_units")
            avg_rating  = st.slider("Avg Rating (1-5)",  1.0, 5.0, 3.5, 0.1, key="r_rating")
            avg_sent_r  = st.slider("Avg Sentiment",    -1.0, 1.0, 0.0, 0.01, key="r_sent")
        with c3:
            pct_neg_r   = st.slider("% Negative Reviews",     0, 50, 15, key="r_neg") / 100.0
            defect_rate = st.slider("Avg Defect Rate (%)",     0, 30,  3, key="r_def") / 100.0
            rolling_ret = st.slider("Rolling 3m Return Rate (%)", 0, 30, 5, key="r_roll") / 100.0

        r_payload = {
            "supplier_enc": float(SUPPLIERS.index(r_supplier)),
            "production_line_enc": float(PROD_LINES.index(r_line)),
            "category_enc": float(CATEGORIES.index(r_category)),
            "total_units_sold": float(units_sold), "avg_rating": avg_rating,
            "avg_sentiment": avg_sent_r, "pct_negative": pct_neg_r,
            "avg_defect_rate": defect_rate, "defect_risk_label": float(defect_rate > 0.05),
            "rolling_3m_return": rolling_ret, "rolling_6m_return": rolling_ret,
            "return_momentum": 0.0, "avg_discount": 0.1, "avg_lead_time": 15.0,
        }

        if st.button("🔮 Forecast Return Volume", type="primary", key="btn_return"):
            with st.spinner("Running forecast…"):
                result_r, err_r = call_api("/predict/returns", r_payload)
                if err_r:
                    result_r, err_r = predict_local_returns(r_payload)
                if err_r:
                    st.error(err_r); st.stop()

            col_chart, col_stats = st.columns([3, 2])
            with col_chart:
                st.plotly_chart(return_forecast_chart(result_r["predicted_return_qty"], float(units_sold)), use_container_width=True)
            with col_stats:
                risk = result_r["risk_level"]
                colors = {"High": "red", "Medium": "orange", "Low": "green"}
                c = colors.get(risk, "gray")
                st.markdown(f"### Risk: :{c}[**{risk}**]")
                st.metric("Predicted Return Qty",  f"{result_r['predicted_return_qty']:,.1f} units")
                st.metric("Predicted Return Rate", f"{result_r['predicted_return_rate_pct']:.2f}%")
                if risk == "High":
                    st.error("**⚠️ Actions:** Review description accuracy · Improve packaging · Investigate return reasons")
                elif risk == "Medium":
                    st.warning("Monitor return trends weekly for this product.")
                else:
                    st.success("✅ Return volume is within acceptable range.")
