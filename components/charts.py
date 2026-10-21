"""
Reusable Plotly Chart Components
==================================
All chart helpers used across Streamlit pages.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design Tokens ─────────────────────────────────────────────────────────────
PALETTE   = px.colors.qualitative.Vivid
BG        = "#0F1117"
CARD_BG   = "#1A1D27"
BORDER    = "#2D3148"
TEXT      = "#E2E8F0"
ACCENT    = "#6C63FF"
SUCCESS   = "#22D3A5"
WARNING   = "#F59E0B"
DANGER    = "#EF4444"

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    margin=dict(l=20, r=20, t=45, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)


def apply_base(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT_BASE)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def defect_rate_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = df.nlargest(top_n, "avg_defect_rate").sort_values("avg_defect_rate")
    label = top["product_name"] if "product_name" in top.columns else top["product_id"]
    fig = go.Figure(go.Bar(
        x=top["avg_defect_rate"] * 100,
        y=label,
        orientation="h",
        marker=dict(
            color=top["avg_defect_rate"] * 100,
            colorscale=[[0, SUCCESS], [0.5, WARNING], [1, DANGER]],
            showscale=True,
            colorbar=dict(title="Defect %"),
        ),
        hovertemplate="<b>%{y}</b><br>Defect Rate: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="🔴 Top Products by Defect Rate", xaxis_title="Defect Rate (%)")
    return apply_base(fig)


def return_rate_heatmap(df: pd.DataFrame) -> go.Figure:
    if "supplier" not in df.columns or "category" not in df.columns:
        return go.Figure()
    pivot = df.groupby(["category", "supplier"])["return_rate"].mean().reset_index()
    pivot_wide = pivot.pivot(index="category", columns="supplier", values="return_rate")
    fig = go.Figure(go.Heatmap(
        z=pivot_wide.values * 100,
        x=pivot_wide.columns.tolist(),
        y=pivot_wide.index.tolist(),
        colorscale="RdYlGn_r",
        hovertemplate="Category: %{y}<br>Supplier: %{x}<br>Return Rate: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return %"),
    ))
    fig.update_layout(
        title="📦 Return Rate Heatmap (Category × Supplier)",
        xaxis_title="Supplier",
        yaxis_title="Category",
    )
    return apply_base(fig)


def sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    if "sentiment_label" not in df.columns:
        return go.Figure()
    counts = df["sentiment_label"].value_counts()
    colors = {"Positive": SUCCESS, "Neutral": WARNING, "Negative": DANGER}
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=[colors.get(l, ACCENT) for l in counts.index]),
        hole=0.45,
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="💬 Customer Sentiment Distribution")
    return apply_base(fig)


def sentiment_vs_rating(df: pd.DataFrame) -> go.Figure:
    needed = {"avg_sentiment", "avg_rating", "quality_score"}
    if not needed.issubset(df.columns):
        return go.Figure()
    label_col = "product_name" if "product_name" in df.columns else "product_id"
    hover_data = {}
    if "avg_defect_rate" in df.columns: hover_data["avg_defect_rate"] = ":.3f"
    if "return_rate"     in df.columns: hover_data["return_rate"]     = ":.3f"
    fig = px.scatter(
        df.dropna(subset=list(needed)),
        x="avg_sentiment", y="avg_rating",
        color="quality_score",
        color_continuous_scale="Viridis",
        hover_name=label_col,
        hover_data=hover_data,
        labels={
            "avg_sentiment": "Avg Sentiment Score",
            "avg_rating":    "Avg Customer Rating",
            "quality_score": "Quality Score",
        },
    )
    fig.update_layout(title="⭐ Sentiment vs Rating (coloured by Quality Score)")
    return apply_base(fig)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols  = ["avg_defect_rate", "return_rate", "avg_sentiment",
             "avg_rating", "quality_score", "total_units_sold"]
    avail = [c for c in cols if c in df.columns]
    corr  = df[avail].corr().round(2)
    labels = [c.replace("_", " ").title() for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{x} × %{y}: %{z}<extra></extra>",
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(title="🔗 Cross-Metric Correlation Matrix")
    return apply_base(fig)


def monthly_trend(ts: pd.DataFrame, product_id: str = None) -> go.Figure:
    df = ts.copy()
    if product_id:
        df = df[df["product_id"] == product_id]
    df["month_dt"] = pd.to_datetime(df["month"])
    df = df.sort_values("month_dt")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df["month_dt"], y=df["units_sold"],
        name="Units Sold", line=dict(color=ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.1)",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["month_dt"], y=df["return_rate_monthly"] * 100,
        name="Return Rate %", line=dict(color=DANGER, width=2, dash="dot"),
    ), secondary_y=True)

    title = f"📈 Monthly Trend — {product_id}" if product_id else "📈 Aggregate Monthly Trend"
    fig.update_layout(title=title, **LAYOUT_BASE)
    fig.update_yaxes(title_text="Units Sold",     secondary_y=False, gridcolor=BORDER)
    fig.update_yaxes(title_text="Return Rate (%)", secondary_y=True,  gridcolor=BORDER)
    return fig


def quality_score_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.dropna(subset=["quality_score"]),
        x="quality_score", nbins=30,
        color_discrete_sequence=[ACCENT],
        labels={"quality_score": "Quality Score (0–100)"},
    )
    mean_val = df["quality_score"].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color=WARNING,
                  annotation_text=f"Mean: {mean_val:.1f}",
                  annotation_position="top right")
    fig.update_layout(title="📊 Quality Score Distribution")
    return apply_base(fig)


def anomaly_score_gauge(score: float, product_name: str = "") -> go.Figure:
    display = max(0, min(100, (score + 0.5) * 100))
    color   = DANGER if score < -0.1 else (WARNING if score < 0 else SUCCESS)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(display, 1),
        title={"text": f"Risk Score<br><span style='font-size:0.8em'>{product_name}</span>"},
        delta={"reference": 50, "increasing": {"color": DANGER}},
        gauge={
            "axis":    {"range": [0, 100], "tickcolor": TEXT},
            "bar":     {"color": color},
            "bgcolor": CARD_BG,
            "steps": [
                {"range": [0, 30],   "color": DANGER},
                {"range": [30, 60],  "color": WARNING},
                {"range": [60, 100], "color": SUCCESS},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": display},
        },
    ))
    fig.update_layout(paper_bgcolor=BG, font=dict(color=TEXT))
    return fig


def category_quality_box(df: pd.DataFrame) -> go.Figure:
    if "category" not in df.columns or "quality_score" not in df.columns:
        return go.Figure()
    fig = px.box(
        df.dropna(subset=["quality_score", "category"]),
        x="category", y="quality_score",
        color="category",
        color_discrete_sequence=PALETTE,
        labels={"quality_score": "Quality Score", "category": "Category"},
    )
    fig.update_layout(title="📦 Quality Score by Category", showlegend=False)
    return apply_base(fig)


def defect_probability_gauge(prob: float) -> go.Figure:
    color = DANGER if prob > 0.6 else (WARNING if prob > 0.35 else SUCCESS)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%"},
        title={"text": "Defect Risk Probability"},
        gauge={
            "axis":    {"range": [0, 100]},
            "bar":     {"color": color},
            "bgcolor": CARD_BG,
            "steps": [
                {"range": [0, 35],   "color": SUCCESS},
                {"range": [35, 60],  "color": WARNING},
                {"range": [60, 100], "color": DANGER},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor=BG, font=dict(color=TEXT),
                      margin=dict(l=20, r=20, t=60, b=20))
    return fig
