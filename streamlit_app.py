"""
AI-Powered Product Quality Intelligence System
================================================
Main Streamlit Application Entry Point

Run from project root:
    streamlit run streamlit_app.py
"""

import sys
import os
from pathlib import Path

# ── Project root is THIS file's directory ────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quality Intelligence System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0F1A 0%, #141728 100%);
    border-right: 1px solid #2D3148;
}

/* Main area */
.main .block-container {
    background-color: #0F1117;
    padding-top: 1.5rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1D27 0%, #1E2235 100%);
    border: 1px solid #2D3148;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.15);
}
[data-testid="stMetricLabel"] { color: #9CA3AF !important; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #E2E8F0 !important; font-size: 1.6rem; font-weight: 600; }

/* Nav buttons in sidebar */
div[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: 1px solid #2D3148;
    border-radius: 8px;
    color: #9CA3AF;
    transition: all 0.2s ease;
    text-align: left;
    width: 100%;
    margin: 2px 0;
}
div[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(108, 99, 255, 0.15);
    border-color: #6C63FF;
    color: #E2E8F0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1A1D27;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #9CA3AF;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
    color: white !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    margin: 8px 0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2D3148;
    border-radius: 10px;
}

/* Section divider */
hr { border-color: #2D3148 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #1A1D27;
    border: 1px solid #2D3148;
    border-radius: 10px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0F1117; }
::-webkit-scrollbar-thumb { background: #2D3148; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6C63FF; }
</style>
""", unsafe_allow_html=True)


# ── Page Navigation ───────────────────────────────────────────────────────────

PAGES = {
    "📊 EDA & Insights":   "eda",
    "🤖 AI Chatbot":       "chatbot",
    "🚨 Proactive Alerts": "alerts",
    "📈 Predictions":      "predictions",
}

if "current_page" not in st.session_state:
    st.session_state.current_page = "eda"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.5rem">🏭</div>
        <h2 style="color:#E2E8F0;font-size:1.1rem;font-weight:700;margin:8px 0 2px;">
            Quality Intelligence
        </h2>
        <p style="color:#6B7280;font-size:0.75rem;margin:0">
            AI-Powered · Global Quality System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Navigation**")

    for label, key in PAGES.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.current_page = key

    st.divider()
    st.markdown("**Stack**")
    st.markdown("""
    <div style="font-size:0.75rem; color:#6B7280; line-height:1.8">
    🔵 XGBoost · Isolation Forest<br>
    🟣 LangChain · ChromaDB<br>
    🟢 Google Gemini Flash<br>
    🟡 VADER Sentiment<br>
    🔴 FastAPI · Streamlit<br>
    ⚪ MLflow · sentence-transformers
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("v1.0.0 · Built for Quality AI Internship Demo")


# ── Page Routing ──────────────────────────────────────────────────────────────
page = st.session_state.current_page

if page == "eda":
    from _pages.eda_insights import render
    render()

elif page == "chatbot":
    from _pages.chatbot import render
    render()

elif page == "alerts":
    from _pages.alerts import render
    render()

elif page == "predictions":
    from _pages.predictions import render
    render()
