"""
AI Chatbot Page — LangChain RAG
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

STARTER_QUESTIONS = [
    "Which products have the highest defect rates?",
    "What are the most common reasons customers are returning products?",
    "Which suppliers are associated with the most quality issues?",
    "What is the relationship between sentiment scores and return rates?",
    "Which product categories have the lowest quality scores?",
    "Give me a summary of the top 5 highest-risk products right now.",
    "What actions would you recommend to reduce defect rates?",
    "Which production lines have the worst quality track record?",
]


def init_state():
    if "messages"    not in st.session_state: st.session_state.messages    = []
    if "agent"       not in st.session_state: st.session_state.agent       = None
    if "agent_error" not in st.session_state: st.session_state.agent_error = None


@st.cache_resource(show_spinner="🔧 Loading AI Quality Agent…")
def load_agent():
    from src.rag.agent import QualityAgent
    return QualityAgent()


def _show_sources(sources: list):
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} source(s) retrieved"):
        for s in sources:
            st.markdown(
                f"**[{s['doc_type'].replace('_', ' ').title()}]** "
                f"`{s['product_id']}` — {s['snippet']}"
            )


def _display_result(result: dict):
    """Render one assistant result, showing rate-limit warnings differently."""
    answer = result.get("answer", "")
    error  = result.get("error")

    if error == "rate_limit":
        st.warning(answer)          # yellow box for quota warnings
    elif error and error not in ("rate_limit", None):
        st.error(answer)            # red box for real errors
    else:
        st.markdown(answer)         # normal rendering
        _show_sources(result.get("sources", []))


def render():
    init_state()

    st.markdown("## 🤖 AI Quality Intelligence Chatbot")
    st.markdown(
        "Ask anything about product quality, defects, returns, or supplier performance. "
        "Powered by **LangChain + Google Gemini** with Retrieval-Augmented Generation."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Chatbot Controls")
        if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_chat"):
            st.session_state.messages = []
            if st.session_state.agent:
                st.session_state.agent.reset()
            st.rerun()

        # Show which model is active
        if st.session_state.agent:
            model = getattr(st.session_state.agent, "_model_name", "unknown")
            st.caption(f"🟢 Model: `{model}`")

        st.markdown("---")
        st.markdown("**💡 Starter Questions**")
        for q in STARTER_QUESTIONS[:5]:
            if st.button(q, key=f"starter_{q[:20]}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.session_state["_pending_query"] = q
                st.rerun()

        st.markdown("---")
        st.markdown("**⚙️ Model Config**")
        st.caption(
            "Edit `GEMINI_MODEL` in `.env`:\n"
            "- `gemini-1.5-flash`\n"
            "- `gemini-1.5-flash-8b`\n"
            "- `gemini-1.0-pro`"
        )

    # ── Load Agent ────────────────────────────────────────────────────────────
    if st.session_state.agent is None and st.session_state.agent_error is None:
        try:
            st.session_state.agent = load_agent()
        except Exception as e:
            st.session_state.agent_error = str(e)

    if st.session_state.agent_error:
        st.error(f"⚠️ Agent failed to load:\n\n`{st.session_state.agent_error}`")
        st.info(
            "**Checklist:**\n"
            "1. `GOOGLE_API_KEY` is set in `.env`\n"
            "2. Vector store is built: `python src/rag/vectorstore.py`\n"
            "3. Model name is valid in `.env` (try `gemini-1.5-flash`)"
        )
        if st.button("🔄 Retry", key="retry_agent"):
            st.session_state.agent_error = None
            load_agent.clear()
            st.rerun()
        return

    # ── Chat History Display ──────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            """<div style="text-align:center;padding:60px 20px;color:#6B7280">
                <div style="font-size:3rem">🤖</div>
                <h3 style="color:#9CA3AF">Hello! I'm your Quality Intelligence Assistant.</h3>
                <p>Ask me about product defects, return patterns, supplier performance, or customer sentiment.</p>
                <p style="font-size:0.8rem;margin-top:20px">Use the starter questions in the sidebar ←</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                if msg["role"] == "assistant":
                    _display_result(msg)
                else:
                    st.markdown(msg["content"])

    # ── Handle Sidebar Button Press ───────────────────────────────────────────
    pending = st.session_state.pop("_pending_query", None)
    if pending and st.session_state.agent:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analysing quality data…"):
                result = st.session_state.agent.chat(pending)
            _display_result(result)
        st.session_state.messages.append({
            "role": "assistant", **result
        })
        st.rerun()

    # ── Chat Input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about product quality, defects, returns…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analysing quality data…"):
                result = st.session_state.agent.chat(prompt)
            _display_result(result)
        st.session_state.messages.append({
            "role": "assistant", **result
        })
