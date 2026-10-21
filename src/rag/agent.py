"""
LangChain RAG Agent  (LangChain 1.x / LCEL-based)
===================================================
Conversational AI agent answering product quality questions
using ChromaDB retrieval + Google Gemini Flash.

Usage:
    from src.rag.agent import QualityAgent
    agent = QualityAgent()
    result = agent.chat("Which products have the highest defect rates?")
    print(result["answer"])
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI Quality Intelligence Analyst at a global manufacturing company.
You have access to comprehensive product data including:
- Manufacturing defect rates by supplier and production line
- Customer return rates and reasons
- Customer review sentiments and ratings
- Sales performance and revenue data
- Anomaly detection results

Your role is to:
1. Answer questions about product quality, defects, returns, and customer satisfaction
2. Identify root causes of quality issues
3. Provide actionable recommendations for quality improvement
4. Highlight high-risk products and suppliers
5. Summarise trends and correlations in the data

Always base your answers on the retrieved context below.
Be specific, data-driven, and concise.
If you cannot find relevant information in the context, say so clearly.
Format your responses with bullet points and clear headings when listing multiple items.

--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---
"""

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


# ── Agent Class ───────────────────────────────────────────────────────────────

class QualityAgent:
    """
    Conversational RAG agent using LCEL. No probing at init —
    the LLM is only called when chat() is invoked.
    """

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at: https://aistudio.google.com/"
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        # Initialise LLM object — no API call made here
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            max_retries=1,      # built-in retry inside langchain
        )
        self._model_name = GEMINI_MODEL

        # Retriever (ChromaDB — no API calls here either)
        from src.rag.vectorstore import get_retriever
        self.retriever = get_retriever(k=6)

        self._history: list[dict] = []
        log.info("✅ QualityAgent initialised — model: %s", self._model_name)

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, question: str) -> dict:
        """
        Returns {"answer": str, "sources": list[dict], "error": str|None}.
        Never raises — quota / rate-limit errors are returned as {"error": ...}.
        """
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_core.output_parsers import StrOutputParser

        # 1. Retrieve context
        try:
            docs    = self.retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            log.error("Retrieval failed: %s", e)
            return {"answer": "", "sources": [], "error": f"Retrieval error: {e}"}

        # 2. Build history messages
        history_messages = []
        for turn in self._history[-10:]:
            if turn["role"] == "user":
                history_messages.append(HumanMessage(content=turn["content"]))
            else:
                history_messages.append(AIMessage(content=turn["content"]))

        # 3. Build LCEL chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        # 4. Invoke — catch quota / rate-limit gracefully
        try:
            answer = chain.invoke({
                "context":  context,
                "history":  history_messages,
                "question": question,
            })
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                # Try to extract retry delay from message
                import re
                delay_m = re.search(r"retry[^\d]*(\d+)", err_str, re.IGNORECASE)
                delay   = delay_m.group(1) + "s" if delay_m else "~60s"
                user_msg = (
                    f"⏳ **API rate limit reached** (model: `{self._model_name}`).\n\n"
                    f"Please wait **{delay}** and try again.\n\n"
                    "**Why?** Google's free tier has per-minute and per-day quotas. "
                    "You can:\n"
                    "- Wait and retry\n"
                    "- Get a new API key at [aistudio.google.com](https://aistudio.google.com)\n"
                    "- Try the `gemini-1.5-flash-8b` model (lighter quota) by updating `GEMINI_MODEL` in `.env`"
                )
                log.warning("Rate limited: %s", err_str[:200])
                return {"answer": user_msg, "sources": [], "error": "rate_limit"}
            elif "NOT_FOUND" in err_str or "404" in err_str:
                user_msg = (
                    f"❌ **Model not found**: `{self._model_name}`\n\n"
                    "Update `GEMINI_MODEL` in your `.env` file to one of:\n"
                    "- `gemini-1.5-flash`\n- `gemini-1.5-flash-8b`\n- `gemini-1.0-pro`"
                )
                return {"answer": user_msg, "sources": [], "error": "not_found"}
            else:
                log.error("LLM error: %s", err_str)
                return {"answer": f"❌ LLM error: {err_str[:300]}", "sources": [], "error": err_str}

        # 5. Update history
        self._history.append({"role": "user",      "content": question})
        self._history.append({"role": "assistant",  "content": answer})

        sources = [
            {
                "doc_type":   d.metadata.get("doc_type", "unknown"),
                "product_id": d.metadata.get("product_id", ""),
                "snippet":    d.page_content[:200] + "…",
            }
            for d in docs
        ]
        return {"answer": answer, "sources": sources, "error": None}

    def reset(self):
        self._history.clear()
        log.info("Conversation history cleared.")


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = QualityAgent()
    print(f"🤖 Quality AI Agent [{agent._model_name}] — Type 'quit' to exit\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        result = agent.chat(q)
        print(f"\nAgent: {result['answer']}\n")
