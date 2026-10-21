# 🏭 AI-Powered Product Quality Intelligence System

> **A production-grade ML + LangChain system for global product quality monitoring, anomaly detection, and conversational AI analysis.**

Built as a portfolio demonstration for the **Global Quality AI/ML Internship** — showcasing end-to-end ML engineering, RAG-based conversational AI, and proactive quality alerting.

---

## 📐 System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                               │
│          Sales · Returns · Reviews · Manufacturing                 │
└──────────────────────────────┬─────────────────────────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                           │
│           Cleaning · Merging · EDA · Sentiment Scoring             │
│               src/data/ · src/features/                            │
└──────────────────────────────┬─────────────────────────────────────┘
                               ↓
           ┌───────────────────┴────────────────────┐
           ↓                                        ↓
┌─────────────────────┐               ┌─────────────────────────┐
│    ML MODELS        │               │   VECTOR DATABASE       │
│  XGBoost Classifier │               │   ChromaDB + MiniLM     │
│  XGBoost Regressor  │               │   (Product + Reviews +  │
│  Isolation Forest   │               │    Manufacturing Docs)  │
└────────┬────────────┘               └──────────┬──────────────┘
         ↓                                       ↓
┌─────────────────────┐               ┌─────────────────────────┐
│  FastAPI Prediction │               │  LangChain RAG Pipeline │
│  /predict/defect    │               │  Gemini Flash + Memory  │
│  /predict/returns   │               │  ConversationalChain    │
│  /anomalies         │               └──────────┬──────────────┘
└────────┬────────────┘                          │
         └─────────────────────┬─────────────────┘
                               ↓
┌────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                             │
│   📊 EDA & Insights  │  🤖 AI Chatbot  │  🚨 Alerts  │  📈 ML    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
AI-Powered Product Quality Intelligence System/
├── data/
│   ├── raw/                    ← Generated synthetic CSVs
│   └── processed/              ← Cleaned Parquet files
├── src/
│   ├── data/
│   │   ├── generate_data.py    ← Synthetic dataset generator (4 CSVs)
│   │   └── preprocess.py       ← Cleaning + merging pipeline
│   ├── features/
│   │   ├── sentiment.py        ← VADER sentiment scoring
│   │   └── engineering.py      ← Lag features, correlations, ML labels
│   ├── models/
│   │   ├── defect_predictor.py ← XGBoost binary classifier
│   │   ├── return_forecaster.py← XGBoost regressor
│   │   └── anomaly_detector.py ← Isolation Forest + alert table
│   ├── rag/
│   │   ├── vectorstore.py      ← ChromaDB ingestion (embeddings)
│   │   └── agent.py            ← LangChain RAG conversational agent
│   └── api/
│       └── server.py           ← FastAPI prediction endpoints
├── app/
│   ├── streamlit_app.py        ← Main dashboard entry point
│   ├── pages/
│   │   ├── eda_insights.py     ← EDA charts & metrics
│   │   ├── chatbot.py          ← AI chatbot interface
│   │   ├── alerts.py           ← Proactive anomaly alerts
│   │   └── predictions.py      ← Interactive ML prediction forms
│   └── components/
│       └── charts.py           ← Reusable Plotly charts
├── models/                     ← Saved model artifacts (.joblib)
├── chroma_db/                  ← ChromaDB persistent vector store
├── mlruns/                     ← MLflow experiment tracking
├── run_pipeline.py             ← One-click full pipeline runner
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "AI-Powered Product Quality Intelligence System"
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the template
copy .env.example .env

# Edit .env and add your Gemini API key:
# GOOGLE_API_KEY=your_key_here
# Get free key at: https://aistudio.google.com/
```

### 3. Run the Full Pipeline

```bash
python run_pipeline.py
```

This runs all 7 steps in sequence (~5–15 minutes depending on hardware):
1. Generate synthetic datasets
2. Preprocess & merge data
3. VADER sentiment scoring
4. Feature engineering
5. Train defect predictor (XGBoost)
6. Train return forecaster (XGBoost)
7. Run anomaly detection (Isolation Forest)
8. Build ChromaDB vector store

### 4. Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

### 5. (Optional) Start FastAPI Server

In a separate terminal:
```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```
> The dashboard works without the API server — it falls back to direct model inference automatically.

---

## 📊 Datasets (Synthetic)

All data is synthetically generated but realistic:

| Dataset | Rows | Key Columns |
|---|---|---|
| `products.csv` | 300 | product_id, category, unit_price |
| `sales.csv` | 12,000 | product_id, date, units_sold, revenue, region |
| `returns.csv` | 3,500 | product_id, date, reason, defect_type, return_qty |
| `reviews.csv` | 9,000 | product_id, date, rating, review_text |
| `manufacturing.csv` | 2,500 | product_id, batch_id, supplier, defect_rate |

Shared key: `product_id` — enables cross-dataset correlation analysis.

---

## 🤖 ML Models

### Defect Risk Classifier (XGBoost)
- **Task**: Binary classification — will this product have a high defect rate (>5%)?
- **Features**: supplier, production line, lead time, sentiment, rolling return rate
- **Tracking**: MLflow — accuracy, F1, AUC-ROC, feature importances

### Return Volume Forecaster (XGBoost)
- **Task**: Regression — predict total return quantity per product
- **Features**: units sold, avg rating, sentiment, defect rate, rolling returns
- **Tracking**: MLflow — MAE, RMSE, R²

### Anomaly Detector (Isolation Forest)
- **Task**: Unsupervised anomaly detection on quality signal combinations
- **Features**: defect rate, return rate, sentiment, % negative reviews
- **Output**: Severity labels (Critical / High / Medium / Normal)

---

## 🔗 LangChain RAG Pipeline

The chatbot is powered by a **Retrieval-Augmented Generation** pipeline:

```
User Query
    ↓
[Query Rewriting]  ← ConversationBufferWindowMemory (last 10 turns)
    ↓
[ChromaDB Retrieval]  ← MMR search, k=6 documents
    ↓
[Context Assembly]  ← Product summaries + Reviews + Manufacturing docs
    ↓
[Gemini Flash Generation]  ← Custom quality analyst system prompt
    ↓
[Answer + Sources]
```

**Document types embedded**:
- Product quality summaries (all 300 products)
- Top customer reviews (positive + negative samples per product)
- Manufacturing batch records (recent 2 batches per product)

---

## 🚨 Proactive Alert System

The **Isolation Forest** model monitors 6 quality signals simultaneously:

| Signal | Description |
|---|---|
| `avg_defect_rate` | Average defect rate from manufacturing batches |
| `return_rate` | Total returns / units sold |
| `avg_sentiment` | VADER compound sentiment score |
| `pct_negative` | % of reviews classified as Negative |
| `rolling_3m_return` | 3-month rolling return rate trend |
| `defect_return_rate` | % of returns attributed to defects |

**Severity Levels**: 🔴 Critical → 🟠 High → 🟡 Medium → 🟢 Normal

---

## 💬 Sample Chatbot Questions

```
"Which products have the highest defect rates?"
"What are the most common reasons for returns?"
"Which suppliers are associated with the most quality issues?"
"What is the correlation between sentiment and return rates?"
"Give me a summary of the top 5 highest-risk products."
"What actions would you recommend to reduce defects in Electronics?"
"Which production lines have the worst quality track record?"
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | Pandas, NumPy, PyArrow |
| NLP / Sentiment | VADER Sentiment, NLTK |
| ML Models | XGBoost, Scikit-learn |
| Anomaly Detection | Isolation Forest (sklearn) |
| Experiment Tracking | MLflow |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM | Google Gemini 1.5 Flash |
| RAG Framework | LangChain |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Visualization | Plotly |

---

## 🎯 Interview Talking Points

### "Walk me through your architecture"
> "I built a 4-layer system. The **data layer** ingests and merges 4 data sources on a shared product key. The **feature engineering layer** adds VADER sentiment scores, lag features, and correlation analysis. The **ML layer** trains 3 models: an XGBoost classifier for defect prediction, an XGBoost regressor for return forecasting, and an Isolation Forest for anomaly detection. The **intelligence layer** uses a LangChain RAG pipeline backed by ChromaDB and Gemini Flash to answer natural language quality queries."

### "How does your RAG chatbot work?"
> "Product summaries, customer reviews, and manufacturing records are embedded using sentence-transformers and stored in ChromaDB. When a user asks a question, LangChain retrieves the top 6 most relevant documents using Max-Marginal Relevance search, then passes them as context to Gemini Flash with a custom quality analyst system prompt. The conversation memory retains the last 10 turns for contextual follow-up."

### "How do you detect quality anomalies proactively?"
> "An Isolation Forest model monitors 6 quality signals simultaneously — defect rate, return rate, sentiment score, negative review percentage, rolling return trends, and defect return rate. Products that deviate significantly from the normal distribution get flagged and severity-scored. The system generates actionable recommendations automatically based on the severity."

### "How did you handle the lack of real data?"
> "I designed a realistic synthetic data generator using domain knowledge about supply chain quality systems. The 4 datasets share a product key and have embedded correlations — for example, suppliers are assigned different baseline defect rates, and review text templates match product quality levels. This makes the ML models and RAG results meaningful even without real data."

### "What MLOps practices did you implement?"
> "I used MLflow to log all model parameters, metrics (F1, AUC, RMSE, R²), and feature importances for every training run. Models are serialized with joblib and served via a FastAPI REST API with Pydantic schema validation. The Streamlit dashboard falls back to direct model inference if the API server is not running, ensuring resilience."

---

## 📝 License

MIT — For educational and portfolio demonstration purposes.
