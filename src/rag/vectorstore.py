"""
Vector Store — ChromaDB Ingestion
===================================
Embeds product summaries and top reviews using sentence-transformers
and persists them to a local ChromaDB collection.

Run:  python src/rag/vectorstore.py
"""

import os
import logging
import pandas as pd
from tqdm import tqdm
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
CHROMA_DIR = os.environ.get(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
)
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION  = "product_quality"


def get_embeddings():
    log.info("Loading embedding model: %s", EMBED_MODEL)
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_product_documents(master: pd.DataFrame) -> list[Document]:
    """Convert each product row into a rich text document for embedding."""
    docs = []
    for _, row in master.iterrows():
        text = (
            f"Product: {row.get('product_name', row['product_id'])} "
            f"(ID: {row['product_id']}).\n"
            f"Category: {row.get('category', 'Unknown')}.\n"
            f"Supplier: {row.get('supplier', 'Unknown')} | "
            f"Production Line: {row.get('production_line', 'Unknown')}.\n"
            f"Average Defect Rate: {float(row.get('avg_defect_rate', 0))*100:.2f}%.\n"
            f"Return Rate: {float(row.get('return_rate', 0))*100:.2f}% | "
            f"Total Returns: {int(row.get('total_return_qty', 0))}.\n"
            f"Average Customer Rating: {float(row.get('avg_rating', 0)):.1f}/5 "
            f"from {int(row.get('n_reviews', 0))} reviews.\n"
            f"Average Sentiment Score: {float(row.get('avg_sentiment', 0)):.3f} "
            f"({float(row.get('pct_positive', 0))*100:.0f}% positive).\n"
            f"Quality Score: {float(row.get('quality_score', 0)):.1f}/100.\n"
            f"Total Units Sold: {int(row.get('total_units_sold', 0))} | "
            f"Revenue: ${float(row.get('total_revenue', 0)):,.0f}."
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "product_id":    str(row["product_id"]),
                "category":      str(row.get("category", "")),
                "supplier":      str(row.get("supplier", "")),
                "defect_rate":   float(row.get("avg_defect_rate", 0)),
                "return_rate":   float(row.get("return_rate", 0)),
                "quality_score": float(row.get("quality_score", 0)),
                "doc_type":      "product_summary",
            }
        ))
    return docs


def build_review_documents(reviews: pd.DataFrame, max_per_product: int = 3) -> list[Document]:
    """Sample top reviews per product and embed them."""
    docs = []
    for pid, grp in reviews.groupby("product_id"):
        # Take highest & lowest rated reviews for diversity
        sample = pd.concat([
            grp.nlargest(max_per_product, "rating"),
            grp.nsmallest(max_per_product, "rating"),
        ]).drop_duplicates()

        for _, row in sample.iterrows():
            text = (
                f"Customer review for product {pid}:\n"
                f"Rating: {int(row.get('rating', 0))}/5 stars.\n"
                f"Sentiment: {row.get('sentiment_label', 'Unknown')} "
                f"(compound score: {float(row.get('sentiment_compound', 0)):.3f}).\n"
                f"Review: \"{row.get('review_text', '')}\""
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "product_id": str(pid),
                    "rating":     int(row.get("rating", 0)),
                    "sentiment":  str(row.get("sentiment_label", "")),
                    "doc_type":   "customer_review",
                }
            ))
    return docs


def build_manufacturing_documents(mfg: pd.DataFrame) -> list[Document]:
    """Embed manufacturing batch summaries."""
    docs = []
    # Summarize by product (top 2 recent batches)
    mfg_sorted = mfg.sort_values("production_date", ascending=False)
    for pid, grp in mfg_sorted.groupby("product_id"):
        recent = grp.head(2)
        for _, row in recent.iterrows():
            text = (
                f"Manufacturing batch {row.get('batch_id', '')} for product {pid}.\n"
                f"Supplier: {row.get('supplier', 'Unknown')} | "
                f"Line: {row.get('production_line', 'Unknown')}.\n"
                f"Batch Size: {int(row.get('batch_size', 0))} units | "
                f"Defect Rate: {float(row.get('defect_rate', 0))*100:.2f}%.\n"
                f"Quality Check Passed: {'Yes' if row.get('quality_check_passed', 0) else 'No'}.\n"
                f"Lead Time: {int(row.get('lead_time_days', 0))} days."
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "product_id":  str(pid),
                    "supplier":    str(row.get("supplier", "")),
                    "defect_rate": float(row.get("defect_rate", 0)),
                    "doc_type":    "manufacturing_batch",
                }
            ))
    return docs


def ingest(force_rebuild: bool = False) -> Chroma:
    """Build or load the ChromaDB vector store."""
    embeddings = get_embeddings()

    if os.path.exists(CHROMA_DIR) and not force_rebuild:
        log.info("Loading existing ChromaDB from %s", CHROMA_DIR)
        return Chroma(
            collection_name=COLLECTION,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )

    log.info("Building ChromaDB vector store from scratch…")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    master  = pd.read_parquet(os.path.join(PROC_DIR, "features_df.parquet"))
    reviews = pd.read_parquet(os.path.join(PROC_DIR, "reviews_sentiment.parquet"))
    mfg     = pd.read_parquet(os.path.join(PROC_DIR, "manufacturing_clean.parquet"))

    product_docs = build_product_documents(master)
    review_docs  = build_review_documents(reviews)
    mfg_docs     = build_manufacturing_documents(mfg)

    all_docs = product_docs + review_docs + mfg_docs
    log.info("Total documents to embed: %d", len(all_docs))

    # Batch ingest to avoid OOM
    BATCH = 100
    vectorstore = None
    for i in tqdm(range(0, len(all_docs), BATCH), desc="Embedding batches"):
        batch = all_docs[i: i + BATCH]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION,
                persist_directory=CHROMA_DIR,
            )
        else:
            vectorstore.add_documents(batch)

    log.info("✅ ChromaDB built and persisted → %s (%d documents)", CHROMA_DIR, len(all_docs))
    return vectorstore


def get_retriever(k: int = 6):
    """Return a LangChain retriever from the persisted vector store."""
    vectorstore = ingest()
    return vectorstore.as_retriever(
        search_type="mmr",          # Max-Marginal Relevance for diversity
        search_kwargs={"k": k, "fetch_k": 20},
    )


if __name__ == "__main__":
    vs = ingest(force_rebuild=True)
    # Quick test
    results = vs.similarity_search("Which products have the highest defect rates?", k=3)
    print("\n🔍 Test retrieval:")
    for r in results:
        print(f"  [{r.metadata['doc_type']}] {r.page_content[:150]}…\n")
