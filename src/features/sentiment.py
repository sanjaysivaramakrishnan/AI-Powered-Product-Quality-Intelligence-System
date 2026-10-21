"""
Sentiment Scoring (VADER)
=========================
Adds VADER compound sentiment scores to the reviews dataset and
saves an enriched Parquet file.

Run:  python src/features/sentiment.py
"""

import os
import logging
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def ensure_vader_lexicon():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        log.info("Downloading VADER lexicon…")
        nltk.download("vader_lexicon", quiet=True)


def score_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment columns to a reviews DataFrame."""
    ensure_vader_lexicon()
    analyzer = SentimentIntensityAnalyzer()

    log.info("Scoring %d reviews with VADER…", len(reviews))

    scores = reviews["review_text"].apply(
        lambda text: analyzer.polarity_scores(str(text))
    )
    score_df = pd.DataFrame(list(scores))
    reviews = pd.concat([reviews.reset_index(drop=True), score_df], axis=1)

    reviews.rename(columns={
        "compound": "sentiment_compound",
        "pos":      "sentiment_pos",
        "neu":      "sentiment_neu",
        "neg":      "sentiment_neg",
    }, inplace=True)

    reviews["sentiment_label"] = reviews["sentiment_compound"].apply(
        lambda s: "Positive" if s >= 0.05 else ("Negative" if s <= -0.05 else "Neutral")
    )

    log.info("Sentiment distribution:\n%s", reviews["sentiment_label"].value_counts().to_string())
    return reviews


def aggregate_sentiment_per_product(reviews: pd.DataFrame) -> pd.DataFrame:
    """Return per-product average sentiment stats."""
    agg = reviews.groupby("product_id").agg(
        avg_sentiment=("sentiment_compound", "mean"),
        pct_positive=("sentiment_label", lambda x: (x == "Positive").mean()),
        pct_negative=("sentiment_label", lambda x: (x == "Negative").mean()),
        n_sentiment_reviews=("sentiment_compound", "count"),
    ).reset_index()
    agg["avg_sentiment"]  = agg["avg_sentiment"].round(4)
    agg["pct_positive"]   = agg["pct_positive"].round(4)
    agg["pct_negative"]   = agg["pct_negative"].round(4)
    return agg


def main():
    reviews_path = os.path.join(PROC_DIR, "reviews_clean.parquet")
    if not os.path.exists(reviews_path):
        raise FileNotFoundError(
            "Run src/data/preprocess.py first to generate reviews_clean.parquet"
        )

    reviews = pd.read_parquet(reviews_path)
    reviews_scored = score_reviews(reviews)
    reviews_scored.to_parquet(
        os.path.join(PROC_DIR, "reviews_sentiment.parquet"), index=False
    )
    log.info("✅ reviews_sentiment.parquet saved")

    sentiment_agg = aggregate_sentiment_per_product(reviews_scored)
    sentiment_agg.to_parquet(
        os.path.join(PROC_DIR, "sentiment_per_product.parquet"), index=False
    )
    log.info("✅ sentiment_per_product.parquet saved (%d products)", len(sentiment_agg))
    print(sentiment_agg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
