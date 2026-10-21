"""
Defect Risk Predictor
======================
XGBoost binary classifier predicting whether a product batch
will have a HIGH defect rate (> 5%).

Features used:
  supplier_enc, production_line_enc, category_enc,
  avg_lead_time, n_batches, avg_discount,
  avg_sentiment, pct_negative, rolling_3m_return

Run:  python src/models/defect_predictor.py
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
# import mlflow
# mlflow.set_tracking_uri("file:./mlruns")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_default_mlruns = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mlruns")
)
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file:///{_default_mlruns}")

FEATURE_COLS = [
    "supplier_enc", "production_line_enc", "category_enc",
    "avg_lead_time", "n_batches", "avg_discount",
    "avg_sentiment", "pct_negative", "rolling_3m_return",
    "return_rate", "n_reviews",
]
TARGET_COL = "defect_risk_label"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(PROC_DIR, "features_df.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError("Run feature engineering first: python src/features/engineering.py")
    df = pd.read_parquet(path)

    # Keep only products that have manufacturing data
    df = df.dropna(subset=["avg_defect_rate"])

    # Fill any remaining NaN in feature cols
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    X = df[[c for c in FEATURE_COLS if c in df.columns]].astype(float)
    y = df[TARGET_COL].astype(int)
    log.info("Data loaded: %d samples | class dist: %s",
             len(y), dict(y.value_counts()))
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series):
    os.makedirs(MODELS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators":    200,
        "max_depth":       5,
        "learning_rate":   0.08,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        "scale_pos_weight":max(1, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        "eval_metric":     "logloss",
        "random_state":    42,
        "n_jobs":          -1,
    }

    model = xgb.XGBClassifier(**params)

    with mlflow.start_run(run_name="defect_predictor"):
        mlflow.log_params(params)

        # Cross-validation on train set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std",  cv_scores.std())
        log.info("CV F1: %.3f ± %.3f", cv_scores.mean(), cv_scores.std())

        # Final fit
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({"test_accuracy": acc, "test_f1": f1, "test_auc": auc})
        log.info("Test Accuracy: %.3f | F1: %.3f | AUC-ROC: %.3f", acc, f1, auc)
        print("\n📊 Classification Report:\n",
              classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

        # Feature importances
        fi = pd.DataFrame({
            "feature": X.columns.tolist(),
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        log.info("Top features:\n%s", fi.to_string(index=False))

        # Save model artifact and log path (avoids model registry requirement)
        _tmp_model_path = os.path.join(MODELS_DIR, "defect_predictor.joblib")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump({"model": model, "features": X.columns.tolist()}, _tmp_model_path)
        mlflow.log_artifact(_tmp_model_path, artifact_path="model")

    # Save locally
    model_path = os.path.join(MODELS_DIR, "defect_predictor.joblib")
    joblib.dump({"model": model, "features": X.columns.tolist()}, model_path)
    log.info("✅ Model saved → %s", model_path)
    return model, fi


if __name__ == "__main__":
    X, y = load_data()
    train_model(X, y)
