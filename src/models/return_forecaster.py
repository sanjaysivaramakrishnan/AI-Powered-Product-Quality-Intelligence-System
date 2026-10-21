"""
Return Volume Forecaster
=========================
XGBoost regressor predicting total return quantity per product
over the next 30 days.

Run:  python src/models/return_forecaster.py
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

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
    "total_units_sold", "avg_rating", "avg_sentiment",
    "pct_negative", "avg_defect_rate", "defect_risk_label",
    "rolling_3m_return", "rolling_6m_return", "return_momentum",
    "avg_discount", "avg_lead_time",
]
TARGET_COL = "return_qty_target"


def load_data():
    path = os.path.join(PROC_DIR, "features_df.parquet")
    df = pd.read_parquet(path)
    df = df.dropna(subset=["total_return_qty"])
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    X = df[[c for c in FEATURE_COLS if c in df.columns]].astype(float)
    y = df[TARGET_COL].astype(float)
    log.info("Data loaded: %d samples | target range [%.1f, %.1f]",
             len(y), y.min(), y.max())
    return X, y, df["product_id"]


def train_model(X, y, product_ids):
    os.makedirs(MODELS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, product_ids, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
    }

    model = xgb.XGBRegressor(**params)

    with mlflow.start_run(run_name="return_forecaster"):
        mlflow.log_params(params)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv,
                                   scoring="neg_root_mean_squared_error")
        mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
        log.info("CV RMSE: %.3f ± %.3f", cv_rmse.mean(), cv_rmse.std())

        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        mlflow.log_metrics({"test_mae": mae, "test_rmse": rmse, "test_r2": r2})
        log.info("Test MAE: %.2f | RMSE: %.2f | R²: %.3f", mae, rmse, r2)

        fi = pd.DataFrame({
            "feature": X.columns.tolist(),
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        log.info("Top return drivers:\n%s", fi.head(8).to_string(index=False))

        # Save model artifact and log path (avoids model registry requirement)
        _tmp_model_path = os.path.join(MODELS_DIR, "return_forecaster.joblib")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump({"model": model, "features": X.columns.tolist()}, _tmp_model_path)
        mlflow.log_artifact(_tmp_model_path, artifact_path="model")

    model_path = os.path.join(MODELS_DIR, "return_forecaster.joblib")
    joblib.dump({
        "model":    model,
        "features": X.columns.tolist(),
    }, model_path)
    log.info("✅ Model saved → %s", model_path)
    return model, fi


if __name__ == "__main__":
    X, y, pids = load_data()
    train_model(X, y, pids)
