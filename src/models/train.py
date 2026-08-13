import json
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import (
    ARTIFACTS_DIR,
    FEATURES,
    HOLDOUT_DAYS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    TARGET,
    XGBOOST_PARAMS,
)
from src.data.loader import load_train_data
from src.features.engineering import (
    build_category_mappings,
    build_features,
    save_category_mappings,
)


def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    return float(100 * np.mean(2 * diff / np.where(denominator == 0, 1, denominator)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_model(df=None, output_dir=None, register=True):
    output_dir = Path(output_dir or ARTIFACTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_train_data()

    mappings = build_category_mappings(df)
    save_category_mappings(mappings, output_dir / "category_mappings.json")

    df = build_features(df, mappings)
    df = df.dropna(subset=FEATURES)

    cutoff = df["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train = df[df["date"] <= cutoff]
    holdout = df[df["date"] > cutoff]

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(train[FEATURES], train[TARGET])

    preds = model.predict(holdout[FEATURES])
    metrics = {
        "smape": round(smape(holdout[TARGET].values, preds), 4),
        "mae": round(mae(holdout[TARGET].values, preds), 4),
        "rmse": round(rmse(holdout[TARGET].values, preds), 4),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "cutoff_date": str(cutoff.date()),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    try:
        with mlflow.start_run() as run:
            mlflow.log_params(XGBOOST_PARAMS)
            mlflow.log_metrics(
                {
                    "smape": metrics["smape"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "train_rows": metrics["train_rows"],
                    "holdout_rows": metrics["holdout_rows"],
                }
            )
            mlflow.xgboost.log_model(model, name="model")
            mlflow.log_artifact(str(output_dir / "category_mappings.json"))

            metrics["mlflow_run_id"] = run.info.run_id

            if register:
                model_uri = f"runs:/{run.info.run_id}/model"
                mv = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)
                metrics["mlflow_model_version"] = mv.version
    except Exception as e:  # noqa: BLE001 -- MLflow is best-effort; training must still complete
        metrics["mlflow_error"] = str(e)

    joblib.dump(model, output_dir / "model.joblib")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    latest = df.groupby(["store", "item"]).tail(364)
    latest.to_parquet(output_dir / "serving_table.parquet", index=False)

    return model, metrics


if __name__ == "__main__":
    model, metrics = train_model()
    print(f"Training complete. Metrics: {metrics}")
