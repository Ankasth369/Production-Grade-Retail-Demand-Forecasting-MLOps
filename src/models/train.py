import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

from src.config import (
    FEATURES, TARGET, XGBOOST_PARAMS, HOLDOUT_DAYS, ARTIFACTS_DIR,
)
from src.data.loader import load_train_data
from src.features.engineering import (
    build_features, build_category_mappings, save_category_mappings,
)


def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    return float(100 * np.mean(2 * diff / np.where(denominator == 0, 1, denominator)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_model(df=None, output_dir=None):
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

    joblib.dump(model, output_dir / "model.joblib")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    latest = df.groupby(["store", "item"]).tail(364)
    latest.to_parquet(output_dir / "serving_table.parquet", index=False)

    return model, metrics


if __name__ == "__main__":
    model, metrics = train_model()
    print(f"Training complete. Metrics: {metrics}")
