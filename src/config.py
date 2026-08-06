from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DRIFT_LOG_DIR = ARTIFACTS_DIR / "drift_logs"

TRAIN_CSV = DATA_DIR / "train.csv"

FEATURES = [
    "store_code", "item_code", "dow", "is_weekend", "month",
    "weekofyear", "year", "dayofyear",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_lag_364",
    "sales_rmean_7", "sales_rstd_7", "sales_rmean_28", "sales_rstd_28",
    "sales_rmean_90", "sales_rstd_90",
]
TARGET = "sales"

XGBOOST_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.0356,
    "n_estimators": 750,
    "min_child_weight": 9,
    "subsample": 0.799,
    "colsample_bytree": 0.776,
    "gamma": 3.564,
    "reg_alpha": 3.21e-07,
    "reg_lambda": 3.12e-07,
    "random_state": 42,
}

PSI_THRESHOLD = 0.25
KS_P_THRESHOLD = 0.05
DRIFT_FEATURES = [
    "sales_lag_364", "sales_rmean_7", "sales_rmean_28",
    "month", "sales_rmean_90",
]

PROMOTION_TOLERANCE = 1.05
RETRAIN_COOLDOWN_HOURS = 24
HOLDOUT_DAYS = 28

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
CATEGORY_MAP_PATH = ARTIFACTS_DIR / "category_mappings.json"
SERVING_TABLE_PATH = ARTIFACTS_DIR / "serving_table.parquet"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
