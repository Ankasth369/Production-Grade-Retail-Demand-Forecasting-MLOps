import json
from pathlib import Path

import numpy as np

from src.config import METRICS_PATH


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    return float(100 * np.mean(2 * diff / np.where(denominator == 0, 1, denominator)))


def evaluate_performance(y_true, y_pred, baseline_metrics_path=None):
    baseline_path = Path(baseline_metrics_path or METRICS_PATH)
    current_smape = smape(y_true, y_pred)

    result = {"current_smape": round(current_smape, 4)}

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_smape = baseline["smape"]
        result["baseline_smape"] = baseline_smape
        result["degradation_pct"] = round(
            (current_smape - baseline_smape) / baseline_smape * 100, 2
        )

    return result
