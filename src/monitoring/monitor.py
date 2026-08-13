import json
import logging
from datetime import datetime, timezone

from src.config import DRIFT_FEATURES, DRIFT_LOG_DIR, KS_P_THRESHOLD, PSI_THRESHOLD
from src.monitoring.drift import ks_test, psi
from src.monitoring.performance import evaluate_performance

logger = logging.getLogger(__name__)


def monitor_once(reference_df, current_df, y_true=None, y_pred=None):
    """Lightweight PSI/KS drift check. Fast, no heavy dependencies."""
    drift_results = {}
    any_drift = False

    for feature in DRIFT_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ref_vals = reference_df[feature].dropna().values
        cur_vals = current_df[feature].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        psi_val = psi(ref_vals, cur_vals)
        ks_stat, ks_p = ks_test(ref_vals, cur_vals)

        drifted = psi_val > PSI_THRESHOLD or ks_p < KS_P_THRESHOLD
        if drifted:
            any_drift = True

        drift_results[feature] = {
            "psi": round(psi_val, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(ks_p, 6),
            "drifted": drifted,
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_detected": any_drift,
        "feature_drift": drift_results,
    }

    if y_true is not None and y_pred is not None:
        report["performance"] = evaluate_performance(y_true, y_pred)

    DRIFT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = DRIFT_LOG_DIR / f"drift_{ts}.json"
    with open(log_file, "w") as f:
        json.dump(report, f, indent=2)

    return report


def monitor_full(reference_df, current_df, y_true=None, y_pred=None):
    """Full monitoring: custom PSI/KS + Evidently AI reports."""
    custom = monitor_once(reference_df, current_df, y_true, y_pred)

    evidently_result = None
    try:
        from src.monitoring.evidently_monitor import generate_drift_report

        evidently_result = generate_drift_report(reference_df, current_df)
    except ImportError:
        logger.warning("Evidently not installed, skipping rich reports")
    except Exception as e:  # noqa: BLE001 -- Evidently is best-effort; custom PSI/KS check must still return
        logger.warning("Evidently report generation failed: %s", e)

    return {
        "custom": custom,
        "evidently": evidently_result,
    }
