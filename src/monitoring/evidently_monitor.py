"""Evidently AI integration for production-grade drift monitoring.

Generates interactive HTML drift reports alongside the lightweight PSI/KS checks.
Compatible with Evidently >= 0.7.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import ARTIFACTS_DIR, DRIFT_FEATURES, FEATURES, HOLDOUT_DAYS

EVIDENTLY_REPORTS_DIR = ARTIFACTS_DIR / "evidently_reports"


def generate_drift_report(reference_df, current_df, features=None):
    features = features or DRIFT_FEATURES
    ref = reference_df[features].dropna()
    cur = current_df[features].dropna()

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref, current_data=cur)

    EVIDENTLY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = EVIDENTLY_REPORTS_DIR / f"drift_report_{ts}.html"
    snapshot.save_html(str(report_path))

    d = snapshot.dump_dict()
    metric_results = d.get("metric_results", {})

    n_drifted = 0
    drift_share = 0.0
    column_details = {}

    for mid, mr in metric_results.items():
        if not isinstance(mr, dict):
            continue
        dn = mr.get("display_name", "")

        if "Count of Drifted" in dn:
            count_val = mr.get("count", {})
            share_val = mr.get("share", {})
            if isinstance(count_val, dict):
                n_drifted = int(count_val.get("value", 0))
                drift_share = float(share_val.get("value", 0))
            else:
                n_drifted = int(count_val) if count_val else 0
                drift_share = float(share_val) if share_val else 0

        if dn.startswith("Value drift for "):
            col_name = dn.replace("Value drift for ", "")
            drift_score = mr.get("value", 0)
            if isinstance(drift_score, dict):
                drift_score = drift_score.get("value", 0)
            column_details[col_name] = {
                "drift_score": float(drift_score) if drift_score else 0,
                "drift_detected": float(drift_score) < 0.05 if drift_score else False,
            }

    dataset_drift = n_drifted > 0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path),
        "dataset_drift": dataset_drift,
        "share_drifted": drift_share,
        "n_drifted": n_drifted,
        "n_columns": len(features),
        "column_details": column_details,
    }

    summary_path = EVIDENTLY_REPORTS_DIR / f"drift_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_full_monitoring(reference_df, current_df):
    """Run both custom PSI/KS and Evidently drift detection."""
    from src.monitoring.monitor import monitor_once

    custom_report = monitor_once(reference_df, current_df)
    evidently_report = generate_drift_report(reference_df, current_df)

    return {
        "custom": custom_report,
        "evidently": evidently_report,
    }


if __name__ == "__main__":
    from src.data.loader import load_train_data
    from src.features.engineering import build_features, load_category_mappings

    print("Loading data...")
    df = load_train_data()
    mappings = load_category_mappings()
    df = build_features(df, mappings).dropna(subset=FEATURES)

    cutoff = df["date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    reference = df[df["date"] <= cutoff]
    current = df[df["date"] > cutoff]

    print("Generating Evidently drift report...")
    summary = generate_drift_report(reference, current)
    print(f"  Dataset drift: {summary['dataset_drift']}")
    print(f"  Drifted columns: {summary['n_drifted']}/{summary['n_columns']}")
    print(f"  Report: {summary['report_path']}")
    print("\nDone.")
