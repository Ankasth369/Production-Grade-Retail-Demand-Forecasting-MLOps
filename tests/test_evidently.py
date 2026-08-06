import numpy as np
import pandas as pd
import pytest

from src.monitoring.evidently_monitor import generate_drift_report, EVIDENTLY_REPORTS_DIR


@pytest.fixture
def reference_df():
    rng = np.random.RandomState(42)
    n = 5000
    return pd.DataFrame({
        "sales_lag_364": rng.normal(50, 10, n),
        "sales_rmean_7": rng.normal(50, 5, n),
        "sales_rmean_28": rng.normal(50, 5, n),
        "month": rng.randint(1, 13, n).astype(float),
        "sales_rmean_90": rng.normal(50, 5, n),
    })


@pytest.fixture
def drifted_df():
    rng = np.random.RandomState(99)
    n = 5000
    return pd.DataFrame({
        "sales_lag_364": rng.normal(80, 15, n),
        "sales_rmean_7": rng.normal(80, 8, n),
        "sales_rmean_28": rng.normal(80, 8, n),
        "month": rng.randint(1, 13, n).astype(float),
        "sales_rmean_90": rng.normal(80, 8, n),
    })


def test_evidently_drift_report_no_drift(reference_df, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.monitoring.evidently_monitor.EVIDENTLY_REPORTS_DIR", tmp_path
    )
    result = generate_drift_report(reference_df, reference_df)
    assert result["n_drifted"] == 0
    assert "report_path" in result


def test_evidently_drift_report_with_drift(reference_df, drifted_df, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.monitoring.evidently_monitor.EVIDENTLY_REPORTS_DIR", tmp_path
    )
    result = generate_drift_report(reference_df, drifted_df)
    assert result["dataset_drift"] is True
    assert result["n_drifted"] > 0

    report_file = tmp_path / result["report_path"].replace("\\", "/").split("/")[-1]
    assert report_file.exists()


def test_evidently_report_saves_summary(reference_df, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.monitoring.evidently_monitor.EVIDENTLY_REPORTS_DIR", tmp_path
    )
    generate_drift_report(reference_df, reference_df)
    summaries = list(tmp_path.glob("drift_summary_*.json"))
    assert len(summaries) == 1
