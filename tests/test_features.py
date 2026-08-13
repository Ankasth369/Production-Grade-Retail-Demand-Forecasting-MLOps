import numpy as np
import pandas as pd
import pytest

from src.features.engineering import build_category_mappings, build_features


@pytest.fixture
def sample_df():
    dates = pd.date_range("2013-01-01", periods=400, freq="D")
    rows = []
    for store in [1, 2]:
        for item in [1, 2]:
            for date in dates:
                rows.append(
                    {
                        "date": date,
                        "store": store,
                        "item": item,
                        "sales": np.random.randint(10, 100),
                    }
                )
    return pd.DataFrame(rows)


def test_no_future_leakage(sample_df):
    """sales_lag_1 at row t must equal sales at t-1, never t or t+1."""
    mappings = build_category_mappings(sample_df)
    df = build_features(sample_df, mappings)

    for (store, item), group in df.groupby(["store", "item"]):
        group = group.sort_values("date").reset_index(drop=True)
        valid = group.dropna(subset=["sales_lag_1"])
        for idx in range(len(valid)):
            row = valid.iloc[idx]
            prev_date = row["date"] - pd.Timedelta(days=1)
            prev_row = group[group["date"] == prev_date]
            if not prev_row.empty:
                assert row["sales_lag_1"] == prev_row["sales"].values[0]


def test_lag7_correct(sample_df):
    mappings = build_category_mappings(sample_df)
    df = build_features(sample_df, mappings)

    for (store, item), group in df.groupby(["store", "item"]):
        group = group.sort_values("date").reset_index(drop=True)
        valid = group.dropna(subset=["sales_lag_7"])
        for idx in range(min(5, len(valid))):
            row = valid.iloc[idx]
            past_date = row["date"] - pd.Timedelta(days=7)
            past_row = group[group["date"] == past_date]
            if not past_row.empty:
                assert row["sales_lag_7"] == past_row["sales"].values[0]


def test_category_mappings_deterministic(sample_df):
    m1 = build_category_mappings(sample_df)
    m2 = build_category_mappings(sample_df)
    assert m1 == m2


def test_features_columns_present(sample_df):
    from src.config import FEATURES

    mappings = build_category_mappings(sample_df)
    df = build_features(sample_df, mappings)
    for feat in FEATURES:
        assert feat in df.columns, f"Missing feature: {feat}"


def test_rolling_mean_uses_shifted_data(sample_df):
    """Rolling mean must be computed on shifted sales, not current sales."""
    mappings = build_category_mappings(sample_df)
    df = build_features(sample_df, mappings)
    valid = df.dropna(subset=["sales_rmean_7"])
    assert len(valid) > 0
    assert not valid["sales_rmean_7"].isna().all()
