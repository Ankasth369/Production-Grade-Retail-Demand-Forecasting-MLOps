import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.data.schema import validate_raw_data


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
        "store": [1, 2, 3],
        "item": [1, 2, 3],
        "sales": [10, 20, 0],
    })


def test_valid_data_passes(valid_df):
    result = validate_raw_data(valid_df)
    assert len(result) == len(valid_df)


def test_negative_sales_rejected(valid_df):
    bad = valid_df.copy()
    bad.loc[0, "sales"] = -5
    with pytest.raises(SchemaErrors):
        validate_raw_data(bad)


def test_non_positive_store_rejected(valid_df):
    bad = valid_df.copy()
    bad.loc[0, "store"] = 0
    with pytest.raises(SchemaErrors):
        validate_raw_data(bad)


def test_null_date_rejected(valid_df):
    bad = valid_df.copy()
    bad.loc[0, "date"] = pd.NaT
    with pytest.raises(SchemaErrors):
        validate_raw_data(bad)
