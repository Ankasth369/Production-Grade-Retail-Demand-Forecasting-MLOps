import pandas as pd

from src.config import TRAIN_CSV
from src.data.schema import validate_raw_data


def load_train_data(path=None, validate=True):
    path = path or TRAIN_CSV
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    if validate:
        df = validate_raw_data(df)
    return df
