import pandas as pd
from src.config import TRAIN_CSV


def load_train_data(path=None):
    path = path or TRAIN_CSV
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    return df
