import json
import pandas as pd
from pathlib import Path
from src.config import CATEGORY_MAP_PATH


def build_category_mappings(df):
    store_map = {int(v): i for i, v in enumerate(sorted(df["store"].unique()))}
    item_map = {int(v): i for i, v in enumerate(sorted(df["item"].unique()))}
    return {"store": store_map, "item": item_map}


def save_category_mappings(mappings, path=None):
    path = Path(path or CATEGORY_MAP_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: {str(kk): vv for kk, vv in v.items()}
        for k, v in mappings.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_category_mappings(path=None):
    path = Path(path or CATEGORY_MAP_PATH)
    with open(path) as f:
        raw = json.load(f)
    return {
        "store": {int(k): v for k, v in raw["store"].items()},
        "item": {int(k): v for k, v in raw["item"].items()},
    }


def build_features(df, category_mappings=None):
    d = df.sort_values(["store", "item", "date"]).copy()

    if category_mappings is None:
        category_mappings = build_category_mappings(d)

    d["store_code"] = d["store"].map(category_mappings["store"]).astype("int16")
    d["item_code"] = d["item"].map(category_mappings["item"]).astype("int16")

    d["dow"] = d["date"].dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype("int8")
    d["month"] = d["date"].dt.month
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype("int16")
    d["year"] = d["date"].dt.year
    d["dayofyear"] = d["date"].dt.dayofyear

    g = d.groupby(["store", "item"], group_keys=False)
    for lag in [1, 7, 14, 28, 364]:
        d[f"sales_lag_{lag}"] = g["sales"].shift(lag)

    sh = g["sales"].shift(1)
    for w in [7, 28, 90]:
        d[f"sales_rmean_{w}"] = (
            sh.groupby([d["store"], d["item"]])
            .transform(lambda s: s.rolling(w).mean())
        )
        d[f"sales_rstd_{w}"] = (
            sh.groupby([d["store"], d["item"]])
            .transform(lambda s: s.rolling(w).std())
        )

    return d
