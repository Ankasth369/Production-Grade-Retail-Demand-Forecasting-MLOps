import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR


class ModelRegistry:
    def __init__(self, artifacts_dir=None):
        self._dir = Path(artifacts_dir or ARTIFACTS_DIR)
        self._model = None
        self._category_mappings = None
        self._serving_table = None
        self._metrics = None

    def load(self):
        self._model = joblib.load(self._dir / "model.joblib")
        with open(self._dir / "category_mappings.json") as f:
            raw = json.load(f)
        self._category_mappings = {
            "store": {int(k): v for k, v in raw["store"].items()},
            "item": {int(k): v for k, v in raw["item"].items()},
        }
        self._serving_table = pd.read_parquet(self._dir / "serving_table.parquet")
        with open(self._dir / "metrics.json") as f:
            self._metrics = json.load(f)
        return self

    @property
    def model(self):
        return self._model

    @property
    def category_mappings(self):
        return self._category_mappings

    @property
    def serving_table(self):
        return self._serving_table

    @property
    def metrics(self):
        return self._metrics

    def is_loaded(self):
        return self._model is not None
