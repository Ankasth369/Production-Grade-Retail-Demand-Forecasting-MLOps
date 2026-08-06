from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import FEATURES
from src.features.engineering import build_features
from src.models.registry import ModelRegistry

registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load()
    yield


app = FastAPI(title="Demand Forecast API", lifespan=lifespan)


class PredictRequest(BaseModel):
    store_id: int
    item_id: int
    date: str


class PredictResponse(BaseModel):
    store_id: int
    item_id: int
    date: str
    forecast: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": registry.is_loaded(),
        "metrics": registry.metrics,
    }


@app.post("/reload")
def reload():
    registry.load()
    return {"status": "reloaded", "metrics": registry.metrics}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not registry.is_loaded():
        raise HTTPException(503, "Model not loaded")

    mappings = registry.category_mappings
    if req.store_id not in mappings["store"]:
        raise HTTPException(400, f"Unknown store_id: {req.store_id}")
    if req.item_id not in mappings["item"]:
        raise HTTPException(400, f"Unknown item_id: {req.item_id}")

    target_date = pd.Timestamp(req.date)
    serving = registry.serving_table
    series = serving[
        (serving["store"] == req.store_id) & (serving["item"] == req.item_id)
    ].copy()

    if series.empty:
        raise HTTPException(
            400, f"No history for store={req.store_id}, item={req.item_id}"
        )

    new_row = pd.DataFrame(
        {"date": [target_date], "store": [req.store_id],
         "item": [req.item_id], "sales": [np.nan]}
    )
    series = pd.concat([series, new_row], ignore_index=True)
    series = build_features(series, mappings)

    row = series[series["date"] == target_date]
    if row.empty or row[FEATURES].isna().any(axis=1).all():
        raise HTTPException(
            400, "Insufficient history to compute features for this date"
        )

    forecast = float(registry.model.predict(row[FEATURES].values)[0])
    forecast = max(0, round(forecast, 2))

    return PredictResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        date=req.date,
        forecast=forecast,
    )
