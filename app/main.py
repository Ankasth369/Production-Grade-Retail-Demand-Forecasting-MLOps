import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import API_KEY, FEATURES, RATE_LIMIT
from src.features.engineering import build_features
from src.logging_config import configure_logging
from src.models.registry import ModelRegistry

configure_logging()
logger = logging.getLogger("demand_forecast.api")

registry = ModelRegistry()
limiter = Limiter(key_func=get_remote_address)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "path", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds", "Request latency in seconds", ["method", "path"]
)
PREDICTION_COUNT = Counter("predictions_total", "Total predictions served")


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load()
    yield


app = FastAPI(title="Demand Forecast API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def log_and_instrument_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(duration)

    logger.info(
        "request_handled",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "client_ip": request.client.host if request.client else None,
        },
    )
    return response


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


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
@limiter.limit(RATE_LIMIT)
def reload(request: Request, api_key: str = Depends(verify_api_key)):
    registry.load()
    return {"status": "reloaded", "metrics": registry.metrics}


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(RATE_LIMIT)
def predict(
    request: Request, req: PredictRequest, api_key: str = Depends(verify_api_key)
):
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
        {
            "date": [target_date],
            "store": [req.store_id],
            "item": [req.item_id],
            "sales": [np.nan],
        }
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

    PREDICTION_COUNT.inc()

    return PredictResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        date=req.date,
        forecast=forecast,
    )
