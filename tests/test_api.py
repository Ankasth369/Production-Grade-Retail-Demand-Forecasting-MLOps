from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.config import API_KEY

AUTH_HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_registry():
    with patch("app.main.registry") as mock_reg:
        mock_reg.is_loaded.return_value = True
        mock_reg.metrics = {"smape": 12.40, "mae": 5.85, "rmse": 7.57}
        mock_reg.category_mappings = {
            "store": {1: 0, 2: 1},
            "item": {1: 0, 2: 1},
        }

        dates = pd.date_range("2017-01-01", periods=364, freq="D")
        serving_rows = []
        for store in [1, 2]:
            for item in [1, 2]:
                for date in dates:
                    serving_rows.append(
                        {
                            "date": date,
                            "store": store,
                            "item": item,
                            "sales": np.random.randint(20, 80),
                        }
                    )
        mock_reg.serving_table = pd.DataFrame(serving_rows)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([42.5])
        mock_reg.model = mock_model

        yield mock_reg


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_success(client):
    resp = client.post(
        "/predict",
        json={"store_id": 1, "item_id": 1, "date": "2017-12-31"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "forecast" in data
    assert isinstance(data["forecast"], (int, float))
    assert data["forecast"] >= 0


def test_predict_unknown_store(client):
    resp = client.post(
        "/predict",
        json={"store_id": 999, "item_id": 1, "date": "2017-12-31"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 400


def test_predict_unknown_item(client):
    resp = client.post(
        "/predict",
        json={"store_id": 1, "item_id": 999, "date": "2017-12-31"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 400


def test_predict_requires_api_key(client):
    resp = client.post(
        "/predict",
        json={"store_id": 1, "item_id": 1, "date": "2017-12-31"},
    )
    assert resp.status_code == 401


def test_reload(client):
    resp = client.post("/reload", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["status"] == "reloaded"


def test_metrics_endpoint(client):
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"api_requests_total" in resp.content
