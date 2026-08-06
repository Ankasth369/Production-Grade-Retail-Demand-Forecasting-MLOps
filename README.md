# Demand Forecast MLOps

[![CI](https://github.com/YOUR_USERNAME/demand-forecast-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/demand-forecast-mlops/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io)
[![DVC](https://img.shields.io/badge/DVC-3.67+-blueviolet.svg)](https://dvc.org)
[![Evidently](https://img.shields.io/badge/Evidently-0.7+-red.svg)](https://www.evidentlyai.com)

End-to-end demand forecasting system with drift detection, automated retraining, and a serving API — built as a production-grade MLOps pipeline.

---

## Highlights

- **XGBoost global model** trained on 500 store-item series (913K rows, 5 years daily)
- **12.40% SMAPE** after Optuna hyperparameter tuning (50 trials)
- **PSI + KS drift detection** on the top 5 SHAP features
- **Automated retraining** with promotion gate and cooldown safeguard
- **FastAPI** serving with hot model reload
- **Streamlit dashboard** with 5 interactive pages (plotly charts)
- **Docker Compose** for local orchestration, **Kubernetes** manifests for deployment
- **DVC** for data/artifact versioning and reproducible pipelines
- **Evidently AI** for interactive drift reports alongside custom PSI/KS
- **CI/CD** via GitHub Actions: lint, test, build, push, deploy

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                  │
│  train.csv (913K rows) ──► loader.py ──► build_features()          │
│                                          ├─ calendar features      │
│                                          ├─ lag features (1–364d)  │
│                                          ├─ rolling stats (7/28/90)│
│                                          └─ persisted cat mappings │
└────────────────────┬───────────────────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────────────────┐
│                      TRAINING PIPELINE                             │
│  XGBoost (tuned) ──► walk-forward eval ──► artifacts/              │
│                      (28-day holdout)       ├─ model.joblib        │
│                                             ├─ category_mappings   │
│                                             ├─ serving_table       │
│                                             └─ metrics.json        │
└────────────────────┬───────────────────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────────────────┐
│                      SERVING LAYER                                 │
│  FastAPI ──► /predict  (store_id, item_id, date → forecast)        │
│          ──► /health   (model status + metrics)                    │
│          ──► /reload   (hot-swap model without restart)            │
└────────────────────┬───────────────────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────────────────┐
│                    MONITORING & RETRAINING                          │
│  monitor_once() ──► PSI + KS on 5 high-SHAP features              │
│                     ├─ drift detected? ──► retrain_pipeline()      │
│                     │                      ├─ train candidate      │
│                     │                      ├─ promotion gate       │
│                     │                      │  (SMAPE ≤ old × 1.05)│
│                     │                      └─ 24h cooldown         │
│                     └─ log to drift_logs/                          │
└────────────────────┬───────────────────────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────────────────────┐
│                    DASHBOARD (Streamlit)                            │
│  Overview │ Forecast Explorer │ Historical │ Drift │ Model & System │
└────────────────────────────────────────────────────────────────────┘
```

---

## Model Performance

### Model Comparison (3 walk-forward folds, 28 days each)

| Model | SMAPE | MAE | RMSE |
|-------|------:|----:|-----:|
| **XGBoost (tuned)** | **12.40%** | **5.85** | **7.58** |
| LightGBM | 12.53% | 5.91 | 7.65 |
| Prophet | 13.48% | 6.19 | 8.06 |
| LSTM | 15.96% | 6.36 | 8.28 |
| SARIMA | 17.41% | 8.19 | 10.47 |
| SeasonalNaive | 18.69% | 9.06 | 12.02 |

XGBoost wins because it learns shared patterns across all 500 series as a global model, unlike per-series classical methods.

### SHAP Feature Importance

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------:|
| 1 | sales_lag_364 | 5.80 |
| 2 | sales_rmean_7 | 5.05 |
| 3 | sales_rmean_28 | 4.70 |
| 4 | month | 4.18 |
| 5 | sales_rmean_90 | 3.71 |

The model relies on lagged history and seasonality, not store/item identity (ranked 18th/19th by SHAP).

---

## Project Structure

```
├── src/                         # Core Python package
│   ├── config.py                # Paths, feature list, tuned params, thresholds
│   ├── data/loader.py           # Load and validate train.csv
│   ├── features/engineering.py  # build_features() with persisted category mappings
│   ├── models/
│   │   ├── train.py             # Train XGBoost, evaluate, save artifacts
│   │   └── registry.py          # Load model + serving table for the API
│   ├── monitoring/
│   │   ├── drift.py             # PSI and KS test implementations
│   │   ├── performance.py       # SMAPE tracking vs baseline
│   │   ├── monitor.py           # monitor_once() + monitor_full() with Evidently
│   │   └── evidently_monitor.py # Evidently AI HTML drift reports
│   └── pipelines/
│       └── retrain.py           # Retraining with promotion gate + cooldown
├── app/main.py                  # FastAPI: /predict, /health, /reload
├── dashboard/app.py             # Streamlit: 5-page interactive dashboard
├── scripts/simulate_drift.py   # Inject synthetic drift (3 modes)
├── tests/                       # 19 tests: features, drift, API, Evidently
├── notebooks/                   # EDA, model comparison, tuning, SHAP
├── k8s/                         # Kubernetes: deployments, services, CronJob, ingress
├── .github/workflows/           # CI (lint + test) and CD (build + deploy to K8s)
├── dvc.yaml                     # DVC pipeline definition
├── params.yaml                  # DVC-tracked parameters
├── Dockerfile                   # API container
├── Dockerfile.dashboard         # Dashboard container
└── docker-compose.yml           # Local orchestration
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.models.train
```

Creates `artifacts/` with the trained model, category mappings, serving table, and evaluation metrics.

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

### 4. Start the dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Test a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_id": 1, "item_id": 1, "date": "2018-01-15"}'
```

Response:
```json
{"store_id": 1, "item_id": 1, "date": "2018-01-15", "forecast": 12.8}
```

---

## Feature Engineering

19 features, all leakage-safe (use only past data):

| Category | Features | Description |
|----------|----------|-------------|
| **Calendar** | `dow`, `is_weekend`, `month`, `weekofyear`, `year`, `dayofyear` | Temporal patterns |
| **Lag** | `sales_lag_1`, `_7`, `_14`, `_28`, `_364` | Past sales at specific offsets |
| **Rolling** | `sales_rmean_7/28/90`, `sales_rstd_7/28/90` | Rolling mean/std on shifted series |
| **Identity** | `store_code`, `item_code` | Persisted integer mappings (no train/serve skew) |

Category mappings are saved as JSON at training time and loaded identically at serving time — fixing the `.cat.codes` production skew issue common in notebook-to-production transitions.

---

## Drift Detection

Monitors the top 5 SHAP features using two complementary statistical tests:

| Test | Threshold | What it catches |
|------|-----------|-----------------|
| **PSI** (Population Stability Index) | > 0.25 | Distribution shape changes |
| **KS** (Kolmogorov-Smirnov) | p < 0.05 | Any distributional difference |

### Simulate drift

Three modes to demo the detection system:

```bash
# Sudden demand shock: 1.8x sales for items 1-5
python -m scripts.simulate_drift --mode demand_shock

# Distribution shift: swap sales between high/low items
python -m scripts.simulate_drift --mode distribution_shift

# Gradual drift: 0.2% daily compounding growth
python -m scripts.simulate_drift --mode gradual
```

---

## Retraining Pipeline

Automated retraining with safeguards to prevent bad model deployments:

1. **Detect drift** via PSI/KS on monitored features
2. **Train candidate** model on full dataset
3. **Promotion gate** — candidate SMAPE must be ≤ production SMAPE × 1.05
4. **Cooldown** — 24-hour minimum between attempts (prevents retrain loops)
5. **Hot reload** — call `/reload` to swap the model without API restart

```bash
# Run with cooldown check
python -m src.pipelines.retrain

# Force retrain (skip cooldown)
python -m src.pipelines.retrain --force
```

---

## Dashboard

Interactive Streamlit dashboard with 5 pages:

| Page | Features |
|------|----------|
| **Overview** | Key metrics, model/drift status, quick forecast, dataset summary chart |
| **Forecast Explorer** | Single-day, multi-day (with plotly chart + CSV download), store comparison, demand heatmap |
| **Historical Analysis** | Sales trends with moving averages, store analysis, seasonal patterns, top/bottom items |
| **Drift Monitoring** | PSI timeline with threshold lines, per-feature deep dive, event timeline, raw reports |
| **Model & System** | Feature inventory, API controls (reload/health), one-click retraining, drift simulation |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status and metrics |
| `/predict` | POST | Forecast for a store/item/date |
| `/reload` | POST | Hot-reload model after retraining |

---

## Docker

```bash
# Train the model first
python -m src.models.train

# Start API + dashboard
docker compose up --build

# Run training in Docker
docker compose --profile train run training
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Dashboard | http://localhost:8501 |

---

## Kubernetes

```bash
kubectl apply -f k8s/
kubectl get all -n demand-forecast
```

| Resource | Description |
|----------|-------------|
| `forecast-api` Deployment | 2 replicas with liveness/readiness probes |
| `forecast-dashboard` Deployment | Streamlit UI |
| `forecast-retrain` CronJob | Weekly retraining (Sundays 2 AM) |
| `artifacts-pvc` / `data-pvc` | Persistent storage for model artifacts and data |
| `forecast-ingress` | Nginx ingress routing (`/api` → API, `/` → dashboard) |

---

## CI/CD

| Workflow | Trigger | Steps |
|----------|---------|-------|
| **CI** (`ci.yml`) | Push / PR to `main` | Ruff lint → pytest |
| **CD** (`cd.yml`) | Push to `main` | Test → Build Docker images → Push to GHCR → Deploy to K8s |

### Required GitHub secrets

| Secret | Description |
|--------|-------------|
| `KUBE_CONFIG` | Base64-encoded kubeconfig for cluster access |

`GITHUB_TOKEN` is automatically available for GHCR image pushes.

---

## Data Versioning (DVC)

Data and model artifacts are version-controlled with [DVC](https://dvc.org), keeping large files out of Git while tracking them reproducibly.

```bash
# Pull data files from remote storage
dvc pull

# Reproduce the full pipeline (train → monitor)
dvc repro

# Track a changed data file
dvc add data/train.csv
```

### DVC Pipeline Stages

| Stage | Dependencies | Outputs |
|-------|-------------|---------|
| `train` | `data/train.csv`, `src/` source, `params.yaml` | `artifacts/model.joblib`, `metrics.json`, `serving_table.parquet`, `category_mappings.json` |
| `monitor` | `artifacts/`, `src/monitoring/` | `artifacts/evidently_reports/`, `artifacts/drift_logs/` |

Parameters in `params.yaml` mirror `src/config.py` — DVC tracks them so any parameter change triggers the right pipeline stages.

---

## Evidently AI Drift Reports

Production-grade drift monitoring with [Evidently AI](https://www.evidentlyai.com/) (v0.7+) generates interactive HTML reports alongside the lightweight PSI/KS checks.

```bash
# Generate a drift report on the current data
python -m src.monitoring.evidently_monitor

# Generate reports via drift simulation
python -m scripts.simulate_drift --mode demand_shock
```

Reports are saved to `artifacts/evidently_reports/` as:
- **HTML report** — interactive drift dashboard viewable in any browser
- **JSON summary** — machine-readable results with per-column drift scores

The Streamlit dashboard embeds Evidently HTML reports directly in the **Drift Monitoring** page under the **Evidently AI Reports** tab.

### Custom PSI/KS vs Evidently

| Aspect | Custom PSI/KS | Evidently AI |
|--------|--------------|--------------|
| Speed | Fast (~ms) | Slower (~seconds) |
| Dependency | scipy only | evidently package |
| Output | JSON logs | Interactive HTML + JSON |
| Coverage | Top 5 SHAP features | All features |
| Use case | Real-time monitoring | Detailed investigation |

Both run together via `monitor_full()` — the custom checks are the fast gate, Evidently provides the deep dive.

---

## Tests

```bash
pytest tests/ -v
```

| Test file | What it verifies |
|-----------|-----------------|
| `test_features.py` | Lag features don't leak future data; category mappings are deterministic |
| `test_drift.py` | PSI returns ~0 for identical distributions, >0.25 for shifted ones |
| `test_api.py` | `/predict` returns 200 with numeric forecast; unknown store/item returns 400 |
| `test_evidently.py` | Evidently drift reports detect drift correctly and save HTML + JSON artifacts |

---

## Dataset

**Kaggle Store Item Demand Forecasting Challenge** (`demand-forecasting-kernels-only`)

| Property | Value |
|----------|-------|
| Rows | 913,000 (train) |
| Date range | 2013-01-01 to 2017-12-31 |
| Series | 500 (10 stores × 50 items) |
| Target | Daily sales (dense, non-intermittent) |
| Mean sales | 52.25 units/day |
| Growth | +35% cumulative over 5 years |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Model | XGBoost (Optuna-tuned) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Drift Detection | PSI + KS (scipy) + Evidently AI |
| Data Versioning | DVC |
| Explainability | SHAP |
| Containers | Docker + Docker Compose |
| Orchestration | Kubernetes |
| CI/CD | GitHub Actions → GHCR → K8s |
| Testing | pytest |
| Linting | Ruff |
