# C-MAPSS Predictive Maintenance — MLOps Production Pipeline

End-to-end **MLOps pipeline** for turbofan engine predictive maintenance using the [NASA C-MAPSS dataset](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq). The system predicts whether an engine requires maintenance within the next 30 operational cycles, backed by automated training, model promotion, drift detection, and a production REST API.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Quick Start](#quick-start)
5. [Training Pipeline](#training-pipeline)
6. [API Endpoints](#api-endpoints)
7. [Model Promotion & Rollback](#model-promotion--rollback)
8. [Data Drift Monitoring](#data-drift-monitoring)
9. [Testing](#testing)
10. [CI/CD Pipeline](#cicd-pipeline)
11. [Docker Deployment](#docker-deployment)
12. [Bug Fixes from Phase 1](#bug-fixes-from-phase-1)

---

## Project Overview

| Component | Technology |
|---|---|
| **ML Model** | XGBoost binary classifier |
| **Feature Pipeline** | Custom sklearn `BaseEstimator` / `TransformerMixin` |
| **Experiment Tracking** | MLflow |
| **Serving** | FastAPI + Uvicorn |
| **Data Validation** | Custom schema / range / missing-value gate |
| **Drift Detection** | Evidently AI (KS test, Wasserstein distance) |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Configuration** | YAML + Pydantic Settings |

**Key metrics (FD001 dataset):**
- Validation F1 ≥ **0.91**
- Test AUC ≥ **0.98**
- Optimal decision threshold auto-tuned per training run

---

## Architecture

```
┌────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Raw Data  │────▶│ Data Validation  │────▶│   Feature     │
│  (C-MAPSS) │     │  Quality Gate    │     │  Engineering  │
└────────────┘     └──────────────────┘     └───────┬───────┘
                                                    │
                                    ┌───────────────▼───────────────┐
                                    │     XGBoost Training          │
                                    │  + Threshold Optimisation     │
                                    │  + MLflow Logging             │
                                    └───────────────┬───────────────┘
                                                    │
                           ┌────────────────────────▼────────────────┐
                           │         Model Registry (MLflow)         │
                           │  Candidate → Quality Gate → Production  │
                           └────────────────────────┬────────────────┘
                                                    │
                    ┌───────────────────────────────▼──────────────┐
                    │            FastAPI Serving                    │
                    │  /predict  /health  /model-info  /rollback   │
                    └───────────────────────────────┬──────────────┘
                                                    │
                                    ┌───────────────▼───────────────┐
                                    │    Evidently Drift Monitor    │
                                    │  KS Test · Wasserstein Dist   │
                                    └───────────────────────────────┘
```

---

## Repository Layout

```
MLOps_Pipeline_Prod/
├── .github/workflows/ci_cd.yml      # GitHub Actions CI/CD
├── config/config.yaml                # Pipeline configuration & thresholds
├── data/
│   ├── raw/                          # C-MAPSS FD001 dataset files
│   └── sample_payload.json           # Sample API prediction payload
├── notebooks/
│   └── predictive_maintenance_colab.ipynb  # Phase 1 research notebook
├── src/
│   ├── __init__.py
│   ├── config.py                     # Pydantic settings manager
│   ├── data_validation.py            # Schema & distribution validation gate
│   ├── feature_engineering.py        # Production feature pipeline
│   ├── train.py                      # CLI MLflow training script
│   ├── promote.py                    # Quality gate promotion & rollback
│   ├── api.py                        # FastAPI application
│   └── monitoring.py                 # Evidently data drift monitoring
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_api.py
│   └── test_quality_gate.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-org/MLOps_Pipeline_Prod.git
cd MLOps_Pipeline_Prod

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Extract the dataset (if not already extracted)
# Place archive.zip in the repo root, then:
python -c "import zipfile; zipfile.ZipFile('archive.zip').extractall('data/raw')"

# 5. Train the model
python -m src.train

# 6. Start the API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 7. Test a prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @data/sample_payload.json
```

---

## Training Pipeline

```bash
# Train with default settings (reads config/config.yaml)
python -m src.train

# Train with custom data directory
python -m src.train --data-dir /path/to/data
```

**What happens:**
1. Loads C-MAPSS FD001 train/test/RUL files.
2. Runs `DataQualityGate` validation on both sets.
3. Computes RUL and binary target (fail within 30 cycles).
4. Performs leakage-safe entity-level train/validation split (`GroupShuffleSplit`).
5. Fits `CMapssFeaturePipeline` (rolling stats, trends, scaling).
6. Trains `XGBClassifier` with `scale_pos_weight` for class imbalance.
7. **Optimises decision threshold** on validation set (F1-maximising sweep).
8. Runs safe test-set inference (no feature matrix mutation).
9. Logs parameters, metrics, and artifacts to MLflow.
10. Registers model under single registry name `CMAPSS_XGBoost`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/model-info` | Active model version, threshold, feature count |
| `POST` | `/predict` | Raw telemetry → maintenance prediction |
| `POST` | `/rollback` | Revert to previous production model |

### Example: POST /predict

**Request:**
```json
{
  "readings": [
    {
      "unit_nr": 1,
      "time_cycles": 1,
      "setting_1": -0.0007,
      "setting_2": -0.0004,
      "setting_3": 100.0,
      "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
      "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.06,
      "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
      "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03,
      "s_17": 392, "s_18": 2388, "s_19": 100.0, "s_20": 39.06, "s_21": 23.419
    }
  ]
}
```

**Response:**
```json
[
  {
    "unit_nr": 1,
    "maintenance_required": false,
    "failure_probability": 0.0032,
    "optimal_threshold": 0.42,
    "model_version": "models/model.joblib"
  }
]
```

---

## Model Promotion & Rollback

### Automatic Promotion

```bash
python -m src.promote
```

**Quality Gate conditions:**
1. `F1_candidate ≥ F1_threshold (0.85)`
2. `F1_candidate > F1_production + ε (0.005)`

If both pass → candidate version gets the `production` alias in MLflow.

### Manual Rollback

```bash
# CLI
python -m src.promote --rollback

# API endpoint
curl -X POST http://localhost:8000/rollback
```

Reverts the `production` alias to the previous model version.

---

## Data Drift Monitoring

```bash
# Default: compare train (reference) vs test (current)
python -m src.monitoring

# Custom files
python -m src.monitoring \
  --reference data/raw/train_FD001.txt \
  --current data/raw/test_FD001.txt \
  --report monitoring/my_report.html \
  --metrics monitoring/my_metrics.json
```

**Output:**
- `monitoring/drift_report.html` — interactive Evidently dashboard
- `monitoring/drift_metrics.json` — structured metrics (KS statistics, Wasserstein distances, drift flags per sensor)

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_feature_engineering.py -v
pytest tests/test_api.py -v
pytest tests/test_quality_gate.py -v
```

**Test coverage:**
| Module | Tests |
|--------|-------|
| Feature Engineering | Shape, NaN, column removal, rolling/trend features, class constant |
| API | Health, model-info, predict (loaded & unloaded), schema validation |
| Quality Gate | Promotion pass/fail, threshold enforcement, rollback logic |

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci_cd.yml`) runs on push to `main`/`develop` and on pull requests:

| Stage | Actions |
|-------|---------|
| **Lint** | `black --check`, `flake8` |
| **Test** | `pytest tests/ -v` |
| **Docker** | Build image, start container, verify `/health` endpoint |

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up --build -d

# Services:
#   - API:    http://localhost:8000
#   - MLflow: http://localhost:5000

# Stop
docker-compose down
```

---

## Bug Fixes from Phase 1

### 1. Silent Column Ordering Bug
**Problem:** `unit_nr` was injected directly into the scaled feature matrix, corrupting column order.
**Fix:** A `.copy()` of the feature matrix is used; `unit_nr` is never added to the original scaled DataFrame.

### 2. `sensor_names` Scoping Bug
**Problem:** `_engineer_features()` referenced a module-level `sensor_names` variable that didn't exist outside the notebook.
**Fix:** `SENSOR_NAMES` is now a **class-level constant** on `CMapssFeaturePipeline`, making it fully self-contained and portable.

### 3. MLflow Model Registration Name
**Problem:** Each run registered under a different name (`CMAPSS_LR_Baseline`, `CMAPSS_XGBoost_Challenger`), preventing model lifecycle management.
**Fix:** All versions register under **`CMAPSS_XGBoost`**, enabling proper candidate → staging → production promotion.

### 4. No Decision Threshold Optimisation
**Problem:** Hardcoded `0.5` threshold — suboptimal for predictive maintenance where false negatives are far costlier.
**Fix:** F1-optimal threshold is **swept on the validation set** (`np.arange(0.10, 0.90, 0.01)`) before test/production inference. The threshold is saved as `models/threshold.joblib` and logged to MLflow.

---

## License

This project is for educational and research purposes.
#   M L O p s - P r e d i c t i o n - P i p e l i n e  
 