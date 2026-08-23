"""
FastAPI REST Application for C-MAPSS Predictive Maintenance.

Endpoints:
    POST /predict   — raw engine telemetry → maintenance prediction
    GET  /health    — service health check
    GET  /model-info — active production model metadata
    POST /rollback  — instant rollback to previous stable model

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.feature_engineering import CMapssFeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="C-MAPSS Predictive Maintenance API",
    description="Predict turbofan engine failure from sensor telemetry",
    version="1.0.0",
)


# ── Request / Response schemas ─────────────────────────────────────
class SensorReading(BaseModel):
    unit_nr: int
    time_cycles: int
    setting_1: float
    setting_2: float
    setting_3: float
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float


class PredictionRequest(BaseModel):
    readings: List[SensorReading]


class PredictionResponse(BaseModel):
    unit_nr: int
    maintenance_required: bool
    failure_probability: float
    optimal_threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_path: str
    pipeline_path: str
    threshold: float
    feature_count: int


# ── Global model cache ─────────────────────────────────────────────
_model = None
_pipeline = None
_threshold = 0.5


def _load_artifacts():
    global _model, _pipeline, _threshold
    models_dir = Path("models")
    model_path = models_dir / "model.joblib"
    pipeline_path = models_dir / "pipeline.joblib"
    threshold_path = models_dir / "threshold.joblib"

    if model_path.exists() and pipeline_path.exists():
        _pipeline = joblib.load(pipeline_path)
        _model = joblib.load(model_path)
        if threshold_path.exists():
            _threshold = joblib.load(threshold_path)
        logger.info("Model artifacts loaded (threshold=%.2f)", _threshold)
    else:
        logger.warning("Model artifacts not found — /predict will return 503")


@app.on_event("startup")
async def startup():
    _load_artifacts()


# ── Endpoints ──────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if _model is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_path="models/model.joblib",
        pipeline_path="models/pipeline.joblib",
        threshold=_threshold,
        feature_count=len(_pipeline.feature_names),
    )


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictionRequest):
    if _model is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build DataFrame from request
    records = [r.model_dump() for r in request.readings]
    df = pd.DataFrame(records)

    try:
        # Feature engineering — Bug Fix #1: safe copy approach
        X_full = _pipeline.transform(df)
        X_full_copy = X_full.copy()
        X_full_copy["unit_nr"] = df["unit_nr"].values

        # Extract last reading per engine
        latest = (
            X_full_copy.groupby("unit_nr").tail(1).drop(columns=["unit_nr"])
        )

        probas = _model.predict_proba(latest)[:, 1]
        preds = (probas > _threshold).astype(int)

        unit_nrs = (
            X_full_copy.groupby("unit_nr").tail(1)["unit_nr"].values
        )

        results = []
        for uid, prob, pred in zip(unit_nrs, probas, preds):
            results.append(
                PredictionResponse(
                    unit_nr=int(uid),
                    maintenance_required=bool(pred),
                    failure_probability=float(prob),
                    optimal_threshold=_threshold,
                    model_version="models/model.joblib",
                )
            )
        return results

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rollback")
async def rollback_endpoint():
    """Trigger model rollback via the promote module."""
    try:
        from src.promote import rollback as do_rollback
        do_rollback()
        _load_artifacts()
        return {"status": "rolled back", "message": "Production model reverted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
