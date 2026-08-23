import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint

from src.config import settings
from src.data_validation import DataQualityGate, DataValidationError

logger = logging.getLogger(__name__)
app = FastAPI(title="C-MAPSS Predictive Maintenance API", version="1.1.0")


class SensorReading(BaseModel):
    unit_nr: conint(gt=0)
    time_cycles: conint(gt=0)
    setting_1: float; setting_2: float; setting_3: float
    s_1: float; s_2: float; s_3: float; s_4: float; s_5: float; s_6: float; s_7: float
    s_8: float; s_9: float; s_10: float; s_11: float; s_12: float; s_13: float
    s_14: float; s_15: float; s_16: float; s_17: float; s_18: float; s_19: float; s_20: float; s_21: float


class PredictionRequest(BaseModel):
    readings: List[SensorReading] = Field(min_length=1)


class PredictionResponse(BaseModel):
    unit_nr: int
    maintenance_required: bool
    failure_probability: float
    optimal_threshold: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


_model = None
_pipeline = None
_threshold = 0.5
_model_version = "local-artifact"


def _load_artifacts():
    global _model, _pipeline, _threshold, _model_version
    paths = settings.paths
    model_path = paths.absolute(paths.model_export_path)
    pipeline_path = paths.absolute(paths.pipeline_export_path)
    threshold_path = paths.absolute(paths.threshold_export_path)
    if not model_path.exists() or not pipeline_path.exists():
        _model = _pipeline = None
        return
    _pipeline, _model = joblib.load(pipeline_path), joblib.load(model_path)
    if threshold_path.exists():
        _threshold = float(joblib.load(threshold_path))
    _model_version = model_path.stat().st_mtime_ns.__str__()


@app.on_event("startup")
async def startup():
    _load_artifacts()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model_loaded=_model is not None)


@app.get("/model-info")
async def model_info():
    if _model is None or _pipeline is None:
        raise HTTPException(503, "Model not loaded")
    return {"model_path": str(settings.paths.absolute(settings.paths.model_export_path)), "pipeline_path": str(settings.paths.absolute(settings.paths.pipeline_export_path)), "threshold": _threshold, "feature_count": len(_pipeline.feature_names), "model_version": _model_version}


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictionRequest):
    if _model is None or _pipeline is None:
        raise HTTPException(503, "Model not loaded")
    df = pd.DataFrame([r.model_dump() for r in request.readings])
    try:
        DataQualityGate().validate(df)
        transformed = _pipeline.transform(df)
        transformed["unit_nr"] = df["unit_nr"].values
        latest = transformed.groupby("unit_nr").tail(1)
        unit_nrs = latest["unit_nr"].astype(int).tolist()
        probas = _model.predict_proba(latest.drop(columns="unit_nr"))[:, 1]
        return [PredictionResponse(unit_nr=uid, maintenance_required=bool(prob > _threshold), failure_probability=float(prob), optimal_threshold=_threshold, model_version=_model_version) for uid, prob in zip(unit_nrs, probas)]
    except DataValidationError as exc:
        raise HTTPException(422, "Input failed data quality validation") from exc
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(500, "Prediction failed") from exc


@app.post("/rollback")
async def rollback_endpoint():
    try:
        from src.promote import rollback
        rollback()
        _load_artifacts()
        return {"status": "rolled back"}
    except Exception as exc:
        logger.exception("Rollback failed: %s", exc)
        raise HTTPException(500, "Rollback failed") from exc
