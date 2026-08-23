import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    train_data_path: str = "data/raw/train_FD001.txt"
    test_data_path: str = "data/raw/test_FD001.txt"
    rul_data_path: str = "data/raw/RUL_FD001.txt"
    pipeline_export_path: str = "models/pipeline.joblib"
    model_export_path: str = "models/model.joblib"
    threshold_export_path: str = "models/threshold.joblib"
    drift_report_path: str = "monitoring/drift_report.html"
    drift_metrics_path: str = "monitoring/drift_metrics.json"

    def absolute(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT_DIR / path


class FeaturesConfig(BaseModel):
    rolling_windows: List[int] = [5, 14, 21]
    drop_sensors: List[str] = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
    setting_to_drop: str = "setting_3"


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "CMAPSS_Predictive_Maintenance"
    registered_model_name: str = "CMAPSS_XGBoost"


class QualityGateConfig(BaseModel):
    f1_threshold: float = 0.85
    epsilon: float = 0.005


class MonitoringConfig(BaseModel):
    ks_pvalue_threshold: float = 0.05
    wasserstein_threshold: float = 0.1


class Settings(BaseSettings):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    model_config = SettingsConfigDict(env_nested_delimiter="__", env_prefix="CMAPSS_", case_sensitive=False)

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config/config.yaml") -> "Settings":
        path = Path(yaml_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        yaml_data = {}
        if path.exists():
            with path.open() as handle:
                yaml_data = yaml.safe_load(handle) or {}
        return cls(**yaml_data)


settings = Settings.load_from_yaml()
