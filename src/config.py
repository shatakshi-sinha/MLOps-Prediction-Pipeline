import os
import yaml
from pathlib import Path
from typing import List
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    train_data_path: str = "data/raw/train_FD001.txt"
    test_data_path: str = "data/raw/test_FD001.txt"
    rul_data_path: str = "data/raw/RUL_FD001.txt"
    pipeline_export_path: str = "models/pipeline.joblib"
    model_export_path: str = "models/model.joblib"
    drift_report_path: str = "monitoring/drift_report.html"
    drift_metrics_path: str = "monitoring/drift_metrics.json"

class FeaturesConfig(BaseModel):
    rolling_windows: List[int] = [5, 14, 21]
    drop_sensors: List[str] = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    setting_to_drop: str = 'setting_3'

class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "CMAPSS_Predictive_Maintenance"
    registered_model_name: str = "CMAPSS_XGBoost"

class QualityGateConfig(BaseModel):
    f1_threshold: float = 0.85
    epsilon: float = 0.005

class Settings(BaseSettings):
    paths: PathsConfig = PathsConfig()
    features: FeaturesConfig = FeaturesConfig()
    mlflow: MLflowConfig = MLflowConfig()
    quality_gate: QualityGateConfig = QualityGateConfig()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="CMAPSS_",
        case_sensitive=False
    )

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config/config.yaml") -> "Settings":
        path = Path(yaml_path)
        yaml_data = {}
        if path.exists():
            with open(path, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
        return cls(**yaml_data)

settings = Settings.load_from_yaml()
