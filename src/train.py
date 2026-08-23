"""
CLI Training Script for C-MAPSS Predictive Maintenance.

Loads the FD001 dataset, applies CMapssFeaturePipeline, trains XGBoost,
and logs everything to MLflow — including optimal decision threshold
(Bug Fix #4), single registry name (Bug Fix #3), and safe test inference
(Bug Fix #1).

Usage:
    python -m src.train
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from src.config import settings
from src.data_validation import DataQualityGate
from src.feature_engineering import CMapssFeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Column definitions ────────────────────────────────────────────
INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"s_{i}" for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES


# ── helpers ───────────────────────────────────────────────────────
def load_data(base_path: str):
    """Load C-MAPSS FD001 train / test / RUL files."""
    train_df = pd.read_csv(
        os.path.join(base_path, "train_FD001.txt"),
        sep=r"\s+", header=None, names=COL_NAMES,
    )
    test_df = pd.read_csv(
        os.path.join(base_path, "test_FD001.txt"),
        sep=r"\s+", header=None, names=COL_NAMES,
    )
    y_test = pd.read_csv(
        os.path.join(base_path, "RUL_FD001.txt"),
        sep=r"\s+", header=None, names=["RUL"],
    )
    return train_df, test_df, y_test


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RUL and binary classification target."""
    max_cycles = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_nr", "max_cycle"]
    df = df.merge(max_cycles, on="unit_nr", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df["label_bc"] = np.where(df["RUL"] <= 30, 1, 0)
    return df


def find_optimal_threshold(y_true, y_proba, thresholds=None):
    """Bug Fix #4 — sweep thresholds on validation set to maximise F1."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.90, 0.01)
    best_thresh = float(
        max(thresholds, key=lambda t: f1_score(y_true, (y_proba > t).astype(int)))
    )
    best_f1 = f1_score(y_true, (y_proba > best_thresh).astype(int))
    return best_thresh, best_f1


# ── main ──────────────────────────────────────────────────────────
def train(data_dir: str | None = None):
    base_path = data_dir or settings.paths.raw_data_dir
    logger.info("Loading data from %s …", base_path)
    train_df, test_df, y_test = load_data(base_path)

    # Validate incoming data
    gate = DataQualityGate()
    gate.validate(train_df)
    gate.validate(test_df)

    # Target engineering
    train_df = add_target_columns(train_df)

    # Leakage-safe entity-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(train_df, groups=train_df["unit_nr"]))
    train_data = train_df.iloc[train_idx].copy()
    val_data = train_df.iloc[val_idx].copy()
    logger.info(
        "Train: %s rows (%d units) | Val: %s rows (%d units)",
        len(train_data), train_data["unit_nr"].nunique(),
        len(val_data), val_data["unit_nr"].nunique(),
    )

    # Feature pipeline
    pipeline = CMapssFeaturePipeline(
        windows=settings.features.rolling_windows,
        drop_sensors=settings.features.drop_sensors,
    )
    meta_cols = ["max_cycle", "RUL", "label_bc"]
    X_train = pipeline.fit_transform(train_data.drop(columns=meta_cols))
    y_train = train_data["label_bc"]

    X_val = pipeline.transform(val_data.drop(columns=meta_cols))
    y_val = val_data["label_bc"]

    logger.info("Engineered feature shape: %s", X_train.shape)

    # Class imbalance weight
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    # ── MLflow tracking ────────────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    with mlflow.start_run(run_name="XGBoost_Training") as run:
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        # Predict on validation
        y_proba_val = model.predict_proba(X_val)[:, 1]

        # Bug Fix #4 — optimal threshold
        best_thresh, best_f1 = find_optimal_threshold(y_val, y_proba_val)
        logger.info("Optimal threshold: %.2f  (val F1=%.4f)", best_thresh, best_f1)

        y_pred_val = (y_proba_val > best_thresh).astype(int)

        val_metrics = {
            "val_auc": roc_auc_score(y_val, y_proba_val),
            "val_pr_auc": average_precision_score(y_val, y_proba_val),
            "val_f1": f1_score(y_val, y_pred_val, zero_division=0),
            "val_precision": precision_score(y_val, y_pred_val),
            "val_recall": recall_score(y_val, y_pred_val),
            "optimal_threshold": best_thresh,
        }

        # ── Bug Fix #1 — safe test inference ──────────────────────
        X_test_full = pipeline.transform(test_df)
        X_test_full_copy = X_test_full.copy()
        X_test_full_copy["unit_nr"] = test_df["unit_nr"].values
        test_latest = (
            X_test_full_copy.groupby("unit_nr").tail(1).drop(columns=["unit_nr"])
        )
        test_probs = model.predict_proba(test_latest)[:, 1]
        test_preds = (test_probs > best_thresh).astype(int)
        y_test_binary = (y_test["RUL"] <= 30).astype(int)

        test_metrics = {
            "test_auc": roc_auc_score(y_test_binary, test_probs),
            "test_f1": f1_score(y_test_binary, test_preds),
            "test_precision": precision_score(y_test_binary, test_preds),
            "test_recall": recall_score(y_test_binary, test_preds),
        }

        all_metrics = {**val_metrics, **test_metrics}
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(all_metrics)

        # Save & log artifacts
        models_dir = settings.paths.absolute("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, models_dir / "pipeline.joblib")
        joblib.dump(model, models_dir / "model.joblib")
        joblib.dump(best_thresh, models_dir / "threshold.joblib")
        mlflow.log_artifact(str(models_dir / "pipeline.joblib"))
        mlflow.log_artifact(str(models_dir / "model.joblib"))
        mlflow.log_artifact(str(models_dir / "threshold.joblib"))
        report = classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(models_dir / "classification_report.csv")
        pd.DataFrame(confusion_matrix(y_val, y_pred_val)).to_csv(models_dir / "confusion_matrix.csv", index=False)
        mlflow.log_artifacts(str(models_dir), artifact_path="evaluation")

        # Bug Fix #3 — single registry entry
        mlflow.xgboost.log_model(
            model, "model",
            registered_model_name=settings.mlflow.registered_model_name,
        )

        logger.info("── Validation Results ──")
        for k, v in val_metrics.items():
            logger.info("  %s: %.4f", k, v)
        logger.info("── Test Results ──")
        for k, v in test_metrics.items():
            logger.info("  %s: %.4f", k, v)
        logger.info("MLflow run ID: %s", run.info.run_id)

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train C-MAPSS model")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override path to raw data directory")
    args = parser.parse_args()
    train(data_dir=args.data_dir)
