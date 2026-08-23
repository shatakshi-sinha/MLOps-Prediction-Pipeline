import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import settings
from src.data_validation import DataQualityGate, EXPECTED_COLUMNS, SETTING_NAMES, SENSOR_NAMES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
COL_NAMES = EXPECTED_COLUMNS


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)


def explicit_drift_metrics(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    features = SETTING_NAMES + SENSOR_NAMES
    metrics = {}
    for col in features:
        ref = reference_df[col].dropna().to_numpy(dtype=float)
        cur = current_df[col].dropna().to_numpy(dtype=float)
        if len(ref) == 0 or len(cur) == 0:
            metrics[col] = {"ks_statistic": None, "ks_pvalue": None, "wasserstein_distance": None, "drifted": False}
            continue
        ks = ks_2samp(ref, cur)
        scale = max(float(np.std(ref)), 1e-12)
        distance = float(wasserstein_distance(ref, cur) / scale)
        metrics[col] = {"ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue), "wasserstein_distance": distance,
                        "drifted": bool(ks.pvalue < settings.monitoring.ks_pvalue_threshold or distance > settings.monitoring.wasserstein_threshold)}
    drifted = sum(item["drifted"] for item in metrics.values())
    return {"features": metrics, "summary": {"total_features": len(metrics), "drifted_features": drifted, "share_drifted": drifted / len(metrics) if metrics else 0.0}}


def run_drift_report(reference_df, current_df, report_path=None, metrics_path=None):
    DataQualityGate().validate(reference_df)
    DataQualityGate().validate(current_df)
    report_path = report_path or str(settings.paths.absolute(settings.paths.drift_report_path))
    metrics_path = metrics_path or str(settings.paths.absolute(settings.paths.drift_metrics_path))
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=ColumnMapping(numerical_features=SETTING_NAMES + SENSOR_NAMES))
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(report_path)
    output = {"evidently": report.as_dict(), "explicit": explicit_drift_metrics(reference_df, current_df)}
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(output, indent=2, default=str))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=str(settings.paths.absolute(settings.paths.train_data_path)))
    parser.add_argument("--current", default=str(settings.paths.absolute(settings.paths.test_data_path)))
    parser.add_argument("--report", default=None)
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args()
    run_drift_report(load_raw_data(args.reference), load_raw_data(args.current), args.report, args.metrics)


if __name__ == "__main__":
    main()
