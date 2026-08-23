"""
Evidently-based Data Drift & Performance Monitoring.

Calculates Kolmogorov-Smirnov (KS) test and Wasserstein Distance across
sensor channels between a reference baseline and incoming production
inference batches.

Outputs:
  • HTML interactive report
  • Structured JSON drift metrics

Usage:
    python -m src.monitoring
    python -m src.monitoring --reference data/raw/train_FD001.txt \
                             --current  data/raw/test_FD001.txt
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Column definitions ────────────────────────────────────────────
INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"s_{i}" for i in range(1, 22)]
COL_NAMES = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_path: str | None = None,
    metrics_path: str | None = None,
):
    """Generate Evidently drift report and JSON metrics."""
    report_path = report_path or settings.paths.drift_report_path
    metrics_path = metrics_path or settings.paths.drift_metrics_path

    # Use only sensor columns for drift detection
    column_mapping = ColumnMapping(
        numerical_features=SENSOR_NAMES + SETTING_NAMES,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    # Save HTML report
    out_dir = Path(report_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(report_path)
    logger.info("Drift report saved to %s", report_path)

    # Save JSON metrics
    report_dict = report.as_dict()
    metrics_out = Path(metrics_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    logger.info("Drift metrics saved to %s", metrics_path)

    # Log summary
    try:
        drift_result = report_dict["metrics"][0]["result"]
        drift_share = drift_result.get("share_of_drifted_columns", "N/A")
        is_drift = drift_result.get("dataset_drift", False)
        logger.info(
            "Dataset drift detected: %s  (drifted columns: %s)",
            is_drift, drift_share,
        )
    except (KeyError, IndexError):
        logger.warning("Could not parse drift summary from report.")

    return report_dict


def main():
    parser = argparse.ArgumentParser(description="Data Drift Monitoring")
    parser.add_argument(
        "--reference", type=str,
        default=settings.paths.train_data_path,
        help="Path to reference data file",
    )
    parser.add_argument(
        "--current", type=str,
        default=settings.paths.test_data_path,
        help="Path to current (production) data file",
    )
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    args = parser.parse_args()

    ref_df = load_raw_data(args.reference)
    cur_df = load_raw_data(args.current)

    logger.info(
        "Reference: %d rows | Current: %d rows", len(ref_df), len(cur_df)
    )
    run_drift_report(ref_df, cur_df, args.report, args.metrics)


if __name__ == "__main__":
    main()
