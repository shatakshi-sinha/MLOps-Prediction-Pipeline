"""
Data Quality Gate — validates incoming C-MAPSS data before it enters
the training or inference pipeline.

Checks:
  • Schema validation (expected columns and dtypes).
  • Value-range checks for non-physical readings.
  • Missing-value ratio assertions.
  • Feature count assertions.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Expected schema ────────────────────────────────────────────────
INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"s_{i}" for i in range(1, 22)]
EXPECTED_COLUMNS = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

# Physically plausible ranges per sensor (generous bounds)
SENSOR_RANGES: Dict[str, tuple] = {
    "s_1": (400, 700),
    "s_2": (550, 700),
    "s_3": (1000, 1700),
    "s_4": (900, 1600),
    "s_5": (0, 25),
    "s_6": (0, 80),
    "s_7": (100, 700),
    "s_8": (2300, 2500),
    "s_9": (7500, 10500),
    "s_10": (0, 2),
    "s_11": (30, 60),
    "s_12": (100, 700),
    "s_13": (2000, 2500),
    "s_14": (7000, 9500),
    "s_15": (5, 13),
    "s_16": (0, 1),
    "s_17": (300, 500),
    "s_18": (1800, 2600),
    "s_19": (50, 120),
    "s_20": (10, 50),
    "s_21": (15, 30),
}


class DataValidationError(Exception):
    """Raised when a data quality check fails."""


class DataQualityGate:
    """
    Validates incoming C-MAPSS DataFrames.

    Usage:
        gate = DataQualityGate()
        gate.validate(df)   # raises DataValidationError on failure
    """

    def __init__(
        self,
        expected_columns: Optional[List[str]] = None,
        sensor_ranges: Optional[Dict[str, tuple]] = None,
        max_missing_ratio: float = 0.05,
    ):
        self.expected_columns = expected_columns or EXPECTED_COLUMNS
        self.sensor_ranges = sensor_ranges or SENSOR_RANGES
        self.max_missing_ratio = max_missing_ratio

    # ── public entry point ─────────────────────────────────────────
    def validate(self, df: pd.DataFrame) -> bool:
        """Run all quality checks.  Returns True on pass, raises on fail."""
        self._check_schema(df)
        self._check_missing_values(df)
        self._check_value_ranges(df)
        self._check_feature_count(df)
        logger.info("Data quality gate: ALL CHECKS PASSED  ✓")
        return True

    # ── individual checks ──────────────────────────────────────────
    def _check_schema(self, df: pd.DataFrame) -> None:
        missing = set(self.expected_columns) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Schema validation failed — missing columns: {sorted(missing)}"
            )
        logger.info("Schema check passed.")

    def _check_missing_values(self, df: pd.DataFrame) -> None:
        ratios = df.isnull().mean()
        bad = ratios[ratios > self.max_missing_ratio]
        if not bad.empty:
            raise DataValidationError(
                f"Missing-value ratio exceeded {self.max_missing_ratio} "
                f"for columns: {dict(bad)}"
            )
        logger.info("Missing-value check passed.")

    def _check_value_ranges(self, df: pd.DataFrame) -> None:
        violations: List[str] = []
        for sensor, (lo, hi) in self.sensor_ranges.items():
            if sensor not in df.columns:
                continue
            col = df[sensor]
            out_lo = (col < lo).sum()
            out_hi = (col > hi).sum()
            if out_lo > 0 or out_hi > 0:
                violations.append(
                    f"{sensor}: {out_lo} below {lo}, {out_hi} above {hi}"
                )
        if violations:
            logger.warning(
                "Value-range warnings (may be acceptable): %s",
                "; ".join(violations),
            )
        logger.info("Value-range check completed.")

    def _check_feature_count(self, df: pd.DataFrame) -> None:
        expected = len(self.expected_columns)
        actual = len(df.columns)
        if actual < expected:
            raise DataValidationError(
                f"Feature count too low: expected ≥{expected}, got {actual}"
            )
        logger.info("Feature-count check passed.")
