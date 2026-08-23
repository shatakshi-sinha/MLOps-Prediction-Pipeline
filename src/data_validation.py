import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"s_{i}" for i in range(1, 22)]
EXPECTED_COLUMNS = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES
SENSOR_RANGES: Dict[str, tuple] = {
    "s_1": (400, 700), "s_2": (550, 700), "s_3": (1000, 1700), "s_4": (900, 1600),
    "s_5": (0, 25), "s_6": (0, 80), "s_7": (100, 700), "s_8": (2300, 2500),
    "s_9": (7500, 10500), "s_10": (0, 2), "s_11": (30, 60), "s_12": (100, 700),
    "s_13": (2000, 2500), "s_14": (7000, 9500), "s_15": (5, 13), "s_16": (0, 1),
    "s_17": (300, 500), "s_18": (1800, 2600), "s_19": (50, 120), "s_20": (10, 50), "s_21": (15, 30),
}


class DataValidationError(Exception):
    """Raised when an input data quality check fails."""


class DataQualityGate:
    def __init__(self, expected_columns: Optional[List[str]] = None, sensor_ranges: Optional[Dict[str, tuple]] = None, max_missing_ratio: float = 0.05):
        self.expected_columns = expected_columns or EXPECTED_COLUMNS
        self.sensor_ranges = sensor_ranges or SENSOR_RANGES
        self.max_missing_ratio = max_missing_ratio

    def validate(self, df: pd.DataFrame) -> bool:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise DataValidationError("Input must be a non-empty DataFrame")
        if df.index.has_duplicates or df.duplicated().any():
            raise DataValidationError("Duplicate rows are not allowed")
        self._check_schema(df)
        self._check_dtypes(df)
        self._check_missing_values(df)
        self._check_value_ranges(df)
        return True

    def _check_schema(self, df: pd.DataFrame) -> None:
        missing = sorted(set(self.expected_columns) - set(df.columns))
        extra = sorted(set(df.columns) - set(self.expected_columns))
        if missing or extra:
            raise DataValidationError(f"Schema validation failed; missing={missing}, unexpected={extra}")

    def _check_dtypes(self, df: pd.DataFrame) -> None:
        numeric = [c for c in self.expected_columns if c in df]
        invalid = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
        if invalid:
            raise DataValidationError(f"Non-numeric columns: {invalid}")

    def _check_missing_values(self, df: pd.DataFrame) -> None:
        bad = df.isna().mean()
        bad = bad[bad > self.max_missing_ratio]
        if not bad.empty:
            raise DataValidationError(f"Missing-value ratio exceeded {self.max_missing_ratio}: {bad.to_dict()}")
        if not np.isfinite(df.to_numpy(dtype=float)).all():
            raise DataValidationError("Non-finite values are not allowed")

    def _check_value_ranges(self, df: pd.DataFrame) -> None:
        violations = []
        for sensor, (lo, hi) in self.sensor_ranges.items():
            if sensor in df and ((df[sensor] < lo) | (df[sensor] > hi)).any():
                violations.append(f"{sensor} must be between {lo} and {hi}")
        if violations:
            raise DataValidationError("; ".join(violations))

    def _check_feature_count(self, df: pd.DataFrame) -> None:
        self._check_schema(df)
