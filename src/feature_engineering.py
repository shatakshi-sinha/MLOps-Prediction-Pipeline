"""
Production-grade Feature Engineering Pipeline for C-MAPSS dataset.

Inherits from sklearn BaseEstimator & TransformerMixin for seamless
integration with scikit-learn pipelines and joblib serialization.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from typing import List, Optional


class CMapssFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Stateful feature transformer for C-MAPSS turbofan engine data.

    Steps:
      1. Prune constant / near-constant sensors.
      2. Generate rolling statistics (mean & std) for configurable windows.
      3. Compute trend features (delta from baseline, velocity).
      4. Scale all features using RobustScaler (fitted on training data only).

    Bug Fix #2: SENSOR_NAMES is a class-level constant so the transformer
    is fully self-contained and portable across modules.
    """

    # ── class-level constant (Bug Fix #2) ──────────────────────────
    SENSOR_NAMES: List[str] = [f"s_{i}" for i in range(1, 22)]

    def __init__(
        self,
        windows: Optional[List[int]] = None,
        drop_sensors: Optional[List[str]] = None,
    ):
        self.windows = windows or [5, 14, 21]
        self.drop_sensors = drop_sensors or [
            "s_1", "s_5", "s_10", "s_16", "s_18", "s_19"
        ]
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []

    # ── public sklearn API ─────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y=None):
        X_engineered = self._engineer_features(X)
        self.scaler.fit(X_engineered)
        self.feature_names = X_engineered.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_engineered = self._engineer_features(X)
        X_scaled = self.scaler.transform(X_engineered)
        return pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

    # ── internal feature builder ───────────────────────────────────
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        new_cols = {}

        # 1. Pruning — drop constant sensors & setting_3
        df = df.drop(
            columns=self.drop_sensors + ["setting_3"], errors="ignore"
        )

        # Bug Fix #2 — use self-contained class constant
        remaining_sensors = [
            s for s in self.SENSOR_NAMES if s not in self.drop_sensors
        ]

        # 2. Rolling aggregations
        for w in self.windows:
            for s in remaining_sensors:
                new_cols[f"{s}_roll_mean_{w}"] = (
                    df.groupby("unit_nr")[s]
                    .transform(lambda x: x.rolling(window=w, min_periods=1).mean())
                )
                new_cols[f"{s}_roll_std_{w}"] = (
                    df.groupby("unit_nr")[s]
                    .transform(lambda x: x.rolling(window=w, min_periods=1).std())
                    .fillna(0)
                )

        # 3. Sensor trends & deltas
        for s in remaining_sensors:
            # Delta from engine initial baseline
            first_val = df.groupby("unit_nr")[s].transform("first")
            new_cols[f"{s}_delta_baseline"] = df[s] - first_val

            # Instantaneous velocity (current − prev rolling mean)
            prev_mean = (
                df.groupby("unit_nr")[s]
                .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
                .bfill()
            )
            new_cols[f"{s}_velocity"] = df[s] - prev_mean

        # Concatenate all new features at once
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # Drop metadata columns — they must not enter the feature matrix
        return df.drop(columns=["unit_nr", "time_cycles"], errors="ignore")
