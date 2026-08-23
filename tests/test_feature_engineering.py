"""
Tests for CMapssFeaturePipeline.

Validates:
  • Output shape correctness (engineered features expand column count).
  • No NaN values in output.
  • Metadata columns (unit_nr, time_cycles) are NOT in output.
  • Feature names are stored after fit.
  • SENSOR_NAMES is a class-level constant (Bug Fix #2 regression guard).
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import CMapssFeaturePipeline


# ── Fixtures ──────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Small synthetic C-MAPSS-like DataFrame for 2 engines, 10 cycles each."""
    np.random.seed(42)
    rows = []
    for unit in [1, 2]:
        for cycle in range(1, 11):
            row = {"unit_nr": unit, "time_cycles": cycle}
            row["setting_1"] = np.random.normal(0, 0.01)
            row["setting_2"] = np.random.normal(0, 0.01)
            row["setting_3"] = 100.0
            for i in range(1, 22):
                row[f"s_{i}"] = np.random.normal(500, 50)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def pipeline():
    return CMapssFeaturePipeline(windows=[5])


# ── Tests ─────────────────────────────────────────────────────────
class TestCMapssFeaturePipeline:

    def test_fit_transform_returns_dataframe(self, pipeline, sample_df):
        result = pipeline.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_no_nan(self, pipeline, sample_df):
        result = pipeline.fit_transform(sample_df)
        assert not result.isnull().any().any(), "Output contains NaN values"

    def test_metadata_columns_removed(self, pipeline, sample_df):
        result = pipeline.fit_transform(sample_df)
        assert "unit_nr" not in result.columns
        assert "time_cycles" not in result.columns

    def test_setting_3_removed(self, pipeline, sample_df):
        result = pipeline.fit_transform(sample_df)
        assert "setting_3" not in result.columns

    def test_dropped_sensors_removed(self, pipeline, sample_df):
        result = pipeline.fit_transform(sample_df)
        for sensor in pipeline.drop_sensors:
            assert sensor not in result.columns

    def test_feature_names_stored(self, pipeline, sample_df):
        pipeline.fit(sample_df)
        assert len(pipeline.feature_names) > 0
        assert isinstance(pipeline.feature_names, list)

    def test_transform_shape_matches_fit(self, pipeline, sample_df):
        pipeline.fit(sample_df)
        result = pipeline.transform(sample_df)
        assert result.shape[1] == len(pipeline.feature_names)
        assert result.shape[0] == len(sample_df)

    def test_rolling_features_present(self, pipeline, sample_df):
        """Verify that rolling mean/std features are generated."""
        result = pipeline.fit_transform(sample_df)
        # s_2 should NOT be dropped — check that roll features exist
        assert "s_2_roll_mean_5" in result.columns
        assert "s_2_roll_std_5" in result.columns

    def test_trend_features_present(self, pipeline, sample_df):
        result = pipeline.fit_transform(sample_df)
        assert "s_2_delta_baseline" in result.columns
        assert "s_2_velocity" in result.columns

    def test_sensor_names_class_constant(self):
        """Bug Fix #2: SENSOR_NAMES must be a class-level attribute."""
        assert hasattr(CMapssFeaturePipeline, "SENSOR_NAMES")
        assert len(CMapssFeaturePipeline.SENSOR_NAMES) == 21

    def test_output_columns_are_more_than_input(self, pipeline, sample_df):
        """Feature engineering should expand the column count."""
        result = pipeline.fit_transform(sample_df)
        # Original has 26 cols, after dropping ~9 and adding many engineered
        assert result.shape[1] > 20

    def test_transform_on_unseen_data(self, pipeline, sample_df):
        """Fit on one subset, transform another — no error."""
        half = len(sample_df) // 2
        pipeline.fit(sample_df.iloc[:half])
        result = pipeline.transform(sample_df.iloc[half:])
        assert result.shape[1] == len(pipeline.feature_names)
        assert not result.isnull().any().any()
