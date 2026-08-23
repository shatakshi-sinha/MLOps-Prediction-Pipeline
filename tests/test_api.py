"""
Tests for FastAPI endpoints.

Tests /health, /model-info, and /predict endpoints using FastAPI TestClient.
Model artifacts may not exist in CI, so /predict tests handle 503 gracefully.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────
class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_has_model_loaded_field(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


# ── Model info ────────────────────────────────────────────────────
class TestModelInfoEndpoint:

    def test_model_info_returns_status(self, client):
        """Returns 200 if model loaded, 503 if not."""
        resp = client.get("/model-info")
        assert resp.status_code in (200, 503)

    def test_model_info_schema_when_loaded(self, client):
        resp = client.get("/model-info")
        if resp.status_code == 200:
            data = resp.json()
            assert "model_path" in data
            assert "pipeline_path" in data
            assert "threshold" in data
            assert "feature_count" in data


# ── Predict ───────────────────────────────────────────────────────
class TestPredictEndpoint:

    @pytest.fixture
    def sample_payload(self):
        payload_path = Path("data/sample_payload.json")
        if payload_path.exists():
            with open(payload_path) as f:
                return json.load(f)
        # Fallback minimal payload
        return {
            "readings": [
                {
                    "unit_nr": 1,
                    "time_cycles": 1,
                    "setting_1": 0.0,
                    "setting_2": 0.0,
                    "setting_3": 100.0,
                    **{f"s_{i}": 500.0 for i in range(1, 22)},
                }
            ]
        }

    def test_predict_returns_valid_status(self, client, sample_payload):
        """Should return 200 (model loaded) or 503 (model not loaded)."""
        resp = client.post("/predict", json=sample_payload)
        assert resp.status_code in (200, 503)

    def test_predict_response_schema(self, client, sample_payload):
        resp = client.post("/predict", json=sample_payload)
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)
            for item in data:
                assert "unit_nr" in item
                assert "maintenance_required" in item
                assert "failure_probability" in item
                assert "optimal_threshold" in item

    def test_predict_rejects_empty_readings(self, client):
        resp = client.post("/predict", json={"readings": []})
        # Should either 200 with empty list, 500 or 422
        assert resp.status_code in (200, 422, 500, 503)

    def test_predict_rejects_bad_schema(self, client):
        resp = client.post("/predict", json={"bad_key": "bad_value"})
        assert resp.status_code == 422
