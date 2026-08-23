"""
Tests for Quality Gate Promotion & Rollback Engine.

Uses mock MLflow client to test promotion logic (pass/fail) and
rollback logic without requiring a real MLflow server.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.promote import promote, rollback


# ── Helpers ───────────────────────────────────────────────────────
def _make_model_version(version: str, run_id: str):
    mv = MagicMock()
    mv.version = version
    mv.run_id = run_id
    return mv


def _make_run(f1: float):
    run = MagicMock()
    run.data.metrics = {"val_f1": f1}
    return run


# ── Promotion Tests ───────────────────────────────────────────────
class TestPromote:

    @patch("src.promote.get_client")
    def test_promote_passes_when_above_threshold(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client

        candidate = _make_model_version("1", "run-1")
        client.search_model_versions.return_value = [candidate]
        client.get_model_version_by_alias.side_effect = Exception("no alias")
        client.get_run.return_value = _make_run(0.92)

        result = promote("TestModel")
        assert result is True
        client.set_registered_model_alias.assert_called_once()

    @patch("src.promote.get_client")
    def test_promote_rejects_below_threshold(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client

        candidate = _make_model_version("1", "run-1")
        client.search_model_versions.return_value = [candidate]
        client.get_model_version_by_alias.side_effect = Exception("no alias")
        client.get_run.return_value = _make_run(0.50)

        result = promote("TestModel")
        assert result is False
        client.set_registered_model_alias.assert_not_called()

    @patch("src.promote.get_client")
    def test_promote_rejects_when_not_improving(self, mock_get_client):
        """Candidate must beat production + epsilon."""
        client = MagicMock()
        mock_get_client.return_value = client

        candidate = _make_model_version("2", "run-2")
        prod = _make_model_version("1", "run-1")

        client.search_model_versions.return_value = [
            _make_model_version("1", "run-1"),
            candidate,
        ]
        client.get_model_version_by_alias.return_value = prod

        # Both have identical F1
        def get_run_side_effect(run_id):
            return _make_run(0.90)

        client.get_run.side_effect = get_run_side_effect

        result = promote("TestModel")
        assert result is False

    @patch("src.promote.get_client")
    def test_promote_passes_when_improving(self, mock_get_client):
        """Candidate beats production + epsilon → promote."""
        client = MagicMock()
        mock_get_client.return_value = client

        candidate = _make_model_version("2", "run-2")
        prod = _make_model_version("1", "run-1")

        client.search_model_versions.return_value = [
            _make_model_version("1", "run-1"),
            candidate,
        ]
        client.get_model_version_by_alias.return_value = prod

        def get_run_side_effect(run_id):
            if run_id == "run-1":
                return _make_run(0.88)
            return _make_run(0.92)

        client.get_run.side_effect = get_run_side_effect

        result = promote("TestModel")
        assert result is True


# ── Rollback Tests ────────────────────────────────────────────────
class TestRollback:

    @patch("src.promote.get_client")
    def test_rollback_reverts_to_previous(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client

        prod = _make_model_version("3", "run-3")
        client.get_model_version_by_alias.return_value = prod

        result = rollback("TestModel")
        assert result is True
        client.set_registered_model_alias.assert_called_once_with(
            "TestModel", "production", "2"
        )

    @patch("src.promote.get_client")
    def test_rollback_fails_on_version_1(self, mock_get_client):
        """Cannot roll back from version 1."""
        client = MagicMock()
        mock_get_client.return_value = client

        prod = _make_model_version("1", "run-1")
        client.get_model_version_by_alias.return_value = prod

        with pytest.raises(SystemExit):
            rollback("TestModel")

    @patch("src.promote.get_client")
    def test_rollback_fails_with_no_production(self, mock_get_client):
        client = MagicMock()
        mock_get_client.return_value = client
        client.get_model_version_by_alias.side_effect = Exception("no alias")

        with pytest.raises(SystemExit):
            rollback("TestModel")
