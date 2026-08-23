"""
Quality Gate Promotion & Rollback Engine.

Compares a candidate model against the current production model and
promotes or rejects it based on configurable F1 thresholds.

Usage:
    python -m src.promote              # evaluate & promote latest candidate
    python -m src.promote --rollback   # revert to previous production version
"""

import argparse
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    return MlflowClient()


def _get_production_version(client: MlflowClient, model_name: str):
    """Return the latest version with alias 'production', or None."""
    try:
        mv = client.get_model_version_by_alias(model_name, "production")
        return mv
    except Exception:
        return None


def _get_latest_version(client: MlflowClient, model_name: str):
    """Return the newest registered version that is not production."""
    prod = _get_production_version(client, model_name)
    versions = client.search_model_versions(f"name='{model_name}'")
    candidates = [v for v in versions if prod is None or str(v.version) != str(prod.version)]
    return max(candidates, key=lambda v: int(v.version)) if candidates else None


def _get_previous_version(client: MlflowClient, model_name: str, current_version: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    prior = [v for v in versions if int(v.version) < int(current_version)]
    return max(prior, key=lambda v: int(v.version)) if prior else None


def _get_f1(client: MlflowClient, run_id: str) -> float:
    """Fetch val_f1 metric from a run."""
    run = client.get_run(run_id)
    return float(run.data.metrics.get("val_f1", 0.0))


# ── Promote ───────────────────────────────────────────────────────
def promote(model_name: str | None = None):
    """Evaluate candidate model and promote if it passes quality gate."""
    model_name = model_name or settings.mlflow.registered_model_name
    client = get_client()
    f1_threshold = settings.quality_gate.f1_threshold
    epsilon = settings.quality_gate.epsilon

    candidate = _get_latest_version(client, model_name)
    if candidate is None:
        logger.error("No candidate model found in registry '%s'", model_name)
        return False

    candidate_f1 = _get_f1(client, candidate.run_id)
    logger.info(
        "Candidate: version %s  (F1=%.4f)", candidate.version, candidate_f1
    )

    # Gate 1: absolute threshold
    if candidate_f1 < f1_threshold:
        logger.warning(
            "REJECTED — candidate F1 %.4f < threshold %.4f",
            candidate_f1, f1_threshold,
        )
        return False

    # Gate 2: must improve over production
    prod = _get_production_version(client, model_name)
    if prod is not None:
        prod_f1 = _get_f1(client, prod.run_id)
        logger.info("Production: version %s  (F1=%.4f)", prod.version, prod_f1)
        if candidate_f1 <= prod_f1 + epsilon:
            logger.warning(
                "REJECTED — candidate F1 %.4f ≤ production F1 %.4f + ε %.4f",
                candidate_f1, prod_f1, epsilon,
            )
            return False
    else:
        logger.info("No current production model — first promotion.")

    # Promote
    client.set_registered_model_alias(model_name, "production", candidate.version)
    logger.info(
        "PROMOTED version %s → production  ✓", candidate.version
    )
    return True


# ── Rollback ──────────────────────────────────────────────────────
def rollback(model_name: str | None = None):
    """Revert production alias to the previous version."""
    model_name = model_name or settings.mlflow.registered_model_name
    client = get_client()

    prod = _get_production_version(client, model_name)
    if prod is None:
        logger.error("No production model to roll back from.")
        sys.exit(1)

    current_ver = int(prod.version)
    if current_ver <= 1:
        logger.error("No prior registered version to roll back to.")
        sys.exit(1)

    prev_ver = str(current_ver - 1)
    client.set_registered_model_alias(model_name, "production", prev_ver)
    logger.info(
        "ROLLBACK: production reverted from v%s → v%s  ✓",
        current_ver, prev_ver,
    )
    return True


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Promotion & Rollback")
    parser.add_argument("--rollback", action="store_true",
                        help="Revert production to previous version")
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    if args.rollback:
        rollback(model_name=args.model_name)
    else:
        promote(model_name=args.model_name)
