# mlops-agent/actions/rollback.py
"""
Rollback action — Phase 4 implementation.

Fallback chain (attempts each in order, stops at first success):

  1. MLflow Model Registry
       Find the most recent "Archived" version of "fraud_model".
       Archived versions are former Production models that were superseded by a retrain.
       Download the pipeline artifact, overwrite model.joblib + metadata.json,
       swap the MLflow stages (Archived->Production, current Production->Archived),
       and hot-reload the FastAPI server.

  2. Local backup file  (models/model_backup.joblib)
       Written by trigger_retrain() before every promotion.
       No MLflow needed — just copy the file and reload.

  3. Escalate to alert
       If neither option exists, the system cannot self-heal.
       Call trigger_alert() so a human is notified.

FASTAPI_RELOAD_URL  env var: override default http://localhost:8000/reload
MLFLOW_TRACKING_URI env var: override default ./mlruns
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path

import requests

from agent.state import AgentState

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reload_server() -> bool:
    """POST /reload to the FastAPI server. Returns True on HTTP 200."""
    reload_url = os.environ.get("FASTAPI_RELOAD_URL", "http://localhost:8000/reload")
    try:
        resp = requests.post(reload_url, timeout=10)
        if resp.status_code == 200:
            logger.info("[ROLLBACK] FastAPI server hot-reloaded successfully.")
            return True
        logger.warning("[ROLLBACK] /reload returned HTTP %d", resp.status_code)
        return False
    except requests.exceptions.ConnectionError:
        logger.warning(
            "[ROLLBACK] FastAPI server unreachable at %s — new model loads on next restart.",
            reload_url,
        )
        return False
    except Exception as exc:
        logger.warning("[ROLLBACK] Server reload request failed: %s", exc)
        return False


def _write_rollback_metadata(
    source: str,
    restored_version: str,
    previous_version: str,
) -> None:
    """
    Stamp metadata.json with rollback provenance fields.
    Reads existing metadata so feature_columns / metrics are preserved.
    """
    meta_path = MODELS_DIR / "metadata.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    except Exception:
        meta = {}

    meta.update({
        "rolled_back_at":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rolled_back_source":   source,
        "restored_version":     restored_version,
        "previous_version":     previous_version,
        "triggered_by":         "agent_rollback",
    })
    meta_path.write_text(json.dumps(meta, indent=2))


# ── Fallback 1: MLflow registry ───────────────────────────────────────────────

def _rollback_via_mlflow(state: AgentState) -> str:
    """
    Attempt rollback by restoring the most recent Archived model from the
    MLflow Model Registry.

    Raises:
        Exception: on any MLflow connectivity or data error (caller falls through
                   to the next fallback).

    Returns:
        Status string on success.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Find the current Production version
    prod_versions = client.get_latest_versions("fraud_model", stages=["Production"])
    if not prod_versions:
        raise ValueError("No 'Production' version found in MLflow registry.")

    current = max(prod_versions, key=lambda v: int(v.version))
    logger.info("[ROLLBACK] Current Production: fraud_model v%s", current.version)

    # Find the most recent Archived version (= previous Production before last retrain)
    archived = client.get_latest_versions("fraud_model", stages=["Archived"])
    if not archived:
        raise ValueError("No 'Archived' version found in MLflow registry — cannot roll back.")

    target = max(archived, key=lambda v: int(v.version))
    logger.info("[ROLLBACK] Restoring Archived: fraud_model v%s (run_id=%s)",
                target.version, target.run_id)

    # Download the sklearn pipeline artifact
    artifact_uri = f"runs:/{target.run_id}/pipeline"
    pipeline = mlflow.sklearn.load_model(artifact_uri)
    logger.info("[ROLLBACK] Pipeline downloaded from MLflow artifact store.")

    # Back up the current bad model before overwriting
    current_model = MODELS_DIR / "model.joblib"
    if current_model.exists():
        shutil.copy2(current_model, MODELS_DIR / "model_pre_rollback.joblib")

    # Overwrite production artefact
    import joblib
    joblib.dump(pipeline, current_model)
    _write_rollback_metadata(
        source="mlflow_registry",
        restored_version=target.version,
        previous_version=current.version,
    )
    logger.info("[ROLLBACK] model.joblib overwritten with Archived version.")

    # Swap MLflow registry stages:  target Archived -> Production, current Production -> Archived
    client.transition_model_version_stage(
        name="fraud_model",
        version=target.version,
        stage="Production",
    )
    client.transition_model_version_stage(
        name="fraud_model",
        version=current.version,
        stage="Archived",
    )
    logger.info(
        "[ROLLBACK] Registry: v%s->Production, v%s->Archived",
        target.version, current.version,
    )

    _reload_server()

    return (
        f"rolled_back|source=mlflow_registry|"
        f"restored_version={target.version}|"
        f"previous_version={current.version}"
    )


# ── Fallback 2: Local backup file ─────────────────────────────────────────────

def _rollback_via_local_backup() -> str:
    """
    Restore from models/model_backup.joblib written by trigger_retrain().

    Raises:
        FileNotFoundError: if backup file does not exist.

    Returns:
        Status string on success.
    """
    backup = MODELS_DIR / "model_backup.joblib"
    if not backup.exists():
        raise FileNotFoundError(f"No local backup found at {backup}.")

    current_model = MODELS_DIR / "model.joblib"
    if current_model.exists():
        shutil.copy2(current_model, MODELS_DIR / "model_pre_rollback.joblib")

    shutil.copy2(backup, current_model)
    _write_rollback_metadata(
        source="local_backup",
        restored_version="backup",
        previous_version="unknown",
    )
    logger.info("[ROLLBACK] Restored from local backup: %s", backup)

    _reload_server()
    return "rolled_back|source=local_backup"


# ── Main action function ──────────────────────────────────────────────────────

def trigger_rollback(state: AgentState) -> str:
    """
    Roll back to the most recent stable model, trying MLflow then local backup.

    If neither is available, escalates to trigger_alert() so a human is notified.

    Args:
        state: Full AgentState — reads drift_report and run_id for logging.

    Returns:
        Status string stored as AgentState.action_taken.
    """
    drift_report = state.get("drift_report", {})
    drift_score  = float(drift_report.get("drift_score", 0.0))
    run_id_agent = state.get("run_id", "unknown")

    logger.info(
        "[ROLLBACK] Starting | agent_run_id=%s | drift_score=%.3f",
        run_id_agent, drift_score,
    )

    # ── Attempt 1: MLflow Model Registry ─────────────────────────────────────
    try:
        return _rollback_via_mlflow(state)
    except ImportError:
        logger.warning("[ROLLBACK] mlflow not installed — skipping registry rollback.")
    except Exception as exc:
        logger.warning("[ROLLBACK] MLflow rollback failed (%s) — trying local backup.", exc)

    # ── Attempt 2: Local backup file ──────────────────────────────────────────
    try:
        return _rollback_via_local_backup()
    except FileNotFoundError as exc:
        logger.warning("[ROLLBACK] %s — no recovery options left.", exc)

    # ── Attempt 3: Escalate to alert ─────────────────────────────────────────
    logger.error(
        "[ROLLBACK] All rollback options exhausted for agent_run_id=%s. "
        "Escalating to alert action.",
        run_id_agent,
    )
    # Import here (not at top) to avoid circular import during module loading:
    # actions/__init__.py imports from both rollback and alert, so if rollback
    # imported alert at module level, Python would see a partially-loaded package.
    from actions.alert import trigger_alert  # noqa: PLC0415
    trigger_alert(state)
    return "rollback_failed|escalated_to_alert"
