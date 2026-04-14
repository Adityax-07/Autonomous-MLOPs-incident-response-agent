# mlops-agent/actions/retrain.py
"""
Retrain action — Phase 4 implementation.

Workflow:
  1. Load reference.csv + any labelled rows from data/predictions.log
  2. Retrain XGBoost using the same hyperparameters as app/train.py
  3. Log metrics, params, and feature importances to MLflow
  4. Compare new model ROC-AUC against current production ROC-AUC:
       new > current + PROMOTION_DELTA  → overwrite model.joblib, reload server
       otherwise                        → log as "Staging" only, do NOT promote
  5. Return a status string (stored as action_taken in AgentState)

MLflow tracking URI defaults to ./mlruns (local filesystem).
Set MLFLOW_TRACKING_URI env var to point at a remote server.

FASTAPI_RELOAD_URL defaults to http://localhost:8000/reload.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from agent.state import AgentState
from app.metrics import (
    agent_retrains_total,
    model_accuracy,
    model_last_retrain_promoted,
    model_last_retrain_timestamp,
)

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"

FEATURE_COLUMNS: list[str] = [
    "amount", "hour", "merchant_cat", "device_type",
    "sender_age_days", "receiver_age_days", "txn_count_1h", "same_device",
]
LABEL_COLUMN      = "is_fraud"
DECISION_THRESHOLD = 0.40

# New model must beat current by this much (absolute ROC-AUC) to be promoted
PROMOTION_DELTA = 0.01
# Cap on how many prediction-log rows to merge in (avoids unbounded data growth)
MAX_PROD_ROWS   = 5_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_training_data() -> pd.DataFrame:
    """
    Combine reference.csv with recent labelled rows from data/predictions.log.

    predictions.log rows are only useful when they contain the is_fraud ground-truth
    label (e.g. written after human review). Rows without the label are silently
    dropped. Caps production contribution at MAX_PROD_ROWS.

    Returns:
        Combined DataFrame with FEATURE_COLUMNS + LABEL_COLUMN.

    Raises:
        FileNotFoundError: if reference.csv does not exist.
    """
    ref_path  = DATA_DIR / "reference.csv"
    pred_path = DATA_DIR / "predictions.log"

    if not ref_path.exists():
        raise FileNotFoundError(
            f"reference.csv not found at {ref_path}. "
            "Run: python data/generate_data.py"
        )

    ref_df = pd.read_csv(ref_path)
    logger.info("[RETRAIN] Loaded reference.csv: %d rows", len(ref_df))

    if pred_path.exists():
        records: list[dict] = []
        with pred_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if records:
            prod_df   = pd.DataFrame(records).tail(MAX_PROD_ROWS)
            needed    = FEATURE_COLUMNS + [LABEL_COLUMN]
            available = [c for c in needed if c in prod_df.columns]
            if LABEL_COLUMN in available:
                prod_df = prod_df[available].dropna()
                if len(prod_df) > 0:
                    logger.info(
                        "[RETRAIN] Merged %d labelled rows from predictions.log", len(prod_df)
                    )
                    ref_df = pd.concat([ref_df, prod_df], ignore_index=True)

    return ref_df


def _current_roc_auc() -> float:
    """
    Read the current production model's ROC-AUC from metadata.json.
    Returns 0.0 if the file is missing or unreadable (treats any model as better).
    """
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        return 0.0
    try:
        meta = json.loads(meta_path.read_text())
        return float(meta.get("metrics", {}).get("roc_auc", 0.0))
    except Exception:
        return 0.0


def _reload_server() -> bool:
    """
    POST to the FastAPI /reload endpoint to hot-swap the in-memory model.

    Returns True on HTTP 200, False on any connection/HTTP error.
    A False return is non-fatal — the new model.joblib is already on disk and
    will be picked up on the next server restart.
    """
    reload_url = os.environ.get("FASTAPI_RELOAD_URL", "http://localhost:8000/reload")
    try:
        resp = requests.post(reload_url, timeout=10)
        if resp.status_code == 200:
            logger.info("[RETRAIN] FastAPI server hot-reloaded successfully.")
            return True
        logger.warning("[RETRAIN] /reload returned HTTP %d", resp.status_code)
        return False
    except requests.exceptions.ConnectionError:
        logger.warning(
            "[RETRAIN] FastAPI server unreachable at %s — new model will load on next restart.",
            reload_url,
        )
        return False
    except Exception as exc:
        logger.warning("[RETRAIN] Server reload request failed: %s", exc)
        return False


def _mlflow_transition(
    mlflow_run_id: str,
    target_stage: str,
    archive_existing: bool,
) -> None:
    """
    Transition the MLflow model version (registered in this run) to target_stage.
    Silently logs warnings on failure — MLflow tracking is non-critical.
    """
    try:
        import mlflow
        from mlflow import MlflowClient

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
        client       = MlflowClient(tracking_uri=tracking_uri)

        # The version we just registered sits in "None" stage right after log_model
        versions = client.get_latest_versions("fraud_model", stages=["None"])
        if not versions:
            logger.warning("[RETRAIN] No 'None'-stage version found to transition.")
            return

        latest = max(versions, key=lambda v: int(v.version))
        client.transition_model_version_stage(
            name="fraud_model",
            version=latest.version,
            stage=target_stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(
            "[RETRAIN] MLflow fraud_model v%s -> %s", latest.version, target_stage
        )
    except Exception as exc:
        logger.warning("[RETRAIN] MLflow stage transition failed: %s", exc)


# ── Main action function ──────────────────────────────────────────────────────

def trigger_retrain(state: AgentState) -> str:
    """
    Retrain XGBoost on fresh data, log to MLflow, promote if metrics improve.

    Args:
        state: Full AgentState — reads drift_report, run_id, confidence.

    Returns:
        Status string stored as AgentState.action_taken:
          "promoted|roc_auc=0.XXXX|mlflow_run_id=..."   — new model is live
          "degraded|roc_auc=0.XXXX|current=0.XXXX"      — model logged as Staging only
          "retrain_failed: <reason>"                     — data or training error
    """
    drift_report = state.get("drift_report", {})
    drift_score  = float(drift_report.get("drift_score", 0.0))
    run_id_agent = state.get("run_id", "unknown")

    logger.info(
        "[RETRAIN] Starting | agent_run_id=%s | drift_score=%.3f",
        run_id_agent, drift_score,
    )

    # ── 1. Load combined training data ────────────────────────────────────────
    try:
        df = _load_training_data()
    except FileNotFoundError as exc:
        logger.error("[RETRAIN] %s", exc)
        return f"retrain_failed: {exc}"

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMN].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    n_neg            = int((y_train == 0).sum())
    n_pos            = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    logger.info(
        "[RETRAIN] Data | train=%d val=%d | fraud_rate=%.2f%% | spw=%.2f",
        len(X_train), len(X_val), y.mean() * 100, scale_pos_weight,
    )

    # ── 2. Build and fit pipeline ─────────────────────────────────────────────
    from app.train import build_pipeline
    pipeline = build_pipeline(scale_pos_weight)

    # Fit scaler separately so XGBoost early-stopping sees scaled eval_set
    pipeline.named_steps["scaler"].fit(X_train)
    X_train_sc = pipeline.named_steps["scaler"].transform(X_train)
    X_val_sc   = pipeline.named_steps["scaler"].transform(X_val)

    t0 = time.perf_counter()
    pipeline.named_steps["clf"].fit(
        X_train_sc, y_train,
        eval_set=[(X_val_sc, y_val)],
        verbose=False,
    )
    train_elapsed = time.perf_counter() - t0

    proba_val   = pipeline.named_steps["clf"].predict_proba(X_val_sc)[:, 1]
    new_roc_auc = float(roc_auc_score(y_val, proba_val))
    new_pr_auc  = float(average_precision_score(y_val, proba_val))

    logger.info(
        "[RETRAIN] Training complete | elapsed=%.1fs | roc_auc=%.4f | pr_auc=%.4f",
        train_elapsed, new_roc_auc, new_pr_auc,
    )

    # ── 3. MLflow experiment tracking ─────────────────────────────────────────
    mlflow_run_id: Optional[str] = None
    try:
        import mlflow
        import mlflow.sklearn

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("upi_fraud_retrain")

        feature_importances: dict[str, float] = dict(zip(
            FEATURE_COLUMNS,
            pipeline.named_steps["clf"].feature_importances_.tolist(),
        ))

        with mlflow.start_run(
            tags={
                "triggered_by": "agent",
                "drift_score":  str(round(drift_score, 4)),
                "agent_run_id": run_id_agent,
                "drifted_cols": str(drift_report.get("drifted_columns", [])),
            }
        ) as run:
            mlflow_run_id = run.info.run_id

            mlflow.log_params({
                "n_estimators":       pipeline.named_steps["clf"].n_estimators,
                "max_depth":          6,
                "learning_rate":      0.05,
                "scale_pos_weight":   round(scale_pos_weight, 2),
                "train_rows":         int(len(X_train)),
                "val_rows":           int(len(X_val)),
                "decision_threshold": DECISION_THRESHOLD,
            })
            mlflow.log_metrics({
                "roc_auc":       new_roc_auc,
                "pr_auc":        new_pr_auc,
                "train_elapsed": round(train_elapsed, 2),
                "drift_score":   drift_score,
            })
            for feat, imp in feature_importances.items():
                mlflow.log_metric(f"importance_{feat}", imp)

            # Register model in MLflow Model Registry
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="pipeline",
                registered_model_name="fraud_model",
            )

        logger.info("[RETRAIN] MLflow run logged | run_id=%s", mlflow_run_id)

    except ImportError:
        logger.warning("[RETRAIN] mlflow not installed — skipping experiment tracking.")
    except Exception as exc:
        logger.warning("[RETRAIN] MLflow logging failed (%s) — continuing without tracking.", exc)

    # ── 4. Promotion decision ─────────────────────────────────────────────────
    current_roc_auc = _current_roc_auc()
    should_promote  = new_roc_auc > current_roc_auc + PROMOTION_DELTA

    if should_promote:
        logger.info(
            "[RETRAIN] Promoting | new=%.4f > current=%.4f + delta=%.2f",
            new_roc_auc, current_roc_auc, PROMOTION_DELTA,
        )

        # Back up current model before overwriting (enables local rollback)
        current_model = MODELS_DIR / "model.joblib"
        if current_model.exists():
            shutil.copy2(current_model, MODELS_DIR / "model_backup.joblib")
            logger.info("[RETRAIN] Backed up current model to model_backup.joblib")

        # Overwrite production artefacts
        joblib.dump(pipeline, current_model)
        metadata = {
            "feature_columns":    FEATURE_COLUMNS,
            "label_column":       LABEL_COLUMN,
            "decision_threshold": DECISION_THRESHOLD,
            "n_estimators":       int(pipeline.named_steps["clf"].n_estimators),
            "train_rows":         int(len(X_train)),
            "val_rows":           int(len(X_val)),
            "metrics":            {"roc_auc": round(new_roc_auc, 4), "pr_auc": round(new_pr_auc, 4)},
            "trained_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mlflow_run_id":      mlflow_run_id,
            "triggered_by":       "agent",
            "drift_score":        round(drift_score, 4),
        }
        (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info("[RETRAIN] model.joblib and metadata.json overwritten")

        # Promote MLflow version to Production (archive all previous Production versions)
        if mlflow_run_id:
            _mlflow_transition(mlflow_run_id, "Production", archive_existing=True)

        # Hot-reload the running FastAPI server
        _reload_server()

        # Prometheus metrics — update after successful promotion
        agent_retrains_total.inc()
        model_accuracy.set(new_roc_auc)
        model_last_retrain_timestamp.set(time.time())
        model_last_retrain_promoted.set(1)

        return f"promoted|roc_auc={new_roc_auc:.4f}|mlflow_run_id={mlflow_run_id}"

    else:
        logger.info(
            "[RETRAIN] Not promoted | new=%.4f <= current=%.4f + delta=%.2f — logged as Staging",
            new_roc_auc, current_roc_auc, PROMOTION_DELTA,
        )
        if mlflow_run_id:
            _mlflow_transition(mlflow_run_id, "Staging", archive_existing=False)

        # Still count the retrain attempt; mark as not promoted
        agent_retrains_total.inc()
        model_last_retrain_timestamp.set(time.time())
        model_last_retrain_promoted.set(0)

        return (
            f"degraded|roc_auc={new_roc_auc:.4f}|"
            f"current={current_roc_auc:.4f}|"
            f"mlflow_run_id={mlflow_run_id}"
        )
