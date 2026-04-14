# mlops-agent/app/model.py
"""
Model loader singleton.

Loads the joblib pipeline and metadata.json once at startup,
then exposes a predict() function used by the FastAPI routes.

Using a module-level singleton (rather than reloading per request)
keeps inference latency low and memory use constant.
"""

import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH    = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ── Singleton state ───────────────────────────────────────────────────────────

_pipeline = None        # sklearn Pipeline
_metadata: dict = {}    # contents of metadata.json

# Reentrant lock: prevents two concurrent hot-reloads from racing to swap
# _pipeline mid-request.  RLock (not Lock) allows the same thread to re-acquire
# (e.g. during retrain which calls load_model() from within an already-locked scope).
_reload_lock = threading.RLock()


def load_model() -> None:
    """
    Load (or reload) the model from disk into module-level state.
    Called once at FastAPI startup; can be called again after a retrain.

    Thread-safe: acquires _reload_lock before swapping globals so a concurrent
    predict() never reads a partially-updated pipeline/metadata pair.

    Raises:
        FileNotFoundError: if model artefact does not exist on disk.
        Exception:         propagated from joblib.load on corrupt files.
    """
    global _pipeline, _metadata

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artefact not found at {MODEL_PATH}. "
            "Run `python app/train.py` first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}. "
            "Run `python app/train.py` first."
        )

    with _reload_lock:
        try:
            # Load into locals first; only swap globals on full success
            new_pipeline = joblib.load(MODEL_PATH)
            new_metadata = json.loads(METADATA_PATH.read_text())
            _pipeline = new_pipeline
            _metadata = new_metadata
            logger.info("Model loaded  | version=%s | threshold=%.2f",
                        _metadata.get("trained_at", "unknown"),
                        _metadata.get("decision_threshold", 0.5))

            # Sync Prometheus gauge so Grafana shows accuracy immediately after (re)load
            roc = _metadata.get("metrics", {}).get("roc_auc")
            if roc is not None:
                from app.metrics import model_accuracy
                model_accuracy.set(float(roc))
        except Exception as exc:
            logger.exception("Failed to load model artefact: %s", exc)
            raise


def is_loaded() -> bool:
    """Return True if a model is currently loaded in memory."""
    return _pipeline is not None


def get_version() -> str:
    """Return a human-readable model version string (ISO timestamp from training)."""
    return _metadata.get("trained_at", "unknown")


def get_metadata() -> dict:
    """Return a copy of the full metadata dictionary."""
    return dict(_metadata)


def predict(features: dict) -> dict:
    """
    Run inference on a single transaction.

    Args:
        features: Dict mapping feature name → value (must match FEATURE_COLUMNS).

    Returns:
        Dict with keys:
            transaction_id    – UUID assigned to this prediction
            fraud_probability – float in [0, 1]
            is_fraud          – bool
            decision_threshold– float
            model_version     – str

    Raises:
        RuntimeError:  if model has not been loaded.
        KeyError:      if a required feature is missing from `features`.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Model not loaded. Call load_model() before predict()."
        )

    feature_columns: list[str] = _metadata["feature_columns"]
    threshold: float           = _metadata["decision_threshold"]

    try:
        row = np.array(
            [[features[col] for col in feature_columns]],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise KeyError(f"Missing feature in input: {exc}") from exc

    try:
        # Pipeline: scaler → XGBClassifier
        # predict_proba returns shape (1, 2); column 1 = P(fraud)
        proba: float = float(_pipeline.predict_proba(row)[0, 1])
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise

    return {
        "transaction_id":    str(uuid.uuid4()),
        "fraud_probability": round(proba, 6),
        "is_fraud":          proba >= threshold,
        "decision_threshold": threshold,
        "model_version":     get_version(),
    }
