# mlops-agent/app/main.py
"""
FastAPI model server — UPI Fraud Detection

Endpoints:
  GET  /health          → liveness / readiness probe
  GET  /model/info      → model metadata (version, features, threshold)
  POST /predict         → single-transaction fraud scoring
  POST /predict/batch   → batch scoring (up to 500 rows)

Design decisions:
  • Model is loaded once at startup via lifespan context manager (FastAPI ≥ 0.93)
  • All inference errors return structured JSON (never a raw 500 traceback)
  • Request validation is handled by Pydantic v2 schemas
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app import model as model_store
from app.metrics import prediction_latency_ms, predictions_total
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Predictions log (read by monitor/scheduler.py) ────────────────────────────
_PREDICTIONS_LOG = Path(__file__).parent.parent / "data" / "predictions.log"
_PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)

def _log_prediction(features: dict, result: dict) -> None:
    """Append a single prediction to predictions.log (JSONL format)."""
    try:
        record = {**features, **result, "logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with _PREDICTIONS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed to write to predictions.log: %s", exc)

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artefact on startup; release resources on shutdown."""
    logger.info("Starting UPI Fraud Detection API …")
    try:
        model_store.load_model()
    except FileNotFoundError as exc:
        logger.critical(str(exc))
        # Allow the server to start so /health returns 503 (useful in k8s)
    except Exception as exc:
        logger.critical("Unexpected error loading model: %s", exc)

    yield  # server is live

    logger.info("Shutting down — releasing resources.")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="UPI Fraud Detection API",
    description=(
        "Autonomous MLOps project — XGBoost model served via FastAPI. "
        "Part of the MLOps Incident Response Agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus /metrics endpoint.
# prometheus_client.make_asgi_app() returns a lightweight ASGI app that
# serialises all registered metrics on GET /metrics.
# This sub-app is excluded from FastAPI's OpenAPI schema automatically.
app.mount("/metrics", make_asgi_app())

# ── Middleware: request timing ────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    # Record inference latency for predict endpoints only.
    # /predict/batch latency covers the whole batch (not per-row) — acceptable for now.
    if request.url.path in ("/predict", "/predict/batch"):
        prediction_latency_ms.observe(elapsed_ms)

    return response

# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check server logs."},
    )

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness probe",
    tags=["Ops"],
)
async def health() -> HealthResponse:
    """
    Returns HTTP 200 when the model is loaded and ready to serve traffic.
    Returns HTTP 503 when the model artefact could not be loaded at startup.
    """
    loaded = model_store.is_loaded()
    if not loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Server is not ready.",
        )
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_version=model_store.get_version(),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Model metadata",
    tags=["Ops"],
)
async def model_info() -> ModelInfoResponse:
    """Returns feature list, decision threshold, training timestamp and eval metrics."""
    if not model_store.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    meta = model_store.get_metadata()
    return ModelInfoResponse(
        model_version=meta.get("trained_at", "unknown"),
        feature_columns=meta.get("feature_columns", []),
        decision_threshold=meta.get("decision_threshold", 0.5),
        trained_at=meta.get("trained_at", "unknown"),
        metrics=meta.get("metrics", {}),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraud for a single UPI transaction",
    tags=["Inference"],
)
async def predict(txn: TransactionRequest) -> PredictionResponse:
    """
    Accepts a single UPI transaction feature vector and returns:
    - `fraud_probability` — raw model score in [0, 1]
    - `is_fraud`          — binary decision at the configured threshold
    - `model_version`     — which model artefact was used
    """
    if not model_store.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    try:
        result = model_store.predict(txn.model_dump())
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature mismatch: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference error. Check server logs.",
        ) from exc

    logger.info(
        "predict | txn_id=%s | prob=%.4f | is_fraud=%s",
        result["transaction_id"],
        result["fraud_probability"],
        result["is_fraud"],
    )
    _log_prediction(txn.model_dump(), result)

    # Prometheus counter — labelled by outcome so we can plot fraud vs legit volume
    outcome = "fraud" if result["is_fraud"] else "legit"
    predictions_total.labels(outcome=outcome).inc()

    return PredictionResponse(**result)


@app.post(
    "/reload",
    summary="Hot-reload the model from disk without server restart",
    tags=["Ops"],
)
async def reload_model() -> dict:
    """
    Swaps the in-memory model with whatever is currently on disk.

    Thread-safe: model_store.load_model() holds _reload_lock while swapping globals,
    so concurrent /predict requests always see a complete (old or new) pipeline —
    never a half-swapped state.

    The blocking I/O (joblib.load) runs in a thread-pool executor so the async
    event loop is not stalled during the file read.

    Returns HTTP 200 on success, 503 if no model file exists, 500 on load error.
    """
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, model_store.load_model)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Model reload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {exc}",
        ) from exc

    return {
        "status":  "reloaded",
        "version": model_store.get_version(),
    }


@app.post(
    "/predict/batch",
    summary="Predict fraud for a batch of UPI transactions (max 500)",
    tags=["Inference"],
)
async def predict_batch(transactions: list[TransactionRequest]) -> list[PredictionResponse]:
    """
    Accepts up to 500 transactions and returns predictions in the same order.
    Uses vectorised numpy operations for efficiency.
    """
    if not model_store.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    if len(transactions) == 0:
        return []

    if len(transactions) > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(transactions)} exceeds maximum of 500.",
        )

    try:
        results = [model_store.predict(txn.model_dump()) for txn in transactions]
    except Exception as exc:
        logger.exception("Batch inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch inference error.",
        ) from exc

    return [PredictionResponse(**r) for r in results]
