# mlops-agent/app/metrics.py
"""
Prometheus metrics for the UPI Fraud Detection MLOps stack.

All metrics are module-level singletons. Import them where you need to update:

    from app.metrics import model_drift_score, predictions_total

The /metrics ASGI endpoint is mounted in app/main.py.

NOTE on multiprocess mode:
  When running uvicorn with --workers > 1, set the env var
  PROMETHEUS_MULTIPROC_DIR to a shared tmp directory and install
  prometheus-client >= 0.7.0.  For this project we run a single worker
  (the default), so the default single-process registry is correct.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Model health ──────────────────────────────────────────────────────────────

model_drift_score = Gauge(
    "model_drift_score",
    "Latest Evidently drift score — fraction of feature distributions that shifted "
    "(0.0 = no drift, 1.0 = all features drifted)",
)

model_accuracy = Gauge(
    "model_accuracy",
    "Current production model ROC-AUC measured on the validation slice at training time",
)

model_last_retrain_timestamp = Gauge(
    "model_last_retrain_timestamp_seconds",
    "Unix timestamp of the most recent agent-triggered retrain",
)

model_last_retrain_promoted = Gauge(
    "model_last_retrain_promoted",
    "1 if the last retrain was promoted to production, 0 if it was logged as Staging only",
)

# ── Agent decisions ───────────────────────────────────────────────────────────

agent_decisions_total = Counter(
    "agent_decisions_total",
    "Total LangGraph agent decisions, labelled by the chosen action",
    labelnames=["decision"],        # retrain | rollback | alert | ok
)

agent_retrains_total = Counter(
    "agent_retrains_total",
    "Total number of model retrains triggered by the agent "
    "(regardless of whether the new model was promoted)",
)

# ── Prediction metrics ────────────────────────────────────────────────────────

prediction_latency_ms = Histogram(
    "prediction_latency_ms",
    "End-to-end inference latency per /predict request in milliseconds",
    # Fine-grained low-latency buckets: XGBoost on 8 features is typically < 5ms
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0],
)

predictions_total = Counter(
    "predictions_total",
    "Total predictions served by the FastAPI server, labelled by outcome",
    labelnames=["outcome"],         # fraud | legit
)

# ── Drift monitoring ──────────────────────────────────────────────────────────

drift_check_duration_ms = Histogram(
    "drift_check_duration_ms",
    "Wall-clock time to run the full Evidently DataDrift + DataQuality report in ms",
    # Evidently takes 200ms–5s depending on dataset size
    buckets=[100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000],
)
