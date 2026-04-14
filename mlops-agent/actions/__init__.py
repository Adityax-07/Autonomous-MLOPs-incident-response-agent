# mlops-agent/actions/__init__.py
"""
Actions package for the MLOps Incident Response Agent.

Primary exports:
  execute_action(decision, state) -> dict
      Unified entry point that wraps each action function with:
        - MLflow run tagging (links action event to drift metadata)
        - Wall-clock timing
        - Structured result logging

Individual action functions are also importable directly:
  from actions.retrain  import trigger_retrain
  from actions.rollback import trigger_rollback
  from actions.alert    import trigger_alert

Note on import ordering:
  rollback.py imports trigger_alert lazily (inside the function body) to avoid
  a circular import caused by __init__.py importing all three at module load time.
"""

import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

from agent.state import AgentState
from actions.alert    import trigger_alert
from actions.retrain  import trigger_retrain
from actions.rollback import trigger_rollback

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

# ── Action dispatch table ─────────────────────────────────────────────────────

_ACTION_MAP: dict[str, Callable[[AgentState], str]] = {
    "retrain":  trigger_retrain,
    "rollback": trigger_rollback,
    "alert":    trigger_alert,
}


# ── Public API ────────────────────────────────────────────────────────────────

def execute_action(decision: str, state: AgentState) -> dict:
    """
    Execute the action corresponding to `decision` with cross-cutting concerns:

      1. Validates decision string.
      2. Opens a lightweight MLflow tracking run tagged with drift metadata.
         (If mlflow is not installed, the action still executes — tracking is optional.)
      3. Calls the action function, timing it with time.perf_counter().
      4. Logs the result to the MLflow run and closes it.
      5. Returns a structured result dict.

    This function is an *alternative* entry point to act_node. act_node calls
    the action functions directly for tighter LangGraph integration. Use
    execute_action() from scripts or tests where you want the extra observability
    layer without running the full graph.

    Args:
        decision: One of "retrain", "rollback", "alert".
        state:    Full AgentState containing drift_report, run_id, etc.

    Returns:
        Dict with keys:
          status        (str)            — raw return value from the action function
          elapsed_s     (float)          — wall-clock seconds the action took
          decision      (str)            — the decision that was executed
          mlflow_run_id (str | None)     — MLflow run ID, or None if tracking unavailable
          error         (str | None)     — error message on unexpected failure, else None
    """
    action_fn = _ACTION_MAP.get(decision)
    if action_fn is None:
        msg = f"Unknown decision '{decision}' — valid: {sorted(_ACTION_MAP)}"
        logger.error("[execute_action] %s", msg)
        return {
            "status":        "error",
            "elapsed_s":     0.0,
            "decision":      decision,
            "mlflow_run_id": None,
            "error":         msg,
        }

    drift_score = float(state.get("drift_report", {}).get("drift_score", 0.0))
    run_id      = state.get("run_id", "unknown")

    logger.info(
        "[execute_action] Starting | decision=%s | drift_score=%.3f | agent_run_id=%s",
        decision, drift_score, run_id,
    )

    mlflow_run_id: Optional[str] = None
    status:  str   = ""
    elapsed: float = 0.0
    error:   Optional[str] = None

    # ── MLflow tracking wrapper ───────────────────────────────────────────────
    try:
        import mlflow

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("upi_fraud_agent_actions")

        tags = {
            "decision":      decision,
            "drift_score":   str(round(drift_score, 4)),
            "agent_run_id":  run_id,
            "confidence":    str(round(float(state.get("confidence", 0.0)), 4)),
            "drifted_cols":  str(state.get("drift_report", {}).get("drifted_columns", [])),
        }

        with mlflow.start_run(tags=tags) as mlflow_run:
            mlflow_run_id = mlflow_run.info.run_id
            mlflow.log_metric("drift_score", drift_score)

            t0     = time.perf_counter()
            status = action_fn(state)
            elapsed = time.perf_counter() - t0

            mlflow.log_metric("action_elapsed_s", round(elapsed, 3))
            mlflow.set_tag("action_status", status[:250])   # Slack URLs can be long

    except ImportError:
        logger.warning("[execute_action] mlflow not installed — running without experiment tracking.")
        t0     = time.perf_counter()
        status = action_fn(state)
        elapsed = time.perf_counter() - t0

    except Exception as exc:
        # MLflow infrastructure failure should not block the action
        logger.warning("[execute_action] MLflow tracking failed (%s) — executing action anyway.", exc)
        t0     = time.perf_counter()
        status = action_fn(state)
        elapsed = time.perf_counter() - t0

    result = {
        "status":        status,
        "elapsed_s":     round(elapsed, 3),
        "decision":      decision,
        "mlflow_run_id": mlflow_run_id,
        "error":         error,
    }

    logger.info(
        "[execute_action] Complete | decision=%s | status=%s | elapsed=%.2fs | mlflow_run=%s",
        decision, status[:60], elapsed, mlflow_run_id,
    )
    return result
