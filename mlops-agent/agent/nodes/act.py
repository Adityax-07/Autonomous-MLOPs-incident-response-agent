# mlops-agent/agent/nodes/act.py
"""
act_node — third node in the LangGraph agent.

Routes the decision from reason_node to the correct action function,
catches all exceptions, and writes act results back into AgentState.

Action functions now receive the full AgentState (not just drift_report) so
that trigger_alert can include decision + reasoning in the Slack message,
and trigger_retrain can tag the MLflow run with agent metadata.

On failure:
  - Sets state.error (non-None error triggers the graph retry router)
  - Increments state.retry_count (prevents infinite loops)
  - Does NOT re-raise — the graph router handles recovery

Logging format: [AGENT] Decision=retrain | Confidence=0.92 | Action=promoted|roc_auc=0.9812
"""

import logging
from typing import Callable

from actions.alert import trigger_alert
from actions.retrain import trigger_retrain
from actions.rollback import trigger_rollback
from agent.state import AgentState, Decision
from app.metrics import agent_decisions_total

logger = logging.getLogger(__name__)

# ── Action dispatch table ─────────────────────────────────────────────────────
# Each callable receives the full AgentState so action functions can read
# decision, reasoning, confidence, and drift_report from one place.

ACTION_MAP: dict[str, Callable[[AgentState], str]] = {
    "retrain":  trigger_retrain,
    "rollback": trigger_rollback,
    "alert":    trigger_alert,
    # "ok" is handled before act_node is reached (conditional edge in graph.py)
}


def act_node(state: AgentState) -> AgentState:
    """
    LangGraph node: execute the action chosen by reason_node.

    Args:
        state: AgentState with decision, confidence, reasoning set.

    Returns:
        Partial AgentState with action_taken updated, or error set on failure.
    """
    run_id      = state.get("run_id", "unknown")
    decision    = state.get("decision", "ok")
    confidence  = state.get("confidence", 0.0)
    retry_count = state.get("retry_count", 0)

    logger.info(
        "[AGENT] run_id=%s | Decision=%s | Confidence=%.2f | retry=%d",
        run_id, decision, confidence, retry_count,
    )

    if decision == "ok":
        # Defensive guard — graph should skip act_node for "ok" via conditional edge,
        # but handle it gracefully here too.
        agent_decisions_total.labels(decision="ok").inc()
        logger.info("[AGENT] Decision=ok | Action=skipped")
        return {"action_taken": "no_action_required", "error": None}

    action_fn = ACTION_MAP.get(decision)
    if action_fn is None:
        msg = f"Unknown decision value: '{decision}'"
        logger.error("[AGENT] %s", msg)
        return {
            "error":       msg,
            "retry_count": retry_count + 1,
        }

    try:
        # Pass full state — action functions extract what they need
        status = action_fn(state)
        logger.info(
            "[AGENT] Decision=%s | Confidence=%.2f | Action=%s",
            decision, confidence, status,
        )
        # Prometheus counter — label by decision so Grafana can break down by type
        agent_decisions_total.labels(decision=decision).inc()

        return {
            "action_taken": status,
            "error":        None,
        }

    except Exception as exc:
        msg = f"action_failed: decision={decision} error={exc}"
        logger.exception("[AGENT] %s", msg)
        return {
            "error":        msg,
            "retry_count":  retry_count + 1,
            "action_taken": f"failed:{decision}",
        }
