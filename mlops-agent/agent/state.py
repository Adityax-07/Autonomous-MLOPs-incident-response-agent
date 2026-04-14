# mlops-agent/agent/state.py
"""
Shared state schema for the MLOps Incident Response Agent.

AgentState flows through every LangGraph node unchanged except for the
fields that node is responsible for updating. Each node returns a PARTIAL
dict — only the keys it touched — and LangGraph merges them into the
running state automatically.

Field ownership:
  monitor_node → drift_report, error, run_id
  reason_node  → decision, confidence, reasoning, retry_count
  act_node     → action_taken, error

Why TypedDict (not dataclass/Pydantic)?
  LangGraph requires the state type to be a TypedDict so it can do
  shallow-merge semantics. Pydantic models work too, but TypedDict has
  zero runtime overhead and no import cost.
"""

import uuid
from typing import Literal, Optional
from typing_extensions import TypedDict

# ── Decision type ─────────────────────────────────────────────────────────────

Decision = Literal["retrain", "rollback", "alert", "ok"]

# Every valid decision value — used for validation in reason_node
VALID_DECISIONS: frozenset[str] = frozenset({"retrain", "rollback", "alert", "ok"})

# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    Mutable state passed between LangGraph nodes.

    All fields are optional (total=False) so nodes can return partial updates
    without specifying every field. LangGraph merges partial dicts correctly.

    Fields
    ------
    drift_report : dict
        Structured output from monitor/drift_check.py:generate_drift_report().
        Contains drift_detected, drift_score, drifted_columns, recommendation,
        feature_details, quality_issues, timestamp, report_html_path.

    decision : Decision
        The action chosen by reason_node: "retrain" | "rollback" | "alert" | "ok".

    confidence : float
        How confident the reasoning engine is in the decision (0.0 – 1.0).
        Rule-based engine uses fixed values; LLM engine extracts from response.

    reasoning : str
        Plain-English explanation of WHY this decision was made.
        Logged to MLflow and printed in the runner summary.

    action_taken : str
        Short description of what act_node actually did (e.g. "retrain_triggered",
        "slack_alert_sent", "rollback_to_v2"). Set AFTER the action completes.

    error : str | None
        Set by any node when a recoverable error occurs. Non-None triggers
        the retry edge from act_node back to reason_node (max 1 retry).
        After max retries, the graph terminates with this error preserved.

    run_id : str
        UUID4 assigned at graph entry. Used as the MLflow run name and
        for correlating logs across all nodes in a single agent invocation.

    retry_count : int
        Incremented by act_node before routing back to reason_node.
        Prevents infinite retry loops — max is enforced in the graph router.
    """

    drift_report:  dict
    decision:      Decision
    confidence:    float
    reasoning:     str
    action_taken:  str
    error:         Optional[str]
    run_id:        str
    retry_count:   int


def initial_state() -> AgentState:
    """
    Return a fresh AgentState with safe defaults.

    Call this to seed graph.invoke() — avoids KeyError in nodes that
    read fields before they've been set by upstream nodes.
    """
    return AgentState(
        drift_report={},
        decision="ok",
        confidence=0.0,
        reasoning="",
        action_taken="",
        error=None,
        run_id=str(uuid.uuid4()),
        retry_count=0,
    )
