# mlops-agent/agent/graph.py
"""
LangGraph StateGraph assembly for the MLOps Incident Response Agent.

Graph topology:
                     ┌─────────────┐
              START  │ monitor_node │  (load + validate drift report)
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ reason_node │  (rule-based or LLM decision)
                     └──────┬──────┘
                            │
              ┌─────────────▼──────────────┐
              │   conditional: route_act   │
              └──────┬──────────────┬──────┘
                   "ok"         "retrain/rollback/alert"
                     │               │
                    END       ┌──────▼──────┐
                              │   act_node  │
                              └──────┬──────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │  conditional: route_after_act   │
                    └──────┬──────────────────┬───────┘
                     no error              error + retry_count < 1
                           │                       │
                          END              ┌───────▼───────┐
                                           │  reason_node  │  (retry, downgraded)
                                           └───────┬───────┘
                                                   │
                                               act_node → END

Retry policy:
  - If act_node fails AND retry_count < MAX_RETRIES, route back to reason_node.
  - reason_node detects retry_count > 0 and downgrades the decision one level.
  - After MAX_RETRIES failures the graph terminates with state.error set.

Exports:
  compiled_graph  — the LangGraph CompiledGraph, ready for .invoke() / .stream()
  run_agent()     — convenience wrapper that seeds state and returns final AgentState
"""

import logging
import uuid
from typing import Literal

from langgraph.graph import END, StateGraph

from agent.nodes.act import act_node
from agent.nodes.monitor import monitor_node
from agent.nodes.reason import reason_node
from agent.state import AgentState, Decision, initial_state

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 1   # allow one retry before giving up


# ── Edge routers ──────────────────────────────────────────────────────────────

def _route_after_monitor(
    state: AgentState,
) -> Literal["reason_node", "__end__"]:
    """
    Conditional edge after monitor_node.

    Halts immediately if monitor_node set an error (stale report, missing file,
    bad schema). This prevents reason_node from running on invalid data.
    """
    error = state.get("error")
    if error:
        logger.warning("route_after_monitor: monitor error='%s' -> END", error)
        return END
    return "reason_node"


def _route_after_reason(
    state: AgentState,
) -> Literal["act_node", "__end__"]:
    """
    Conditional edge after reason_node.

    Routes to act_node for any actionable decision.
    Short-circuits to END for "ok" (nothing to do) or if monitor_node
    set an error (no valid report to act on).
    """
    error    = state.get("error")
    decision = state.get("decision", "ok")

    if error:
        logger.info("route_after_reason: error=%s -> END", error)
        return END

    if decision == "ok":
        logger.info("route_after_reason: decision=ok -> END (no action needed)")
        return END

    logger.info("route_after_reason: decision=%s -> act_node", decision)
    return "act_node"


def _route_after_act(
    state: AgentState,
) -> Literal["reason_node", "__end__"]:
    """
    Conditional edge after act_node.

    If act_node set an error AND retry budget remains: route back to
    reason_node for a downgraded retry attempt.
    Otherwise terminate.
    """
    error       = state.get("error")
    retry_count = state.get("retry_count", 0)

    if error and retry_count <= MAX_RETRIES:
        logger.warning(
            "route_after_act: action failed (error=%s), retry %d/%d -> reason_node",
            error, retry_count, MAX_RETRIES,
        )
        return "reason_node"

    if error:
        logger.error(
            "route_after_act: action failed after max retries -> END (error=%s)", error
        )
    return END


# ── Graph construction ────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """Construct and return the uncompiled StateGraph."""
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("monitor_node", monitor_node)
    graph.add_node("reason_node",  reason_node)
    graph.add_node("act_node",     act_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("monitor_node")

    # ── monitor_node → conditional (halt on error, else reason) ──────────────
    graph.add_conditional_edges(
        "monitor_node",
        _route_after_monitor,
        {
            "reason_node": "reason_node",
            END: END,
        },
    )

    # ── Conditional edges ─────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "reason_node",
        _route_after_reason,
        {
            "act_node": "act_node",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "act_node",
        _route_after_act,
        {
            "reason_node": "reason_node",
            END: END,
        },
    )

    return graph


# ── Compile once at import time ───────────────────────────────────────────────

compiled_graph = _build_graph().compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(seed: dict | None = None) -> AgentState:
    """
    Run the full agent graph and return the final AgentState.

    Args:
        seed: Optional dict to merge into the initial state before invoking.
              Useful for injecting a pre-loaded drift_report in tests.

    Returns:
        Final AgentState after all nodes have executed.
    """
    state = initial_state()
    if seed:
        state.update(seed)   # type: ignore[typeddict-item]

    state["run_id"] = str(uuid.uuid4())

    logger.info("Agent run started | run_id=%s | mode=%s",
                state["run_id"],
                __import__("os").environ.get("AGENT_MODE", "rule_based"))

    final: AgentState = compiled_graph.invoke(state)

    logger.info(
        "Agent run complete | run_id=%s | decision=%s | action=%s | error=%s",
        final.get("run_id"),
        final.get("decision"),
        final.get("action_taken"),
        final.get("error"),
    )
    return final
