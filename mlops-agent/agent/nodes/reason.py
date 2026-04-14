# mlops-agent/agent/nodes/reason.py
"""
reason_node — second node in the LangGraph agent.

Two versions, toggled via environment variable:

  AGENT_MODE=rule_based  (default)
    Pure Python threshold logic. Works offline, zero latency, fully deterministic.
    Decision is derived from drift_score + drifted_columns.

  AGENT_MODE=llm
    Calls Claude claude-haiku-4-5 via the Anthropic SDK. Passes the full drift
    report as JSON. Expects a structured JSON response back. Falls back to
    rule-based if the API call fails, the response is unparseable, or
    ANTHROPIC_API_KEY is not set.

Both versions return the same AgentState fields:
  decision   — "retrain" | "rollback" | "alert" | "ok"
  confidence — float 0-1
  reasoning  — plain-English explanation string
"""

import json
import logging
import os
from typing import Optional

from agent.state import VALID_DECISIONS, AgentState, Decision

logger = logging.getLogger(__name__)

# ── Thresholds (must match monitor/drift_check.py for consistency) ─────────────
RETRAIN_THRESHOLD  = 0.50
ROLLBACK_THRESHOLD = 0.30
ALERT_THRESHOLD    = 0.10

# High-impact UPI fraud features — drift in these warrants stronger action
HIGH_IMPACT_FEATURES = {"amount", "txn_count_1h", "sender_age_days"}


# ═══════════════════════════════════════════════════════════════════════════════
# Version A — Rule-based reasoning
# ═══════════════════════════════════════════════════════════════════════════════

def _rule_based_reason(drift_report: dict) -> tuple[Decision, float, str]:
    """
    Apply pure-Python threshold logic to produce a decision.

    Escalation path:
      drift_score > 0.50                    → retrain   (majority of features shifted)
      drift_score > 0.30                    → rollback  (significant drift)
      drift_score > 0.10 OR high-impact hit → alert     (minor but noteworthy)
      else                                  → ok

    High-impact override:
      If a critical fraud-signal feature (amount, txn_count_1h, sender_age_days)
      has drifted, escalate one level higher than the raw score would give.
      Rationale: a 10% shift in transaction amount changes the fraud model's
      score distribution far more than a 10% shift in merchant_cat.

    Args:
        drift_report: Dict from generate_drift_report().

    Returns:
        (decision, confidence, reasoning) tuple.
    """
    drift_score:    float     = float(drift_report.get("drift_score", 0.0))
    drifted_cols:   list[str] = drift_report.get("drifted_columns", [])
    quality_issues: list[str] = drift_report.get("quality_issues", [])
    feature_details: dict     = drift_report.get("feature_details", {})

    high_impact_drifted = [c for c in drifted_cols if c in HIGH_IMPACT_FEATURES]
    n_drifted           = len(drifted_cols)
    has_quality_issues  = len(quality_issues) > 0

    # Build reasoning step by step
    reasons: list[str] = [
        f"Drift score is {drift_score:.1%} "
        f"({n_drifted} of {len(feature_details) or 8} features drifted)."
    ]
    if high_impact_drifted:
        reasons.append(
            f"High-impact fraud features drifted: {', '.join(high_impact_drifted)}. "
            "These directly affect fraud score distributions."
        )
    if has_quality_issues:
        reasons.append(f"Data quality issues detected: {'; '.join(quality_issues)}.")

    # Decision logic with high-impact escalation
    if drift_score > RETRAIN_THRESHOLD:
        decision:    Decision = "retrain"
        confidence:  float    = min(0.95, 0.70 + drift_score * 0.5)
        reasons.append(
            f"Drift score {drift_score:.1%} exceeds retrain threshold "
            f"({RETRAIN_THRESHOLD:.0%}). Model distribution is severely stale."
        )

    elif drift_score > ROLLBACK_THRESHOLD or (
        drift_score > ALERT_THRESHOLD and high_impact_drifted
    ):
        # High-impact features override: escalate alert → rollback
        if drift_score <= ROLLBACK_THRESHOLD and high_impact_drifted:
            decision   = "rollback"
            confidence = 0.72
            reasons.append(
                f"High-impact feature override: {high_impact_drifted} drifted at "
                f"drift_score={drift_score:.1%}. Escalating alert → rollback."
            )
        else:
            decision   = "rollback"
            confidence = min(0.88, 0.60 + drift_score * 0.8)
            reasons.append(
                f"Drift score {drift_score:.1%} exceeds rollback threshold "
                f"({ROLLBACK_THRESHOLD:.0%}). Revert to last stable model version."
            )

    elif drift_score > ALERT_THRESHOLD or has_quality_issues:
        decision   = "alert"
        confidence = min(0.80, 0.50 + drift_score * 1.5)
        reasons.append(
            f"Drift score {drift_score:.1%} exceeds alert threshold "
            f"({ALERT_THRESHOLD:.0%}). Monitor closely but do not act yet."
        )

    else:
        decision   = "ok"
        confidence = min(0.99, 0.90 + (1.0 - drift_score) * 0.1)
        reasons.append(
            f"Drift score {drift_score:.1%} is below all thresholds. "
            "Model distribution is healthy."
        )

    reasoning = " ".join(reasons)
    return decision, round(confidence, 4), reasoning


# ═══════════════════════════════════════════════════════════════════════════════
# Version B — LLM-powered reasoning (Claude claude-haiku-4-5)
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are an expert MLOps engineer responsible for a UPI (Unified Payments Interface) \
fraud detection model served in production. You monitor data drift reports and decide \
the appropriate corrective action to maintain model reliability.

Your decisions and their meanings:
- "retrain"  : Significant feature distribution shift — the model must be retrained \
on fresh data immediately. Use when drift_score > 0.50 or critical features shift heavily.
- "rollback" : Moderate drift — the current model is unreliable. Revert to the last \
known-good checkpoint. Use when drift_score > 0.30 or high-impact features drift.
- "alert"    : Minor drift detected — flag for human review but do not take automated \
action yet. Use when drift_score > 0.10.
- "ok"       : No significant drift — model is healthy, no action needed.

High-impact features (shifts here are especially dangerous for fraud detection):
  amount, txn_count_1h, sender_age_days

Respond ONLY with a JSON object. No explanation text outside the JSON.
Schema:
{
  "decision":   "retrain" | "rollback" | "alert" | "ok",
  "confidence": <float 0.0-1.0>,
  "reasoning":  "<one to three sentences explaining the decision>"
}\
"""


def _llm_reason(drift_report: dict) -> Optional[tuple[Decision, float, str]]:
    """
    Ask Claude claude-haiku-4-5 to reason over the drift report.

    Returns:
        (decision, confidence, reasoning) on success.
        None on any failure — caller will fall back to rule-based.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — falling back to rule-based reasoning.")
        return None

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — falling back to rule-based reasoning.")
        return None

    # Prune large fields to keep prompt short (saves tokens on Haiku)
    report_for_prompt = {
        k: v for k, v in drift_report.items()
        if k != "report_html_path"   # path is not useful context for the LLM
    }

    user_message = (
        "Here is the drift report for our UPI fraud detection model:\n\n"
        + json.dumps(report_for_prompt, indent=2, default=str)
        + "\n\nDecide the corrective action."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text: str = response.content[0].text.strip()
        logger.debug("LLM raw response: %s", raw_text)
    except Exception as exc:
        logger.warning("Anthropic API call failed: %s — falling back to rule-based.", exc)
        return None

    # Extract JSON even if the model wrapped it in ```json ... ```
    if "```" in raw_text:
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        raw_text = match.group(1) if match else raw_text

    try:
        parsed: dict = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM response is not valid JSON: %s — falling back.", exc)
        return None

    decision_raw: str = str(parsed.get("decision", "")).strip().lower()
    if decision_raw not in VALID_DECISIONS:
        logger.warning(
            "LLM returned invalid decision '%s' — falling back to rule-based.",
            decision_raw,
        )
        return None

    confidence: float = float(parsed.get("confidence", 0.8))
    confidence = max(0.0, min(1.0, confidence))
    reasoning:  str   = str(parsed.get("reasoning", "LLM provided no reasoning."))

    return decision_raw, round(confidence, 4), reasoning   # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════════
# Node function
# ═══════════════════════════════════════════════════════════════════════════════

def reason_node(state: AgentState) -> AgentState:
    """
    LangGraph node: decide what action to take based on the drift report.

    Reads AGENT_MODE env var:
      "llm"        → try LLM first, fall back to rule-based
      anything else → rule-based only (default)

    Args:
        state: AgentState with drift_report populated by monitor_node.

    Returns:
        Partial AgentState with decision, confidence, reasoning updated.
    """
    run_id       = state.get("run_id", "unknown")
    drift_report = state.get("drift_report", {})
    retry_count  = state.get("retry_count", 0)
    agent_mode   = os.environ.get("AGENT_MODE", "rule_based").strip().lower()

    logger.info(
        "[%s] reason_node | mode=%s | retry=%d | drift_score=%.3f",
        run_id, agent_mode, retry_count,
        drift_report.get("drift_score", 0.0),
    )

    # On retry: downgrade decision one level to avoid repeated hard failures
    # e.g. if "retrain" failed, try "rollback" instead
    downgrade_on_retry: dict[str, Decision] = {
        "retrain":  "rollback",
        "rollback": "alert",
        "alert":    "ok",
        "ok":       "ok",
    }

    decision:   Decision
    confidence: float
    reasoning:  str

    if agent_mode == "llm":
        result = _llm_reason(drift_report)
        if result is not None:
            decision, confidence, reasoning = result
            logger.info("[%s] reason_node | LLM decision=%s confidence=%.2f", run_id, decision, confidence)
        else:
            logger.info("[%s] reason_node | LLM failed, using rule-based fallback", run_id)
            decision, confidence, reasoning = _rule_based_reason(drift_report)
    else:
        decision, confidence, reasoning = _rule_based_reason(drift_report)
        logger.info("[%s] reason_node | rule-based decision=%s confidence=%.2f", run_id, decision, confidence)

    # Apply retry downgrade
    if retry_count > 0 and decision != "ok":
        original   = decision
        decision   = downgrade_on_retry[decision]
        confidence = max(0.0, confidence - 0.15)
        reasoning  = (
            f"[Retry {retry_count}] Previous action failed. "
            f"Downgrading decision from '{original}' to '{decision}'. "
            + reasoning
        )
        logger.info(
            "[%s] reason_node | retry downgrade: %s -> %s", run_id, original, decision
        )

    return {
        "decision":   decision,
        "confidence": confidence,
        "reasoning":  reasoning,
        "error":      None,   # clear act_node errors before retry; monitor errors
                              # never reach here (blocked by _route_after_monitor)
    }
