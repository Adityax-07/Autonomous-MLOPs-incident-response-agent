# mlops-agent/agent/nodes/monitor.py
"""
monitor_node — first node in the LangGraph agent.

Responsibilities:
  1. Read monitor/latest_report.json from disk
  2. Reject stale reports (written > MAX_REPORT_AGE_SECONDS ago)
  3. Validate that all required keys are present and correctly typed
  4. Return updated AgentState with drift_report populated (or error set)

Why check staleness?
  The scheduler writes latest_report.json every 5 minutes. If the scheduler
  has died, the agent would act on old data — potentially retraining on a
  drift event that has already self-corrected. Staleness check prevents this.

Why validate schema?
  The agent's reason_node reads specific keys from drift_report. A bad write
  (disk full, partial flush) would cause a silent KeyError deep in reasoning.
  Validate early, fail loudly.
"""

import json
import logging
import os
import time
from pathlib import Path

from agent.state import AgentState

logger = logging.getLogger(__name__)

ROOT              = Path(__file__).parent.parent.parent
LATEST_REPORT_PATH = ROOT / "monitor" / "latest_report.json"

# Reports older than this are considered stale
MAX_REPORT_AGE_SECONDS: int = 600   # 10 minutes

# All keys that reason_node depends on
REQUIRED_REPORT_KEYS: dict[str, type] = {
    "drift_detected":   bool,
    "drift_score":      float,
    "drifted_columns":  list,
    "quality_issues":   list,
    "timestamp":        str,
    "recommendation":   str,
    "feature_details":  dict,
}


def _validate_report(report: dict) -> list[str]:
    """
    Check that report contains all required keys with correct types.

    Returns:
        List of validation error strings. Empty list = report is valid.
    """
    errors: list[str] = []
    for key, expected_type in REQUIRED_REPORT_KEYS.items():
        if key not in report:
            errors.append(f"missing key: '{key}'")
        elif not isinstance(report[key], expected_type):
            actual = type(report[key]).__name__
            errors.append(
                f"key '{key}': expected {expected_type.__name__}, got {actual}"
            )
    return errors


def _is_stale(report: dict, max_age_seconds: int) -> bool:
    """
    Return True if the report's timestamp is older than max_age_seconds.

    Uses calendar.timegm() (not time.mktime()) to parse the ISO-8601 UTC
    string correctly regardless of the local machine timezone.
    Gracefully returns False if the timestamp is missing or unparseable.
    """
    import calendar
    ts_str = report.get("timestamp", "")
    if not ts_str:
        return False
    try:
        # timegm treats the struct_time as UTC — correct for our "Z" timestamps
        report_time = calendar.timegm(time.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ"))
        age_seconds = time.time() - report_time
        return age_seconds > max_age_seconds
    except (ValueError, OverflowError):
        logger.warning("Could not parse report timestamp: %s", ts_str)
        return False


def monitor_node(state: AgentState) -> AgentState:
    """
    LangGraph node: load and validate the latest drift report.

    Args:
        state: Current AgentState (run_id already set by caller).

    Returns:
        Partial AgentState dict. Sets drift_report on success, error on failure.
    """
    run_id = state.get("run_id", "unknown")
    report_path = Path(
        os.environ.get("LATEST_REPORT_PATH", str(LATEST_REPORT_PATH))
    )

    logger.info("[%s] monitor_node | reading %s", run_id, report_path)

    # ── 1. File existence ─────────────────────────────────────────────────────
    if not report_path.exists():
        msg = f"Report file not found: {report_path}"
        logger.error("[%s] monitor_node | %s", run_id, msg)
        return {"error": msg}

    # ── 2. Read and parse JSON ─────────────────────────────────────────────────
    try:
        raw = report_path.read_text(encoding="utf-8")
        report: dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"Malformed JSON in report file: {exc}"
        logger.error("[%s] monitor_node | %s", run_id, msg)
        return {"error": msg}
    except OSError as exc:
        msg = f"Could not read report file: {exc}"
        logger.error("[%s] monitor_node | %s", run_id, msg)
        return {"error": msg}

    # ── 3. Staleness check ─────────────────────────────────────────────────────
    max_age = int(os.environ.get("MAX_REPORT_AGE_SECONDS", MAX_REPORT_AGE_SECONDS))
    if _is_stale(report, max_age):
        msg = (
            f"stale_report: timestamp={report.get('timestamp')} "
            f"is older than {max_age}s"
        )
        logger.warning("[%s] monitor_node | %s", run_id, msg)
        return {"error": msg, "drift_report": report}

    # ── 4. Schema validation ──────────────────────────────────────────────────
    validation_errors = _validate_report(report)
    if validation_errors:
        msg = "invalid_report_schema: " + "; ".join(validation_errors)
        logger.error("[%s] monitor_node | %s", run_id, msg)
        return {"error": msg}

    logger.info(
        "[%s] monitor_node | OK | drift_detected=%s | drift_score=%.3f | ts=%s",
        run_id,
        report.get("drift_detected"),
        report.get("drift_score", 0.0),
        report.get("timestamp"),
    )
    return {"drift_report": report, "error": None}
