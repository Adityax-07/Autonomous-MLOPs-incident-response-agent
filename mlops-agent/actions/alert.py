# mlops-agent/actions/alert.py
"""
Alert action — Phase 4 implementation.

Sends a formatted Slack message using Block Kit (rich cards with fields,
buttons, and context). If SLACK_WEBHOOK_URL is not set or the POST fails,
falls back to printing to stdout and writing alerts/alert_{ts}.json.

Environment variables:
  SLACK_WEBHOOK_URL   Incoming webhook URL from Slack app settings.
                      If unset, file-based fallback is used.
  GRAFANA_URL         Base URL for the Grafana dashboard link.
                      Defaults to http://localhost:3000.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

from agent.state import AgentState

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
ALERTS_DIR = ROOT / "alerts"


# ── Slack Block Kit builder ───────────────────────────────────────────────────

def _decision_emoji(decision: str) -> str:
    return {
        "retrain":  ":red_circle:",
        "rollback": ":large_orange_circle:",
        "alert":    ":large_yellow_circle:",
        "ok":       ":large_green_circle:",
    }.get(decision.lower(), ":white_circle:")


def _build_slack_blocks(
    drift_report: dict[str, Any],
    decision: str,
    reasoning: str,
    run_id: str,
) -> list[dict]:
    """
    Build a Slack Block Kit message for the MLOps alert.

    Sections:
      Header:  rotating_light + model name
      Fields:  decision, drift_score, drifted_columns, timestamp
      Quality: data quality issues (if any)
      Reason:  the agent's reasoning text
      Action:  "Open Grafana" button
      Footer:  auto-generated context line

    Args:
        drift_report: Drift report dict from generate_drift_report().
        decision:     Agent decision string ("alert", "rollback", etc.).
        reasoning:    Plain-English explanation from reason_node.
        run_id:       Agent run UUID for traceability.

    Returns:
        List of Slack Block Kit block dicts.
    """
    drift_score    = float(drift_report.get("drift_score", 0.0))
    drifted_cols   = drift_report.get("drifted_columns", [])
    report_ts      = drift_report.get("timestamp", "unknown")
    quality_issues = drift_report.get("quality_issues", [])
    grafana_url    = os.environ.get("GRAFANA_URL", "http://localhost:3000")
    emoji          = _decision_emoji(decision)

    # Truncate reasoning to Slack's 3000-char field limit
    reasoning_display = reasoning[:800] + ("..." if len(reasoning) > 800 else "")

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":rotating_light: MLOps Agent Alert -- fraud_model",
                "emoji": True,
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Decision:*\n{emoji} `{decision.upper()}`",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Drift Score:*\n`{drift_score:.1%}`",
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Drifted Features:*\n"
                        f"`{', '.join(drifted_cols) if drifted_cols else 'none'}`"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Report Timestamp:*\n`{report_ts}`",
                },
            ],
        },
    ]

    if quality_issues:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    ":warning: *Data Quality Issues:*\n"
                    + "\n".join(f"- {q}" for q in quality_issues)
                ),
            },
        })

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Reasoning:*\n{reasoning_display}",
        },
    })

    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Open Grafana Dashboard", "emoji": True},
                "url":   f"{grafana_url}/d/fraud-model",
                "style": "primary",
            },
        ],
    })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": (
                    f"_Automated alert from MLOps Incident Response Agent | "
                    f"Model: `fraud_model` | "
                    f"Run: `{run_id[:8]}` | "
                    f"Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}_"
                ),
            }
        ],
    })

    return blocks


# ── Main action function ──────────────────────────────────────────────────────

def trigger_alert(state: AgentState) -> str:
    """
    Send a rich Slack alert for the detected drift event, or write to a local
    JSON file if Slack is not configured / unreachable.

    Args:
        state: Full AgentState — reads drift_report, decision, reasoning, run_id.

    Returns:
        Status string:
          "sent|channel=slack"          — Webhook POST succeeded
          "logged|channel=file|path=X"  — Written to alerts/alert_{ts}.json
    """
    drift_report = state.get("drift_report", {})
    decision     = state.get("decision", "alert")
    reasoning    = state.get("reasoning", "No reasoning provided.")
    run_id       = state.get("run_id", "unknown")

    drift_score  = float(drift_report.get("drift_score", 0.0))
    drifted_cols = drift_report.get("drifted_columns", [])

    logger.info(
        "[ALERT] Preparing | run_id=%s | decision=%s | drift_score=%.3f | features=%s",
        run_id, decision, drift_score, drifted_cols,
    )

    blocks  = _build_slack_blocks(drift_report, decision, reasoning, run_id)
    payload = {
        "text": (
            f":rotating_light: MLOps Alert: {decision.upper()} | "
            f"fraud_model drift_score={drift_score:.1%}"
        ),
        "blocks": blocks,
    }

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()

    # ── Try Slack webhook ─────────────────────────────────────────────────────
    if webhook_url:
        try:
            resp = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200 and resp.text == "ok":
                logger.info("[ALERT] Slack message sent successfully.")
                return "sent|channel=slack"

            logger.warning(
                "[ALERT] Slack webhook returned HTTP %d: '%s' — falling back to file.",
                resp.status_code, resp.text[:120],
            )
        except requests.exceptions.ConnectionError as exc:
            logger.warning("[ALERT] Slack webhook unreachable: %s — falling back to file.", exc)
        except Exception as exc:
            logger.warning("[ALERT] Slack send failed: %s — falling back to file.", exc)

    # ── File fallback ─────────────────────────────────────────────────────────
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_slug    = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    alert_path = ALERTS_DIR / f"alert_{ts_slug}.json"

    alert_record: dict[str, Any] = {
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id":          run_id,
        "decision":        decision,
        "drift_score":     drift_score,
        "drifted_columns": drifted_cols,
        "quality_issues":  drift_report.get("quality_issues", []),
        "reasoning":       reasoning,
        "feature_details": drift_report.get("feature_details", {}),
        "slack_payload":   payload,
    }
    alert_path.write_text(json.dumps(alert_record, indent=2))
    logger.info("[ALERT] Alert written to %s", alert_path)

    # Also print a human-readable summary so the console is never silent
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  MLOPS ALERT: {decision.upper()}")
    print(f"  Drift score : {drift_score:.1%}")
    print(f"  Features    : {', '.join(drifted_cols) if drifted_cols else 'none'}")
    print(f"  Reasoning   : {reasoning[:120]}{'...' if len(reasoning) > 120 else ''}")
    print(f"  Alert file  : {alert_path}")
    print(f"  (Set SLACK_WEBHOOK_URL to send to Slack instead)")
    print(f"{sep}\n")

    return f"logged|channel=file|path={alert_path.name}"
