# mlops-agent/agent/runner.py
"""
CLI entry point for the MLOps Incident Response Agent.

Modes:
  --once     Run the graph once, print the final state, exit.
  --watch    Run every N seconds (default 300s = 5 minutes), matching the
             monitor scheduler cadence. Ctrl+C to stop.
  --mock     Inject a synthetic drift report instead of reading the live file.
             Useful for local testing without the scheduler running.

Usage:
    python -m agent.runner --once
    python -m agent.runner --once --mock retrain
    python -m agent.runner --watch --interval 60
    AGENT_MODE=llm python -m agent.runner --once
"""

import argparse
import logging
import signal
import sys
import time
import uuid
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


# ── Mock report factory ───────────────────────────────────────────────────────

def _mock_drift_report(scenario: str) -> dict:
    """
    Build a synthetic drift report for offline testing.

    Scenarios:
      "retrain"  — severe drift on 5 features (drift_score=0.625)
      "rollback" — moderate drift on 3 features (drift_score=0.375)
      "alert"    — minor drift on 1 feature    (drift_score=0.125)
      "ok"       — no drift                    (drift_score=0.0)

    Args:
        scenario: One of "retrain", "rollback", "alert", "ok".

    Returns:
        Dict matching the generate_drift_report() schema.
    """
    import time as _time

    base: dict = {
        "drift_detected":   False,
        "drift_score":      0.0,
        "drifted_columns":  [],
        "quality_issues":   [],
        "timestamp":        _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "recommendation":   "ok",
        "feature_details":  {
            col: {"drift_detected": False, "drift_score": 0.01, "stat_test": "Wasserstein distance (normed)"}
            for col in ["amount", "hour", "merchant_cat", "device_type",
                        "sender_age_days", "receiver_age_days", "txn_count_1h", "same_device"]
        },
        "report_html_path": "mock_report.html",
    }

    if scenario == "retrain":
        drifted = ["amount", "hour", "device_type", "sender_age_days", "txn_count_1h"]
        base.update({
            "drift_detected": True,
            "drift_score":    0.625,
            "drifted_columns": drifted,
            "recommendation": "retrain",
        })
        for col in drifted:
            base["feature_details"][col]["drift_detected"] = True
            base["feature_details"][col]["drift_score"] = 1.8

    elif scenario == "rollback":
        drifted = ["amount", "hour", "txn_count_1h"]
        base.update({
            "drift_detected": True,
            "drift_score":    0.375,
            "drifted_columns": drifted,
            "recommendation": "rollback",
        })
        for col in drifted:
            base["feature_details"][col]["drift_detected"] = True
            base["feature_details"][col]["drift_score"] = 1.3

    elif scenario == "alert":
        drifted = ["txn_count_1h"]
        base.update({
            "drift_detected": True,
            "drift_score":    0.125,
            "drifted_columns": drifted,
            "recommendation": "alert",
        })
        base["feature_details"]["txn_count_1h"]["drift_detected"] = True
        base["feature_details"]["txn_count_1h"]["drift_score"] = 0.42

    # "ok" — defaults are already correct

    return base


# ── Pretty-print final state ─────────────────────────────────────────────────

def _print_summary(state: dict, elapsed: float) -> None:
    """Format and print the final AgentState as a structured summary."""
    report   = state.get("drift_report", {})
    decision = state.get("decision", "unknown")
    error    = state.get("error")

    status_line = f"  Decision     : {decision.upper()}"
    if error:
        status_line += f"  [ERROR: {error}]"

    print("\n" + "=" * 62)
    print(f"  MLOps Agent Run Summary | run_id={state.get('run_id', '?')[:8]}...")
    print("=" * 62)
    print(f"  Drift score  : {report.get('drift_score', 'N/A')}")
    print(f"  Drifted cols : {report.get('drifted_columns', [])}")
    print(f"  Report ts    : {report.get('timestamp', 'N/A')}")
    print("-" * 62)
    print(status_line)
    print(f"  Confidence   : {state.get('confidence', 0.0):.2f}")
    print(f"  Action taken : {state.get('action_taken', 'none')}")
    print(f"  Elapsed      : {elapsed:.2f}s")
    print("-" * 62)
    print(f"  Reasoning:\n  {state.get('reasoning', '')}")
    if error:
        print(f"\n  Error: {error}")
    print("=" * 62 + "\n")


# ── Single run ─────────────────────────────────────────────────────────────────

def _run_once(mock_scenario: str | None) -> dict:
    """Execute the agent graph once and return the final state."""
    from agent.graph import run_agent

    seed: dict = {}

    if mock_scenario:
        # Write a mock report to disk so monitor_node reads it,
        # OR inject directly via seed to bypass monitor_node file I/O.
        import json, os
        mock_report = _mock_drift_report(mock_scenario)
        report_path = ROOT / "monitor" / "latest_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(mock_report, indent=2))
        logger.info("Mock '%s' report written to %s", mock_scenario, report_path)

    t0    = time.perf_counter()
    final = run_agent(seed=seed)
    return final, time.perf_counter() - t0


# ── Watch loop ────────────────────────────────────────────────────────────────

def _watch(interval: int, mock_scenario: str | None) -> None:
    """Run the agent on a fixed interval until Ctrl+C."""
    def _handle_signal(sig, frame):
        print("\nShutting down agent watcher. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"Agent watcher started | interval={interval}s | Ctrl+C to stop\n")
    run_count = 0

    while True:
        run_count += 1
        tick = time.strftime("%H:%M:%S")
        print(f"[{tick}] Run #{run_count} starting...")
        try:
            final, elapsed = _run_once(mock_scenario)
            _print_summary(final, elapsed)
        except Exception as exc:
            logger.exception("Agent run failed: %s", exc)

        print(f"Next run in {interval}s...\n")
        time.sleep(interval)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLOps Incident Response Agent runner"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--once", action="store_true",
        help="Run the agent once then exit",
    )
    group.add_argument(
        "--watch", action="store_true",
        help="Run on a fixed interval until Ctrl+C",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Watch mode interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--mock",
        choices=["retrain", "rollback", "alert", "ok"],
        default=None,
        help="Inject a synthetic drift report (no scheduler needed). "
             "Writes a mock latest_report.json before running.",
    )
    args = parser.parse_args()

    if not (args.once or args.watch):
        # Default to --once if neither flag given
        args.once = True

    if args.once:
        try:
            final, elapsed = _run_once(args.mock)
            _print_summary(final, elapsed)
            sys.exit(1 if final.get("error") else 0)
        except Exception as exc:
            logger.exception("Fatal error: %s", exc)
            sys.exit(2)
    else:
        _watch(args.interval, args.mock)


if __name__ == "__main__":
    main()
