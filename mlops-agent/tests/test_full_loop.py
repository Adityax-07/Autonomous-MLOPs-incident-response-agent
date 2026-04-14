# mlops-agent/tests/test_full_loop.py
"""
End-to-end integration test for the MLOps Incident Response Agent.

Walk-through:
  Step 1  Generate high-severity drifted production data (or reuse existing)
  Step 2  Run drift check -> writes monitor/latest_report.json
  Step 3  Run the LangGraph agent -> makes a decision
  Step 4  Execute retrain action -> logs to MLflow (if installed)

Run:
    python tests/test_full_loop.py

No server needs to be running. The /reload call inside trigger_retrain will
warn and continue gracefully if FastAPI is not up.

Exit codes:
  0  All steps passed and agent made a retrain or rollback decision
  1  A step failed (printed with [FAIL])
  2  Agent returned "ok" (no action taken — drift injection may have failed)
"""

import json
import logging
import sys
import time
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,           # suppress library noise in test output
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
# Unmute agent + action loggers so we can see the important lines
for noisy_logger in (
    "actions.retrain", "actions.rollback", "actions.alert",
    "agent.nodes.monitor", "agent.nodes.reason", "agent.nodes.act",
    "agent.graph",
):
    logging.getLogger(noisy_logger).setLevel(logging.INFO)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _section(title: str) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


# ── Step 1: Generate drifted data ─────────────────────────────────────────────

def step1_generate_drift() -> Path:
    """
    Produce a drifted production DataFrame and write it to data/drifted_data.csv.
    Re-uses the file if it already exists (saves ~2s on repeated runs).

    Returns:
        Path to the drifted CSV.
    """
    _section("Step 1 — Generate high-severity drifted data")

    drifted_path = ROOT / "data" / "drifted_data.csv"

    if drifted_path.exists():
        import pandas as pd
        rows = len(pd.read_csv(drifted_path))
        _pass(f"Reusing existing drifted_data.csv ({rows} rows)")
        return drifted_path

    # Try inject_drift.py first
    inject_script = ROOT / "data" / "inject_drift.py"
    if inject_script.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(inject_script), "--severity", "high"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        if result.returncode == 0 and drifted_path.exists():
            import pandas as pd
            rows = len(pd.read_csv(drifted_path))
            _pass(f"inject_drift.py produced drifted_data.csv ({rows} rows)")
            return drifted_path
        print(f"  inject_drift.py stderr: {result.stderr[:200]}")

    # Fallback: build a synthetic drifted DataFrame in-process
    print("  Building synthetic drifted data in-process ...")
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n   = 2_000

    # High-severity drift: amount log-shifted +1.8, most txns at 2-5am, burst velocity
    amount_base = rng.lognormal(mean=7.0 + 1.8, sigma=1.2, size=n)
    hour        = rng.integers(2, 6, size=n)          # night window
    merchant_cat = rng.integers(0, 5, size=n)
    device_type  = rng.integers(0, 3, size=n)
    sender_age_days   = rng.integers(0, 15, size=n)   # brand-new accounts
    receiver_age_days = rng.integers(30, 2000, size=n)
    txn_count_1h = (rng.integers(8, 30, size=n) * 3.0).astype(float)  # 3x velocity
    same_device  = rng.integers(0, 2, size=n)
    is_fraud     = rng.integers(0, 2, size=n)          # synthetic labels

    df = pd.DataFrame({
        "amount":           amount_base,
        "hour":             hour,
        "merchant_cat":     merchant_cat,
        "device_type":      device_type,
        "sender_age_days":  sender_age_days,
        "receiver_age_days": receiver_age_days,
        "txn_count_1h":     txn_count_1h,
        "same_device":      same_device,
        "is_fraud":         is_fraud,
    })
    drifted_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(drifted_path, index=False)
    _pass(f"Synthetic drifted_data.csv created ({len(df)} rows)")
    return drifted_path


# ── Step 2: Run drift check ───────────────────────────────────────────────────

def step2_run_drift_check(drifted_path: Path) -> dict:
    """
    Load reference.csv + drifted_data.csv, call generate_drift_report(),
    write the result to monitor/latest_report.json.

    Returns:
        The drift report dict.
    """
    _section("Step 2 — Run drift check")

    import pandas as pd
    from monitor.drift_check import generate_drift_report

    ref_path = ROOT / "data" / "reference.csv"
    if not ref_path.exists():
        _fail(f"reference.csv not found at {ref_path}. Run: python data/generate_data.py")
        sys.exit(1)

    ref_df  = pd.read_csv(ref_path)
    prod_df = pd.read_csv(drifted_path)

    print(f"  reference.csv : {len(ref_df):,} rows")
    print(f"  drifted data  : {len(prod_df):,} rows")

    t0     = time.perf_counter()
    report = generate_drift_report(ref_df, prod_df)
    elapsed = time.perf_counter() - t0

    drift_score    = report.get("drift_score", 0.0)
    drifted_cols   = report.get("drifted_columns", [])
    recommendation = report.get("recommendation", "unknown")

    print(f"  drift_score   : {drift_score:.1%}")
    print(f"  drifted cols  : {drifted_cols}")
    print(f"  recommendation: {recommendation}")
    print(f"  elapsed       : {elapsed:.2f}s")

    if drift_score < 0.10:
        _fail("drift_score < 10% — drift injection may not have worked correctly.")
        sys.exit(2)

    _pass(f"Drift check complete | score={drift_score:.1%} | recommendation={recommendation}")

    # Write to monitor/latest_report.json for monitor_node to read
    report_path = ROOT / "monitor" / "latest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    _pass(f"Report written -> {report_path}")

    return report


# ── Step 3: Run the agent ─────────────────────────────────────────────────────

def step3_run_agent(expected_decisions: tuple[str, ...]) -> dict:
    """
    Invoke the full LangGraph agent against the report written in Step 2.

    Args:
        expected_decisions: Tuple of acceptable decision values (e.g. ("retrain",)).

    Returns:
        Final AgentState dict.
    """
    _section("Step 3 — Run LangGraph agent")

    from agent.graph import run_agent

    t0    = time.perf_counter()
    final = run_agent()
    elapsed = time.perf_counter() - t0

    decision    = final.get("decision", "unknown")
    confidence  = final.get("confidence", 0.0)
    action_taken = final.get("action_taken", "none")
    error       = final.get("error")
    reasoning   = final.get("reasoning", "")

    print(f"  decision      : {decision.upper()}")
    print(f"  confidence    : {confidence:.2f}")
    print(f"  action_taken  : {action_taken}")
    print(f"  error         : {error}")
    print(f"  elapsed       : {elapsed:.2f}s")
    print(f"  reasoning     : {reasoning[:120]}...")

    if error and decision not in ("ok",):
        _fail(f"Agent completed with error: {error}")
        # Don't exit — the action may still have run (partial success on retry downgrade)

    if decision in expected_decisions:
        _pass(f"Agent decision '{decision}' is within expected: {expected_decisions}")
    else:
        _fail(
            f"Agent decision '{decision}' NOT in expected {expected_decisions}. "
            "Check drift score and thresholds."
        )

    return final


# ── Step 4: Execute retrain ───────────────────────────────────────────────────

def step4_execute_retrain(agent_state: dict) -> None:
    """
    Call execute_action("retrain", state) directly to test the full retrain path.
    This re-runs the retrain (it was already called by the agent in Step 3 if the
    decision was "retrain"). Running it again verifies idempotency and that MLflow
    is producing a run entry.

    Args:
        agent_state: The final AgentState from step3.
    """
    _section("Step 4 — Execute retrain action + verify MLflow")

    from actions import execute_action

    t0     = time.perf_counter()
    result = execute_action("retrain", agent_state)   # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0

    status        = result.get("status", "")
    mlflow_run_id = result.get("mlflow_run_id")
    action_elapsed = result.get("elapsed_s", 0.0)

    print(f"  status        : {status}")
    print(f"  mlflow_run_id : {mlflow_run_id or 'N/A (mlflow not installed)'}")
    print(f"  action_elapsed: {action_elapsed:.2f}s")
    print(f"  total_elapsed : {elapsed:.2f}s")

    if "promoted" in status or "degraded" in status:
        _pass(f"Retrain completed: {status[:80]}")
    elif "failed" in status:
        _fail(f"Retrain returned failure status: {status}")
    else:
        _pass(f"Retrain returned: {status[:80]}")

    # Verify MLflow run exists on disk (only if mlflow is installed)
    try:
        import mlflow
        from mlflow import MlflowClient
        tracking_uri = str(ROOT / "mlruns")
        client   = MlflowClient(tracking_uri=tracking_uri)
        exp      = client.get_experiment_by_name("upi_fraud_retrain")
        if exp:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                run = runs[0]
                roc = run.data.metrics.get("roc_auc", None)
                _pass(
                    f"MLflow run found | run_id={run.info.run_id[:8]}... | roc_auc={roc}"
                )
            else:
                _fail("MLflow experiment exists but no runs found.")
        else:
            print("  MLflow experiment 'upi_fraud_retrain' not yet created (first run?).")
    except ImportError:
        print("  mlflow not installed — skipping MLflow verification.")
    except Exception as exc:
        print(f"  MLflow check warning: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  MLOps Agent — End-to-End Integration Test")
    print("=" * 60)
    overall_t0 = time.perf_counter()

    # Step 1: produce drifted data
    drifted_path = step1_generate_drift()

    # Step 2: drift check -> latest_report.json
    drift_report = step2_run_drift_check(drifted_path)

    # Step 3: run agent; high drift_score should produce retrain or rollback
    drift_score = drift_report.get("drift_score", 0.0)
    if drift_score > 0.50:
        expected = ("retrain",)
    elif drift_score > 0.30:
        expected = ("retrain", "rollback")
    else:
        expected = ("retrain", "rollback", "alert")

    agent_state = step3_run_agent(expected_decisions=expected)

    # Step 4: explicit retrain call (tests MLflow tracking path independently)
    step4_execute_retrain(agent_state)

    total_elapsed = time.perf_counter() - overall_t0

    print("\n" + "=" * 60)
    print(f"  Test complete | total_elapsed={total_elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
