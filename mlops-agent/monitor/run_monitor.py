# mlops-agent/monitor/run_monitor.py
"""
Monitor entry point — runs drift detection and prints a structured report.

This script is the bridge between raw CSVs and the LangGraph agent.
It is called:
  • Directly from the terminal (for testing)
  • By the LangGraph MonitorNode in Phase 3 (programmatic import)

Usage:
    python -m monitor.run_monitor
    python -m monitor.run_monitor --ref data/reference.csv --cur data/production.csv

Output:
    Prints a human-readable summary to stdout.
    Saves HTML + JSON reports to monitor/reports/.
    Returns a DriftReport TypedDict (when imported programmatically).

Exit codes:
    0 — drift NOT detected
    1 — drift DETECTED  (signals the agent to wake up)
    2 — runtime error
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from monitor.drift_detector import FEATURE_COLUMNS, DriftDetector, DriftReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "monitor" / "reports"


# ── Programmatic API (used by LangGraph agent in Phase 3) ─────────────────────

def run_drift_check(
    ref_path:  Path = DATA_DIR / "reference.csv",
    cur_path:  Path = DATA_DIR / "production.csv",
    reports_dir: Path = REPORTS_DIR,
) -> DriftReport:
    """
    Load CSVs and run Evidently drift detection.

    This function is imported by the LangGraph MonitorNode — keep the
    signature stable between phases.

    Args:
        ref_path:    Path to the reference (training) CSV.
        cur_path:    Path to the current production CSV.
        reports_dir: Directory to save HTML / JSON reports.

    Returns:
        DriftReport TypedDict.

    Raises:
        FileNotFoundError: if either CSV is missing.
        RuntimeError:      if Evidently detection fails.
    """
    for path in (ref_path, cur_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                "Run: python data/generate_data.py"
            )

    try:
        ref_df = pd.read_csv(ref_path)
        cur_df = pd.read_csv(cur_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV files: {exc}") from exc

    logger.info(
        "Loaded datasets | reference=%d rows | current=%d rows",
        len(ref_df), len(cur_df),
    )

    detector = DriftDetector(
        reference_df=ref_df,
        reports_dir=reports_dir,
        feature_columns=FEATURE_COLUMNS,
    )
    report = detector.detect(cur_df)
    return report


# ── CLI formatting ────────────────────────────────────────────────────────────

def _print_report(report: DriftReport) -> None:
    """Pretty-print a DriftReport to stdout."""
    status = "DRIFT DETECTED" if report["drift_detected"] else "NO DRIFT"
    border = "=" * 60

    print(f"\n{border}")
    print(f"  UPI Fraud Monitor — {report['timestamp']}")
    print(f"  Status: {status}")
    print(border)
    print(f"  Features monitored : {report['total_features']}")
    print(f"  Features drifted   : {report['n_drifted_features']}")
    print(f"  Drift share        : {report['drift_share'] * 100:.1f}%")
    print()
    print(f"  {'Feature':<22} {'Drifted?':>10} {'Score':>12}  {'Test'}")
    print(f"  {'-'*22} {'-'*10} {'-'*12}  {'-'*16}")

    for col, detail in report["feature_details"].items():
        flag  = "[DRIFT]" if detail["drift_detected"] else "ok"
        score = detail["drift_score"]
        test  = detail["stat_test"]
        print(f"  {col:<22} {flag:>10} {score:>12.6f}  {test}")

    if report["drifted_features"]:
        print(f"\n  Drifted features: {', '.join(report['drifted_features'])}")

    print(f"\n  HTML report : {report['report_html_path']}")
    print(f"  JSON report : {report['report_json_path']}")
    print(border + "\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Evidently drift detection on UPI fraud data"
    )
    parser.add_argument(
        "--ref", type=Path, default=DATA_DIR / "reference.csv",
        help="Path to reference (training) CSV",
    )
    parser.add_argument(
        "--cur", type=Path, default=DATA_DIR / "production.csv",
        help="Path to current production CSV",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=REPORTS_DIR,
        help="Directory to save HTML/JSON drift reports",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Print raw JSON report to stdout (for piping to other tools)",
    )
    args = parser.parse_args()

    try:
        report = run_drift_check(
            ref_path=args.ref,
            cur_path=args.cur,
            reports_dir=args.reports_dir,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(2)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(2)

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_report(report)

    # Exit code 1 if drift detected — useful for shell scripts / CI
    sys.exit(1 if report["drift_detected"] else 0)


if __name__ == "__main__":
    main()
