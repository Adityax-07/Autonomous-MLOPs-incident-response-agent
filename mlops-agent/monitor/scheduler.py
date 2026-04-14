# mlops-agent/monitor/scheduler.py
"""
APScheduler-based drift monitor that runs every 5 minutes.

On each tick:
  1. Read the latest 500 rows from data/predictions.log (JSONL format)
     — each line is a JSON object containing feature values logged at prediction time
  2. Load reference.csv as the baseline distribution
  3. Run generate_drift_report() from drift_check.py
  4. Write the result to monitor/latest_report.json  (agent reads this file)
  5. Print a one-liner: [HH:MM:SS] drift_score=0.42 → recommendation=rollback

predictions.log format (one JSON object per line):
    {"amount": 4999.0, "hour": 14, "merchant_cat": 3, "device_type": 0,
     "sender_age_days": 365, "receiver_age_days": 200, "txn_count_1h": 2,
     "same_device": 1, "fraud_probability": 0.002, "timestamp": "..."}

Usage:
    python -m monitor.scheduler            # runs forever, checks every 5 min
    python -m monitor.scheduler --once     # run a single check then exit
    python -m monitor.scheduler --interval 60  # check every 60 seconds (testing)

The FastAPI /predict endpoint logs to predictions.log automatically
(see app/main.py — logging middleware added in Phase 2).
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.metrics import drift_check_duration_ms, model_drift_score
from monitor.drift_check import FEATURE_COLUMNS, generate_drift_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT             = Path(__file__).parent.parent
DATA_DIR         = ROOT / "data"
REF_PATH         = DATA_DIR / "reference.csv"
PREDICTIONS_LOG  = DATA_DIR / "predictions.log"
LATEST_REPORT    = ROOT / "monitor" / "latest_report.json"
REPORTS_DIR      = ROOT / "monitor" / "reports"

# How many recent prediction rows to use as "production" distribution
PRODUCTION_WINDOW = 500


# ── Log reader ────────────────────────────────────────────────────────────────

def _load_recent_predictions(n: int = PRODUCTION_WINDOW) -> pd.DataFrame | None:
    """
    Read the last n rows from predictions.log (JSONL).

    Returns None if the file doesn't exist or has fewer than 50 rows
    (not enough data for a meaningful drift check).

    Args:
        n: Maximum number of recent rows to use.

    Returns:
        DataFrame with FEATURE_COLUMNS, or None.
    """
    if not PREDICTIONS_LOG.exists():
        logger.warning(
            "predictions.log not found at %s. "
            "Start the FastAPI server and make some predictions first.",
            PREDICTIONS_LOG,
        )
        return None

    rows: list[dict] = []
    try:
        with PREDICTIONS_LOG.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip malformed lines
    except Exception as exc:
        logger.exception("Failed to read predictions.log: %s", exc)
        return None

    if len(rows) < 50:
        logger.warning(
            "Only %d rows in predictions.log (need >= 50 for reliable drift check).",
            len(rows),
        )
        return None

    # Take the most recent n rows
    recent = rows[-n:]
    try:
        df = pd.DataFrame(recent)
        # Keep only feature columns that are present
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        return df[available].dropna()
    except Exception as exc:
        logger.exception("Failed to parse prediction rows into DataFrame: %s", exc)
        return None


# ── Core check ────────────────────────────────────────────────────────────────

def run_check() -> None:
    """
    Single drift check cycle. Called by APScheduler on every tick.

    Steps:
      1. Load reference distribution from CSV
      2. Load recent predictions from predictions.log
      3. Run Evidently drift check
      4. Write result to latest_report.json
      5. Print one-line summary
    """
    tick = time.strftime("%H:%M:%S")

    if not REF_PATH.exists():
        logger.error("Reference CSV not found: %s", REF_PATH)
        return

    try:
        ref_df = pd.read_csv(REF_PATH)
    except Exception as exc:
        logger.exception("Failed to read reference CSV: %s", exc)
        return

    prod_df = _load_recent_predictions(PRODUCTION_WINDOW)
    if prod_df is None:
        print(f"[{tick}] drift_score=N/A -> recommendation=skip (insufficient data)")
        return

    try:
        t0     = time.perf_counter()
        report = generate_drift_report(
            reference_df=ref_df,
            production_df=prod_df,
            reports_dir=REPORTS_DIR,
            feature_columns=FEATURE_COLUMNS,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Prometheus metrics — updated after every successful drift check
        model_drift_score.set(report["drift_score"])
        drift_check_duration_ms.observe(elapsed_ms)

    except Exception as exc:
        logger.exception("Drift check failed: %s", exc)
        print(f"[{tick}] drift_score=ERROR -> recommendation=error ({exc})")
        return

    # Write latest_report.json (agent polls this file)
    try:
        LATEST_REPORT.parent.mkdir(parents=True, exist_ok=True)
        LATEST_REPORT.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not write latest_report.json: %s", exc)

    # One-line status output
    score_pct = report["drift_score"] * 100
    rec       = report["recommendation"]
    drifted   = ", ".join(report["drifted_columns"]) or "none"
    print(
        f"[{tick}] drift_score={report['drift_score']:.2f} ({score_pct:.0f}%) "
        f"| drifted=[{drifted}] "
        f"-> recommendation={rec}"
    )


# ── Graceful shutdown ─────────────────────────────────────────────────────────

def _handle_signal(signum: int, frame) -> None:
    logger.info("Received signal %d — shutting down scheduler.", signum)
    sys.exit(0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="APScheduler drift monitor for UPI fraud detection"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single check then exit (useful for CI / testing)",
    )
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Check interval in seconds (default: 300 = 5 minutes)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if args.once:
        logger.info("Running single drift check...")
        run_check()
        return

    logger.info(
        "Starting drift scheduler | interval=%ds | log=%s | report=%s",
        args.interval, PREDICTIONS_LOG, LATEST_REPORT,
    )

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        func=run_check,
        trigger=IntervalTrigger(seconds=args.interval),
        id="drift_monitor",
        name="UPI Fraud Drift Monitor",
        next_run_time=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),  # run immediately on start
    )

    print(f"Drift monitor running every {args.interval}s. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except Exception as exc:
        logger.exception("Scheduler crashed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
