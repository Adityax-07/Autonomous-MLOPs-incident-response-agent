# mlops-agent/monitor/drift_check.py
"""
Evidently-based drift + quality monitor for UPI fraud detection.

Public API
----------
generate_drift_report(reference_df, production_df) -> dict

Returns a structured report consumed by the LangGraph agent (Phase 3):

    {
        "drift_detected":  bool,
        "drift_score":     float,      # share of drifted columns (0–1)
        "drifted_columns": list[str],  # which features drifted
        "quality_issues":  list[str],  # missing values, constant cols, etc.
        "timestamp":       str,        # ISO-8601 UTC
        "recommendation":  str,        # "retrain"|"rollback"|"alert"|"ok"
        "feature_details": dict,       # per-column drift scores
        "report_html_path": str,
    }

Recommendation logic (drift_score thresholds):
    > 0.50  → "retrain"   (majority of features have shifted — model is stale)
    > 0.30  → "rollback"  (significant drift — revert to last stable version)
    > 0.10  → "alert"     (minor drift — watch but don't act yet)
    else    → "ok"

Run standalone:
    python -m monitor.drift_check
    python -m monitor.drift_check --ref data/reference.csv --cur data/drifted_data.csv
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.report import Report

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
REPORTS_DIR = ROOT / "monitor" / "reports"

FEATURE_COLUMNS: list[str] = [
    "amount", "hour", "merchant_cat", "device_type",
    "sender_age_days", "receiver_age_days", "txn_count_1h", "same_device",
]

# ── Recommendation thresholds ──────────────────────────────────────────────────
THRESHOLDS = {
    "retrain":  0.50,
    "rollback": 0.30,
    "alert":    0.10,
}


def _recommend(drift_score: float) -> str:
    """
    Map a drift score (fraction of drifted columns) to an action label.

    Args:
        drift_score: float in [0, 1]

    Returns:
        "retrain" | "rollback" | "alert" | "ok"
    """
    if drift_score > THRESHOLDS["retrain"]:
        return "retrain"
    if drift_score > THRESHOLDS["rollback"]:
        return "rollback"
    if drift_score > THRESHOLDS["alert"]:
        return "alert"
    return "ok"


def _extract_quality_issues(quality_result: dict) -> list[str]:
    """
    Parse Evidently DataQualityPreset output and return human-readable issues.

    Args:
        quality_result: The 'result' dict from the DataQualityPreset metric.

    Returns:
        List of issue strings (empty if no issues found).
    """
    issues: list[str] = []

    columns_info = quality_result.get("columns", {})
    if not columns_info:
        # Fallback: scan for known quality keys at top level
        columns_info = quality_result

    for col, info in columns_info.items():
        if not isinstance(info, dict):
            continue

        missing_pct = info.get("missing_percentage", 0) or 0
        if missing_pct > 5.0:
            issues.append(f"{col}: {missing_pct:.1f}% missing values")

        n_unique = info.get("unique_percentage", None)
        n_vals   = info.get("count", 1)
        if n_vals and n_unique == 0:
            issues.append(f"{col}: constant column (zero variance)")

        # Check for all-null
        n_missing = info.get("missing_count", 0) or 0
        if n_vals and n_missing == n_vals:
            issues.append(f"{col}: entirely null")

    return issues


def generate_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    reports_dir: Path = REPORTS_DIR,
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> dict:
    """
    Run Evidently DataDriftPreset + DataQualityPreset and return a structured report.

    This is the primary function imported by the LangGraph MonitorNode (Phase 3).
    Keep the return schema stable — the agent's reasoning nodes depend on it.

    Args:
        reference_df:    Training / baseline distribution.
        production_df:   Live / current distribution to compare.
        reports_dir:     Where to save the HTML report.
        feature_columns: Columns to monitor (must exist in both DataFrames).

    Returns:
        Dict conforming to the schema documented in this module's docstring.

    Raises:
        ValueError:   if required columns are missing from either DataFrame.
        RuntimeError: if Evidently report generation fails.
    """
    for label, df in [("reference", reference_df), ("production", production_df)]:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{label} DataFrame missing columns: {missing}")

    ref_features  = reference_df[feature_columns].copy()
    prod_features = production_df[feature_columns].copy()

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    safe_ts   = timestamp.replace(":", "-")
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / f"report_{safe_ts}.html"

    logger.info(
        "Generating drift report | ref=%d rows | prod=%d rows | ts=%s",
        len(ref_features), len(prod_features), timestamp,
    )

    # ── Run Evidently ─────────────────────────────────────────────────────────
    try:
        ev_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        ev_report.run(
            reference_data=ref_features,
            current_data=prod_features,
        )
    except Exception as exc:
        logger.exception("Evidently report generation failed: %s", exc)
        raise RuntimeError(f"Evidently failed: {exc}") from exc

    raw: dict = ev_report.as_dict()

    # ── Parse drift metrics ───────────────────────────────────────────────────
    drift_score: float = 0.0
    drifted_columns: list[str] = []
    feature_details: dict = {}
    drift_detected: bool = False

    for metric in raw.get("metrics", []):
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            drift_score     = float(result.get("share_of_drifted_columns", 0.0))
            drift_detected  = bool(result.get("dataset_drift", False))
            drift_by_columns: dict = result.get("drift_by_columns", {})

            for col in feature_columns:
                col_data = drift_by_columns.get(col, {})
                col_drifted = bool(col_data.get("drift_detected", False))
                col_score   = float(col_data.get("drift_score", 1.0))
                if col_drifted:
                    drifted_columns.append(col)
                feature_details[col] = {
                    "drift_detected": col_drifted,
                    "drift_score":    round(col_score, 8),
                    "stat_test":      str(col_data.get("stattest_name", "unknown")),
                }
            break  # found drift metric, stop scanning

    # ── Parse quality metrics ─────────────────────────────────────────────────
    quality_issues: list[str] = []
    for metric in raw.get("metrics", []):
        result = metric.get("result", {})
        # DataQualityPreset result has per-column info under 'current'
        if "current" in result and isinstance(result["current"], dict):
            quality_issues = _extract_quality_issues(result["current"])
            break
        elif "columns" in result:
            quality_issues = _extract_quality_issues(result)
            break

    # ── Drift detected flag ───────────────────────────────────────────────────
    # Evidently's internal dataset_drift uses its own 50% threshold.
    # We override it: drift_detected = True if ANY feature drifted AND
    # drift_score exceeds our "alert" threshold (> 0.10).
    # This makes drift_detected consistent with recommendation != "ok".
    drift_detected = len(drifted_columns) > 0 and drift_score > THRESHOLDS["alert"]

    # ── Recommendation ────────────────────────────────────────────────────────
    recommendation = _recommend(drift_score)

    # ── Save HTML report ──────────────────────────────────────────────────────
    try:
        ev_report.save_html(str(html_path))
        logger.info("HTML report saved -> %s", html_path)
    except Exception as exc:
        logger.warning("Could not save HTML report: %s", exc)
        html_path = Path("unavailable")

    report = {
        "drift_detected":   drift_detected,
        "drift_score":      round(drift_score, 4),
        "drifted_columns":  drifted_columns,
        "quality_issues":   quality_issues,
        "timestamp":        timestamp,
        "recommendation":   recommendation,
        "feature_details":  feature_details,
        "report_html_path": str(html_path),
    }

    logger.info(
        "Report complete | drift_score=%.2f | drifted=%s | recommendation=%s",
        drift_score, drifted_columns, recommendation,
    )
    return report


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run drift + quality check on UPI fraud data"
    )
    parser.add_argument("--ref", type=Path, default=DATA_DIR / "reference.csv")
    parser.add_argument("--cur", type=Path, default=DATA_DIR / "production.csv")
    args = parser.parse_args()

    for path in (args.ref, args.cur):
        if not path.exists():
            logger.error("File not found: %s", path)
            raise SystemExit(1)

    ref_df = pd.read_csv(args.ref)
    cur_df = pd.read_csv(args.cur)

    report = generate_drift_report(ref_df, cur_df)

    print("\n" + "=" * 55)
    print(f"  Drift Report  |  {report['timestamp']}")
    print("=" * 55)
    print(f"  drift_detected  : {report['drift_detected']}")
    print(f"  drift_score     : {report['drift_score']}  ({report['drift_score']*100:.1f}%)")
    print(f"  drifted_columns : {report['drifted_columns']}")
    print(f"  quality_issues  : {report['quality_issues'] or 'none'}")
    print(f"  recommendation  : {report['recommendation'].upper()}")
    print(f"  html_report     : {report['report_html_path']}")
    print("=" * 55 + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
