# mlops-agent/monitor/drift_detector.py
"""
Core drift detection logic using Evidently AI (0.7.x legacy API).

Compares a reference dataset (training distribution) against a current
production dataset and produces a structured DriftReport consumed by the
LangGraph agent in Phase 3.

Evidently import path for 0.7.x:
    evidently.legacy.report      → Report
    evidently.legacy.metric_preset → DataDriftPreset

The drift test used per column is the Kolmogorov-Smirnov test (numerical)
and Chi-squared (categorical), both at alpha=0.05.

Output:
    DriftReport TypedDict — designed to be JSON-serialisable so the agent
    can log it to MLflow and reason over it in LangGraph nodes.
"""

import json
import logging
import time
from pathlib import Path
from typing import TypedDict

import pandas as pd

from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics import DatasetDriftMetric
from evidently.legacy.report import Report

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────────

class FeatureDriftDetail(TypedDict):
    drift_detected: bool
    drift_score: float       # raw stat-test p-value (lower = more drift)
    stat_test: str           # e.g. "K-S p_value"
    threshold: float         # e.g. 0.05


class DriftReport(TypedDict):
    drift_detected: bool           # True if >= drift_threshold fraction drifted
    drift_share: float             # fraction of features that drifted (0-1)
    n_drifted_features: int
    total_features: int
    drifted_features: list         # column names where drift was detected
    feature_details: dict          # col -> FeatureDriftDetail
    timestamp: str                 # ISO-8601 UTC
    report_html_path: str          # path to Evidently HTML report
    report_json_path: str          # path to JSON copy of this report


# ── Feature columns (must match training) ─────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    "amount",
    "hour",
    "merchant_cat",
    "device_type",
    "sender_age_days",
    "receiver_age_days",
    "txn_count_1h",
    "same_device",
]

# ── Detector ──────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Wraps Evidently's DataDriftPreset to produce a structured DriftReport.

    Usage:
        detector = DriftDetector(reference_df, reports_dir=Path("monitor/reports"))
        report   = detector.detect(current_df)
        print(report["drift_detected"], report["drifted_features"])
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        reports_dir: Path,
        feature_columns: list[str] = FEATURE_COLUMNS,
    ) -> None:
        """
        Args:
            reference_df:    Training / baseline distribution DataFrame.
            reports_dir:     Directory where HTML and JSON reports are saved.
            feature_columns: Columns to monitor. Must exist in both DataFrames.
        """
        missing = [c for c in feature_columns if c not in reference_df.columns]
        if missing:
            raise ValueError(f"Reference data missing columns: {missing}")

        self.reference_df = reference_df[feature_columns].copy()
        self.reports_dir  = reports_dir
        self.feature_columns = feature_columns
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DriftDetector ready | reference rows=%d | features=%d",
            len(reference_df), len(feature_columns),
        )

    def detect(self, current_df: pd.DataFrame) -> DriftReport:
        """
        Run Evidently drift detection on current_df vs the stored reference.

        Args:
            current_df: Live / production DataFrame (same schema as reference).

        Returns:
            DriftReport dict — JSON-serialisable, ready for MLflow logging and
            LangGraph agent reasoning.

        Raises:
            ValueError: if current_df is missing required feature columns.
            RuntimeError: if Evidently report generation fails.
        """
        missing = [c for c in self.feature_columns if c not in current_df.columns]
        if missing:
            raise ValueError(f"Current data missing columns: {missing}")

        current_features = current_df[self.feature_columns].copy()

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        safe_ts   = timestamp.replace(":", "-")
        html_path = self.reports_dir / f"drift_{safe_ts}.html"
        json_path = self.reports_dir / f"drift_{safe_ts}.json"

        logger.info(
            "Running drift detection | current rows=%d | timestamp=%s",
            len(current_df), timestamp,
        )

        try:
            ev_report = Report(metrics=[DataDriftPreset()])
            ev_report.run(
                reference_data=self.reference_df,
                current_data=current_features,
            )
        except Exception as exc:
            logger.exception("Evidently report generation failed: %s", exc)
            raise RuntimeError(f"Drift detection failed: {exc}") from exc

        # ── Parse Evidently output ────────────────────────────────────────────
        try:
            raw: dict = ev_report.as_dict()
            drift_report = self._parse_evidently_output(raw, timestamp)
        except Exception as exc:
            logger.exception("Failed to parse Evidently output: %s", exc)
            raise RuntimeError(f"Report parsing failed: {exc}") from exc

        # ── Save HTML report ──────────────────────────────────────────────────
        try:
            ev_report.save_html(str(html_path))
            logger.info("HTML report saved -> %s", html_path)
        except Exception as exc:
            logger.warning("Could not save HTML report: %s", exc)

        # ── Save JSON report ──────────────────────────────────────────────────
        drift_report["report_html_path"] = str(html_path)
        drift_report["report_json_path"] = str(json_path)

        try:
            json_path.write_text(json.dumps(drift_report, indent=2, default=str))
            logger.info("JSON report saved -> %s", json_path)
        except Exception as exc:
            logger.warning("Could not save JSON report: %s", exc)

        logger.info(
            "Drift detection complete | drift_detected=%s | n_drifted=%d/%d",
            drift_report["drift_detected"],
            drift_report["n_drifted_features"],
            drift_report["total_features"],
        )
        return drift_report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_evidently_output(
        self,
        raw: dict,
        timestamp: str,
    ) -> DriftReport:
        """
        Extract a flat, agent-friendly DriftReport from Evidently's nested output.

        Evidently 0.7.x DataDriftPreset produces two metrics:
          metrics[0] → DatasetDriftMetric   (overall summary)
          metrics[1] → DataDriftTable       (per-column detail)

        We use metrics[1] as it contains both summary AND per-column breakdown.
        """
        metrics = raw.get("metrics", [])

        # Find the metric result that contains drift_by_columns
        per_column_result: dict = {}
        overall_result: dict = {}

        for metric in metrics:
            result = metric.get("result", {})
            if "drift_by_columns" in result:
                per_column_result = result
            elif "dataset_drift" in result and "drift_by_columns" not in result:
                overall_result = result

        # Fall back: if DataDriftPreset only produced one metric dict
        if not per_column_result and overall_result:
            per_column_result = overall_result

        drift_by_columns: dict = per_column_result.get("drift_by_columns", {})

        # Build per-feature details
        feature_details: dict[str, FeatureDriftDetail] = {}
        drifted_features: list[str] = []

        for col in self.feature_columns:
            col_data = drift_by_columns.get(col, {})
            col_drift: bool = bool(col_data.get("drift_detected", False))
            col_score: float = float(col_data.get("drift_score", 1.0))
            stat_test: str   = str(col_data.get("stattest_name", "unknown"))
            threshold: float = float(col_data.get("stattest_threshold", 0.05))

            feature_details[col] = FeatureDriftDetail(
                drift_detected=col_drift,
                drift_score=round(col_score, 8),
                stat_test=stat_test,
                threshold=threshold,
            )
            if col_drift:
                drifted_features.append(col)

        n_total   = len(self.feature_columns)
        n_drifted = len(drifted_features)
        drift_share = n_drifted / n_total if n_total > 0 else 0.0

        # Overall drift_detected: use Evidently's own flag if available,
        # otherwise use > 50% features drifted as heuristic
        dataset_drift: bool = bool(
            per_column_result.get("dataset_drift", drift_share > 0.5)
        )

        return DriftReport(
            drift_detected=dataset_drift,
            drift_share=round(drift_share, 4),
            n_drifted_features=n_drifted,
            total_features=n_total,
            drifted_features=drifted_features,
            feature_details=feature_details,
            timestamp=timestamp,
            report_html_path="",   # filled in by caller
            report_json_path="",   # filled in by caller
        )
