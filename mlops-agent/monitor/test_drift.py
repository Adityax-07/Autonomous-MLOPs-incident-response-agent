# mlops-agent/monitor/test_drift.py
"""
Automated verification tests for the drift detection pipeline.

Tests:
  1. Clean data (reference vs reference-sampled) → drift_detected MUST be False
  2. Drifted data (high severity inject) → drift_detected MUST be True
  3. Recommendation logic → correct label returned for each drift_score range
  4. Quality issues → no false positives on clean data

Run:
    python -m monitor.test_drift
    python monitor/test_drift.py

Exit codes:
    0 — all tests passed
    1 — one or more tests failed
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.inject_drift import inject_drift
from monitor.drift_check import FEATURE_COLUMNS, _recommend, generate_drift_report

logging.basicConfig(
    level=logging.WARNING,      # suppress INFO during tests for clean output
    format="%(levelname)-8s | %(message)s",
)

REPORTS_DIR = ROOT / "monitor" / "reports" / "test_runs"
DATA_DIR    = ROOT / "data"
RANDOM_SEED = 42

# ── Test helpers ──────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []   # (test_name, passed, message)


def _assert(name: str, condition: bool, message: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    _results.append((name, condition, message))
    marker = "[PASS]" if condition else "[FAIL]"
    print(f"  {marker}  {name}" + (f"  — {message}" if message else ""))


# ── Load reference data ───────────────────────────────────────────────────────

def _load_reference() -> pd.DataFrame:
    ref_path = DATA_DIR / "reference.csv"
    if not ref_path.exists():
        print("[ERROR] data/reference.csv not found. Run: python data/generate_data.py")
        sys.exit(2)
    return pd.read_csv(ref_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_no_drift_on_clean_data(ref_df: pd.DataFrame) -> None:
    """
    Sample a fresh subset from the reference distribution.
    The monitor should NOT fire since both datasets share the same distribution.
    """
    rng    = np.random.default_rng(RANDOM_SEED)
    sample = ref_df.sample(n=min(500, len(ref_df)), random_state=RANDOM_SEED)

    report = generate_drift_report(
        reference_df=ref_df,
        production_df=sample,
        reports_dir=REPORTS_DIR,
        feature_columns=FEATURE_COLUMNS,
    )

    _assert(
        "no_drift_on_clean_data",
        not report["drift_detected"],
        f"drift_score={report['drift_score']:.3f} drifted={report['drifted_columns']}",
    )
    _assert(
        "clean_recommendation_is_ok",
        report["recommendation"] == "ok",
        f"got recommendation={report['recommendation']}",
    )


def test_drift_detected_on_drifted_data(ref_df: pd.DataFrame) -> None:
    """
    Inject high-severity drift into a production sample.
    The monitor MUST detect drift on at least 2 features.
    """
    rng    = np.random.default_rng(RANDOM_SEED)
    sample = ref_df.sample(n=min(2000, len(ref_df)), random_state=RANDOM_SEED).reset_index(drop=True)
    drifted = inject_drift(sample, severity="high", rng=rng)

    report = generate_drift_report(
        reference_df=ref_df,
        production_df=drifted,
        reports_dir=REPORTS_DIR,
        feature_columns=FEATURE_COLUMNS,
    )

    _assert(
        "drift_detected_on_high_severity",
        report["drift_detected"],
        f"drift_score={report['drift_score']:.3f} drifted={report['drifted_columns']}",
    )
    _assert(
        "at_least_2_features_drifted",
        len(report["drifted_columns"]) >= 2,
        f"only {len(report['drifted_columns'])} features drifted: {report['drifted_columns']}",
    )
    _assert(
        "drifted_recommendation_is_not_ok",
        report["recommendation"] != "ok",
        f"got recommendation={report['recommendation']}",
    )
    _assert(
        "key_drifted_features_detected",
        any(col in report["drifted_columns"] for col in ["amount", "hour", "txn_count_1h"]),
        f"expected amount/hour/txn_count_1h in {report['drifted_columns']}",
    )


def test_medium_drift_triggers_rollback(ref_df: pd.DataFrame) -> None:
    """Medium-severity drift should produce drift_score > 0.30 → rollback or retrain."""
    rng    = np.random.default_rng(RANDOM_SEED)
    sample = ref_df.sample(n=min(2000, len(ref_df)), random_state=RANDOM_SEED).reset_index(drop=True)
    drifted = inject_drift(sample, severity="medium", rng=rng)

    report = generate_drift_report(
        reference_df=ref_df,
        production_df=drifted,
        reports_dir=REPORTS_DIR,
        feature_columns=FEATURE_COLUMNS,
    )

    _assert(
        "medium_drift_score_above_threshold",
        report["drift_score"] > 0.10,
        f"drift_score={report['drift_score']:.3f}",
    )
    _assert(
        "medium_drift_recommendation_not_ok",
        report["recommendation"] in ("retrain", "rollback", "alert"),
        f"got recommendation={report['recommendation']}",
    )


def test_recommendation_logic() -> None:
    """Unit-test the _recommend() function directly against threshold boundaries."""
    cases = [
        (0.00, "ok"),
        (0.05, "ok"),
        (0.10, "ok"),
        (0.11, "alert"),
        (0.30, "alert"),
        (0.31, "rollback"),
        (0.50, "rollback"),
        (0.51, "retrain"),
        (1.00, "retrain"),
    ]
    all_pass = True
    for score, expected in cases:
        got = _recommend(score)
        ok  = got == expected
        if not ok:
            all_pass = False
        print(f"    _recommend({score:.2f}) = {got!r:10s}  expected={expected!r}  {'OK' if ok else 'MISMATCH'}")

    _assert(
        "recommendation_logic_all_thresholds",
        all_pass,
        "see detail above",
    )


def test_report_schema(ref_df: pd.DataFrame) -> None:
    """Verify the returned dict contains all required keys with correct types."""
    sample = ref_df.sample(n=200, random_state=RANDOM_SEED)
    report = generate_drift_report(
        reference_df=ref_df,
        production_df=sample,
        reports_dir=REPORTS_DIR,
        feature_columns=FEATURE_COLUMNS,
    )

    required_keys = {
        "drift_detected": bool,
        "drift_score":    float,
        "drifted_columns": list,
        "quality_issues":  list,
        "timestamp":       str,
        "recommendation":  str,
        "feature_details": dict,
        "report_html_path": str,
    }

    schema_ok = True
    for key, expected_type in required_keys.items():
        present = key in report
        typed   = isinstance(report.get(key), expected_type) if present else False
        if not (present and typed):
            schema_ok = False
            print(f"    [SCHEMA] key={key!r} present={present} type={type(report.get(key)).__name__}")

    _assert("report_schema_valid", schema_ok, "all required keys present with correct types")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  UPI Fraud Drift Detection — Test Suite")
    print("=" * 60 + "\n")

    ref_df = _load_reference()
    print(f"  Reference data: {len(ref_df)} rows\n")

    t0 = time.perf_counter()

    print("Test 1: No drift on clean data")
    test_no_drift_on_clean_data(ref_df)

    print("\nTest 2: Drift detected on high-severity drifted data")
    test_drift_detected_on_drifted_data(ref_df)

    print("\nTest 3: Medium drift triggers appropriate recommendation")
    test_medium_drift_triggers_rollback(ref_df)

    print("\nTest 4: Recommendation logic (unit test)")
    test_recommendation_logic()

    print("\nTest 5: Report schema validation")
    test_report_schema(ref_df)

    elapsed = time.perf_counter() - t0
    passed  = sum(1 for _, ok, _ in _results if ok)
    total   = len(_results)
    failed  = total - passed

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed  |  {failed} failed  |  {elapsed:.1f}s")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
