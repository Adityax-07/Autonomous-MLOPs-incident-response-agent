# mlops-agent/data/inject_drift.py
"""
Standalone drift injector — reads data/reference.csv (clean baseline),
applies controlled covariate shift, and writes data/drifted_data.csv.

Three features are shifted to simulate realistic UPI fraud signal drift:

  amount              (your spec: transaction_amount)
    → Log-normal mean shifted upward — unusually large transactions
      e.g. a mule account receiving rapid high-value transfers

  hour                (your spec: hour_of_day)
    → Concentrated into 2am-5am window — classic fraud-ring operating hours
      (low bank staff, delayed alerts, sleeping victims)

  txn_count_1h        (your spec: failed_attempts_last_hour)
    → Multiplied 3x — burst velocity from scripted attack tooling

Severity levels change the magnitude of each shift:
  low    → subtle drift, borderline detectable
  medium → clear drift on 2-3 features
  high   → severe drift on all 3 features, obvious to the monitor

Usage:
    python data/inject_drift.py                      # default: medium severity
    python data/inject_drift.py --severity high
    python data/inject_drift.py --severity low --input data/production.csv
    python data/inject_drift.py --rows 1000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
INPUT_PATH = DATA_DIR / "reference.csv"   # clean source
OUTPUT_PATH = DATA_DIR / "drifted_data.csv"

FEATURE_COLUMNS = [
    "amount", "hour", "merchant_cat", "device_type",
    "sender_age_days", "receiver_age_days", "txn_count_1h", "same_device",
]

# ── Severity configuration ────────────────────────────────────────────────────
# Each severity is a dict of (feature -> shift params).
# amount      : lognormal_mean_add  — added to log-space mean
# hour        : night_prob          — probability of replacing hour with 2-5am
# txn_count   : multiplier          — txn_count_1h *= multiplier

SEVERITY_CONFIG: dict[str, dict] = {
    "low": {
        "amount_log_shift":  0.5,   # ~1.65x raw mean
        "night_prob":        0.25,
        "txn_multiplier":    1.5,
    },
    "medium": {
        "amount_log_shift":  1.0,   # ~2.7x raw mean
        "night_prob":        0.55,
        "txn_multiplier":    2.0,
    },
    "high": {
        "amount_log_shift":  1.8,   # ~6x raw mean
        "night_prob":        0.85,
        "txn_multiplier":    3.0,
    },
}

RANDOM_SEED = 77


def inject_drift(
    df: pd.DataFrame,
    severity: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Apply covariate shift to three key UPI fraud features.

    Args:
        df:       Source DataFrame (not modified in place).
        severity: "low" | "medium" | "high"
        rng:      Seeded NumPy Generator.

    Returns:
        New DataFrame with drifted distributions.

    Raises:
        ValueError: if severity is not recognised.
    """
    if severity not in SEVERITY_CONFIG:
        raise ValueError(
            f"Unknown severity '{severity}'. Choose from: {list(SEVERITY_CONFIG.keys())}"
        )

    cfg = SEVERITY_CONFIG[severity]
    n   = len(df)
    out = df.copy()

    # ── 1. amount: shift lognormal mean upward ────────────────────────────────
    # Add cfg["amount_log_shift"] to log(amount) then exponentiate.
    # This models a sudden spike in transaction sizes without clamping small txns.
    log_amounts = np.log(out["amount"].clip(lower=1))
    shifted_log = log_amounts + cfg["amount_log_shift"] + rng.normal(0, 0.15, n)
    out["amount"] = np.exp(shifted_log).clip(1, 200_000).round(2)

    # ── 2. hour: concentrate into 2am-5am ─────────────────────────────────────
    # Replace a fraction of hours with the fraud-ring operating window.
    night_mask = rng.random(n) < cfg["night_prob"]
    out.loc[night_mask, "hour"] = rng.integers(2, 6, size=int(night_mask.sum()))

    # ── 3. txn_count_1h: multiply by severity factor ──────────────────────────
    # Simulates scripted attack bursts — attacker fires many rapid micro-txns.
    noise = rng.uniform(0.8, 1.2, n)
    out["txn_count_1h"] = (
        (out["txn_count_1h"] * cfg["txn_multiplier"] * noise)
        .round()
        .astype(int)
        .clip(0, 30)
    )

    return out


def _distribution_summary(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Print a before/after mean comparison for the three drifted features."""
    targets = ["amount", "hour", "txn_count_1h"]

    print("\n" + "=" * 65)
    print("  Feature distribution shift (mean values)")
    print("=" * 65)
    print(f"  {'Feature':<22} {'Before':>10} {'After':>10} {'Delta':>8}  {'Delta %':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8}  {'-'*8}")
    for col in targets:
        b = before[col].mean()
        a = after[col].mean()
        delta = a - b
        pct   = (delta / (b + 1e-9)) * 100
        print(f"  {col:<22} {b:>10.2f} {a:>10.2f} {delta:>+8.2f}  {pct:>+7.1f}%")
    print("=" * 65)

    print("\n  Distribution details (mean / std / min / max):")
    print(f"  {'Feature':<22} {'Stat':<6} {'Before':>12} {'After':>12}")
    print(f"  {'-'*22} {'-'*6} {'-'*12} {'-'*12}")
    for col in targets:
        for stat in ["mean", "std", "min", "max"]:
            b = getattr(before[col], stat)()
            a = getattr(after[col], stat)()
            print(f"  {col if stat=='mean' else '':<22} {stat:<6} {b:>12.2f} {a:>12.2f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject covariate drift into UPI transaction data"
    )
    parser.add_argument(
        "--severity", choices=["low", "medium", "high"], default="medium",
        help="Drift intensity (default: medium)",
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT_PATH,
        help="Source CSV (default: data/reference.csv)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help="Output CSV (default: data/drifted_data.csv)",
    )
    parser.add_argument(
        "--rows", type=int, default=2000,
        help="Number of rows to sample from input (default: 2000)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    try:
        source_df = pd.read_csv(args.input)
    except Exception as exc:
        logger.exception("Failed to read input CSV: %s", exc)
        sys.exit(1)

    missing = [c for c in FEATURE_COLUMNS if c not in source_df.columns]
    if missing:
        logger.error("Input CSV missing columns: %s", missing)
        sys.exit(1)

    # Sample n rows from source
    n = min(args.rows, len(source_df))
    rng = np.random.default_rng(RANDOM_SEED)
    sample_df = source_df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)

    logger.info(
        "Injecting '%s' severity drift into %d rows from %s ...",
        args.severity, n, args.input,
    )

    drifted_df = inject_drift(sample_df, severity=args.severity, rng=rng)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        drifted_df.to_csv(args.output, index=False)
    except Exception as exc:
        logger.exception("Failed to write output CSV: %s", exc)
        sys.exit(1)

    _distribution_summary(before=sample_df, after=drifted_df)

    print(f"\n  [OK] Drifted data written -> {args.output}")
    print(f"       {n} rows | severity={args.severity}")
    print(f"\n  Next step: python -m monitor.drift_check")


if __name__ == "__main__":
    main()
