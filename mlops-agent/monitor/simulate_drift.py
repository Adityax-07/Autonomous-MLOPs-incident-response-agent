# mlops-agent/monitor/simulate_drift.py
"""
Drift simulator for Phase 2 testing.

Overwrites data/production.csv with a drifted version so that the
drift detector fires. Simulates a real-world scenario:

  "Festival Season Drift" — during Diwali / IPL season:
    • Transaction amounts spike 3-5x (big purchases, P2P transfers)
    • More transactions happen between 10 PM and 4 AM
    • New UPI users onboard in bulk (sender_age_days drops)
    • Transaction velocity increases (txn_count_1h spikes)
    • More web-based transactions (device_type shifts toward 1)

This kind of covariate shift fools a fraud model trained on normal data:
legitimate high-value night transactions get misclassified as fraud.

Usage:
    python -m monitor.simulate_drift              # inject drift
    python -m monitor.simulate_drift --reset      # restore clean data
    python -m monitor.simulate_drift --intensity high
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
PROD_PATH = DATA_DIR / "production.csv"
REF_PATH  = DATA_DIR / "reference.csv"

FEATURE_COLUMNS = [
    "amount", "hour", "merchant_cat", "device_type",
    "sender_age_days", "receiver_age_days", "txn_count_1h", "same_device",
]
LABEL_COLUMN = "is_fraud"

RANDOM_SEED = 99


def inject_drift(df: pd.DataFrame, intensity: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Apply covariate shift to a production DataFrame.

    Args:
        df:        Production DataFrame (will not be modified in place).
        intensity: "low" | "medium" | "high"
        rng:       Seeded random generator.

    Returns:
        New DataFrame with shifted feature distributions.
    """
    drift_params = {
        "low":    dict(amount_scale=1.5, hour_night_prob=0.25, age_max=20,  vel_add=3),
        "medium": dict(amount_scale=3.0, hour_night_prob=0.50, age_max=10,  vel_add=7),
        "high":   dict(amount_scale=5.0, hour_night_prob=0.75, age_max=5,   vel_add=12),
    }
    if intensity not in drift_params:
        raise ValueError(f"intensity must be one of {list(drift_params.keys())}")

    p = drift_params[intensity]
    n = len(df)
    drifted = df.copy()

    # 1. Amount spike — log-normal shift upward
    scale_factors = rng.uniform(p["amount_scale"] * 0.8, p["amount_scale"] * 1.2, size=n)
    drifted["amount"] = (drifted["amount"] * scale_factors).clip(1, 200_000).round(2)

    # 2. Hour shift — more night-time transactions
    night_mask = rng.random(n) < p["hour_night_prob"]
    night_hours = rng.choice(list(range(22, 24)) + list(range(0, 5)), size=n)
    drifted.loc[night_mask, "hour"] = night_hours[night_mask]

    # 3. Sender age drops — influx of new users
    new_user_mask = rng.random(n) < 0.60
    drifted.loc[new_user_mask, "sender_age_days"] = rng.integers(0, p["age_max"], size=int(new_user_mask.sum()))

    # 4. Velocity spike
    drifted["txn_count_1h"] = (drifted["txn_count_1h"] + rng.integers(0, p["vel_add"], size=n)).clip(0, 30)

    # 5. Device type shift toward web
    web_mask = rng.random(n) < 0.35
    drifted.loc[web_mask, "device_type"] = 1

    return drifted


def restore_clean(n_rows: int = 2000, fraud_rate: float = 0.03) -> None:
    """Regenerate a clean (no-drift) production CSV from the reference distribution."""
    if not REF_PATH.exists():
        logger.error("Reference data not found at %s. Run generate_data.py first.", REF_PATH)
        sys.exit(1)

    ref_df = pd.read_csv(REF_PATH)
    sample = ref_df.sample(n=min(n_rows, len(ref_df)), random_state=RANDOM_SEED).reset_index(drop=True)
    sample.to_csv(PROD_PATH, index=False)
    logger.info("Clean production data restored -> %s  (%d rows)", PROD_PATH, len(sample))


def main() -> None:
    parser = argparse.ArgumentParser(description="UPI fraud drift simulator")
    parser.add_argument(
        "--reset", action="store_true",
        help="Remove drift: restore production.csv to the clean reference distribution",
    )
    parser.add_argument(
        "--intensity", choices=["low", "medium", "high"], default="high",
        help="Drift intensity level (default: high)",
    )
    parser.add_argument(
        "--rows", type=int, default=2000,
        help="Number of production rows to generate (default: 2000)",
    )
    args = parser.parse_args()

    if args.reset:
        restore_clean(n_rows=args.rows)
        return

    if not PROD_PATH.exists():
        logger.error(
            "Production CSV not found at %s. Run generate_data.py first.", PROD_PATH
        )
        sys.exit(1)

    try:
        prod_df = pd.read_csv(PROD_PATH)
    except Exception as exc:
        logger.exception("Failed to read %s: %s", PROD_PATH, exc)
        sys.exit(1)

    rng = np.random.default_rng(RANDOM_SEED)

    logger.info(
        "Injecting '%s' intensity drift into %s (%d rows)...",
        args.intensity, PROD_PATH, len(prod_df),
    )

    drifted_df = inject_drift(prod_df, intensity=args.intensity, rng=rng)

    try:
        drifted_df.to_csv(PROD_PATH, index=False)
    except Exception as exc:
        logger.exception("Failed to write drifted CSV: %s", exc)
        sys.exit(1)

    # Print a comparison summary
    print("\nFeature distribution shift (mean values):")
    print(f"{'Feature':<22} {'Before':>10} {'After':>10} {'Delta %':>10}")
    print("-" * 56)
    for col in FEATURE_COLUMNS:
        if col in prod_df.columns and col in drifted_df.columns:
            before = prod_df[col].mean()
            after  = drifted_df[col].mean()
            delta  = ((after - before) / (before + 1e-9)) * 100
            print(f"{col:<22} {before:>10.2f} {after:>10.2f} {delta:>+9.1f}%")

    print(f"\n[OK] Drifted production data written -> {PROD_PATH}")
    print(f"     Run `python -m monitor.run_monitor` to detect this drift.")


if __name__ == "__main__":
    main()
