# mlops-agent/data/generate_data.py
"""
Generates synthetic UPI fraud detection datasets.

Produces two CSVs:
  - reference.csv  → clean historical data used to train the model
  - production.csv → simulated live data (optionally drifted)

UPI fraud signals used:
  • Large amounts at odd hours
  • Very new sender/receiver accounts
  • High transaction velocity in 1 hour
  • Mismatched device from usual pattern
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "amount",            # Transaction amount in INR (1 – 200 000)
    "hour",              # Hour of day the txn was initiated (0–23)
    "merchant_cat",      # Merchant category code (0–9)
    "device_type",       # 0 = mobile UPI, 1 = web, 2 = POS
    "sender_age_days",   # Age of sender VPA account in days
    "receiver_age_days", # Age of receiver VPA account in days
    "txn_count_1h",      # Number of UPI txns by sender in last 1 hour
    "same_device",       # 1 if device matches sender's registered device
]

LABEL_COLUMN = "is_fraud"

RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_legit_rows(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate n legitimate UPI transactions."""
    return pd.DataFrame({
        "amount":            rng.lognormal(mean=8.5, sigma=1.2, size=n).clip(1, 200_000),
        "hour":              rng.integers(6, 23, size=n),      # active hours
        "merchant_cat":      rng.integers(0, 10, size=n),
        "device_type":       rng.choice([0, 1, 2], p=[0.75, 0.15, 0.10], size=n),
        "sender_age_days":   rng.integers(30, 2000, size=n),   # established accounts
        "receiver_age_days": rng.integers(30, 2000, size=n),
        "txn_count_1h":      rng.integers(0, 5, size=n),       # low velocity
        "same_device":       rng.choice([0, 1], p=[0.05, 0.95], size=n),
        LABEL_COLUMN:        np.zeros(n, dtype=int),
    })


def _generate_fraud_rows(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate n fraudulent UPI transactions with realistic fraud patterns."""
    return pd.DataFrame({
        "amount":            rng.lognormal(mean=11.0, sigma=0.8, size=n).clip(5_000, 200_000),
        "hour":              rng.choice(list(range(0, 6)) + list(range(22, 24)), size=n),  # odd hours
        "merchant_cat":      rng.integers(0, 10, size=n),
        "device_type":       rng.choice([0, 1, 2], p=[0.50, 0.45, 0.05], size=n),
        "sender_age_days":   rng.integers(0, 15, size=n),      # brand-new accounts
        "receiver_age_days": rng.integers(0, 10, size=n),
        "txn_count_1h":      rng.integers(8, 30, size=n),      # burst velocity
        "same_device":       rng.choice([0, 1], p=[0.90, 0.10], size=n),  # unfamiliar device
        LABEL_COLUMN:        np.ones(n, dtype=int),
    })


def generate_dataset(
    n_samples: int,
    fraud_rate: float,
    rng: np.random.Generator,
    drift: bool = False,
) -> pd.DataFrame:
    """
    Generate a complete labelled dataset.

    Args:
        n_samples:  Total number of rows.
        fraud_rate: Fraction that should be fraudulent (e.g. 0.03).
        rng:        Seeded NumPy Generator for reproducibility.
        drift:      If True, inject distribution drift into legitimate rows
                    (simulates a real-world covariate shift scenario).

    Returns:
        Shuffled DataFrame with FEATURE_COLUMNS + LABEL_COLUMN.
    """
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    legit_df = _generate_legit_rows(n_legit, rng)
    fraud_df = _generate_fraud_rows(n_fraud, rng)

    if drift:
        # Simulate a festival season: amounts spike, more late-night txns
        legit_df["amount"] *= rng.uniform(2.0, 4.0, size=n_legit)
        legit_df["amount"] = legit_df["amount"].clip(1, 200_000)
        legit_df["hour"] = rng.choice(list(range(18, 24)), size=n_legit)
        legit_df["txn_count_1h"] = rng.integers(4, 15, size=n_legit)

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    return df.sample(frac=1, random_state=int(rng.integers(0, 9999))).reset_index(drop=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UPI fraud datasets")
    parser.add_argument("--reference-size", type=int, default=10_000,
                        help="Number of rows in reference (training) dataset")
    parser.add_argument("--production-size", type=int, default=2_000,
                        help="Number of rows in production (live) dataset")
    parser.add_argument("--fraud-rate", type=float, default=0.03,
                        help="Fraction of fraudulent transactions")
    parser.add_argument("--drift", action="store_true",
                        help="Inject distribution drift into production data")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Output directory for CSV files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Generating reference dataset  ({args.reference_size:,} rows)...")
    ref_df = generate_dataset(args.reference_size, args.fraud_rate, rng, drift=False)
    ref_path = out_dir / "reference.csv"
    ref_df.to_csv(ref_path, index=False)
    print(f"  [OK] Saved -> {ref_path}  |  fraud rows: {ref_df[LABEL_COLUMN].sum()}")

    print(f"Generating production dataset ({args.production_size:,} rows, drift={args.drift})...")
    prod_df = generate_dataset(args.production_size, args.fraud_rate, rng, drift=args.drift)
    prod_path = out_dir / "production.csv"
    prod_df.to_csv(prod_path, index=False)
    print(f"  [OK] Saved -> {prod_path}  |  fraud rows: {prod_df[LABEL_COLUMN].sum()}")

    print("\nColumn summary (reference):")
    print(ref_df[FEATURE_COLUMNS].describe().round(2).to_string())


if __name__ == "__main__":
    main()
