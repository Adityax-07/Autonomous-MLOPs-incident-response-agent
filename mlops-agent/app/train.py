# mlops-agent/app/train.py
"""
Trains an XGBoost binary classifier for UPI fraud detection.

Workflow:
  1. Load reference.csv from data/
  2. Train/val split (80/20, stratified)
  3. Train XGBoost with class imbalance handling
  4. Evaluate and print classification report
  5. Save model artefacts to models/
       models/model.joblib   → trained pipeline
       models/metadata.json  → feature list, thresholds, metrics

Run:
    python -m app.train
  or from repo root:
    python app/train.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent          # mlops-agent/
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ── Feature / label config ────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "merchant_cat",
    "device_type",
    "sender_age_days",
    "receiver_age_days",
    "txn_count_1h",
    "same_device",
]
LABEL_COLUMN = "is_fraud"

# Decision threshold tuned for fraud: prefer recall over precision
DECISION_THRESHOLD = 0.40


# ── Model definition ──────────────────────────────────────────────────────────

def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """
    Returns a sklearn Pipeline:
      StandardScaler → XGBClassifier

    scale_pos_weight compensates for class imbalance
    (n_negative / n_positive).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",      # fast on CPU; switch to "gpu_hist" if GPU present
        )),
    ])


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    ref_path = DATA_DIR / "reference.csv"
    if not ref_path.exists():
        print(
            f"[ERROR] {ref_path} not found.\n"
            "Run first:  python data/generate_data.py"
        )
        sys.exit(1)

    print(f"Loading data from {ref_path} ...")
    df = pd.read_csv(ref_path)

    missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in dataset: {missing}")
        sys.exit(1)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMN].values

    print(f"Dataset shape : {X.shape}")
    print(f"Fraud rate    : {y.mean() * 100:.2f}%  ({y.sum()} / {len(y)} rows)")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f}  (neg={n_neg}, pos={n_pos})")

    pipeline = build_pipeline(scale_pos_weight)

    print("\nTraining XGBoost ...")
    t0 = time.perf_counter()

    # XGBoost early stopping needs the raw classifier, not the Pipeline wrapper
    # So we fit the scaler first, transform, then fit XGB with eval_set
    pipeline.named_steps["scaler"].fit(X_train)
    X_train_sc = pipeline.named_steps["scaler"].transform(X_train)
    X_val_sc   = pipeline.named_steps["scaler"].transform(X_val)

    pipeline.named_steps["clf"].fit(
        X_train_sc, y_train,
        eval_set=[(X_val_sc, y_val)],
        verbose=50,
    )

    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.1f}s")

    # ── Evaluation ────────────────────────────────────────────────────────────

    proba_val = pipeline.named_steps["clf"].predict_proba(X_val_sc)[:, 1]
    y_pred    = (proba_val >= DECISION_THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_val, proba_val)
    pr_auc  = average_precision_score(y_val, proba_val)

    print("\n-- Validation metrics ------------------------------------------")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"\nClassification report (threshold = {DECISION_THRESHOLD}):")
    print(classification_report(y_val, y_pred, target_names=["legit", "fraud"]))

    # -- Persist --------------------------------------------------------------

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    metadata: dict = {
        "feature_columns": FEATURE_COLUMNS,
        "label_column":    LABEL_COLUMN,
        "decision_threshold": DECISION_THRESHOLD,
        "n_estimators":    pipeline.named_steps["clf"].n_estimators,
        "train_rows":      int(len(X_train)),
        "val_rows":        int(len(X_val)),
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc":  round(pr_auc, 4),
        },
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved -> {METADATA_PATH}")


if __name__ == "__main__":
    train()
