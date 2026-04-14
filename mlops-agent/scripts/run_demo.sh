#!/usr/bin/env bash
# mlops-agent/scripts/run_demo.sh
#
# Full demo loop: start stack, inject drift, watch agent recover in Grafana.
# Works on macOS and Linux. Requires Docker Desktop (or Docker Engine + Compose).
#
# Usage:
#   chmod +x scripts/run_demo.sh
#   bash scripts/run_demo.sh
#
# What it does:
#   1. Verify prerequisites
#   2. Generate training data + initial model (if not already done)
#   3. Build images + start Docker Compose stack
#   4. Wait for FastAPI health check to pass
#   5. Warm up the prediction log with 150 synthetic requests
#   6. Inject medium-severity drift into production data
#   7. Wait for monitor scheduler to detect drift (up to 60s)
#   8. Wait for agent to fire and act (up to 120s)
#   9. Open Grafana in the browser
#  10. Tail agent logs so you can watch the decision in real time
#
# To stop: Ctrl+C (signals are forwarded to docker compose)

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[demo]${NC} $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
fail()  { echo -e "${RED}[fail]${NC} $*"; exit 1; }

# ── Navigate to repo root ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
info "Working directory: $ROOT"

# ── Step 1: Check prerequisites ───────────────────────────────────────────────
info "Step 1/9 — Checking prerequisites"

command -v docker  >/dev/null 2>&1 || fail "docker not found. Install Docker Desktop."
command -v python  >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || fail "python not found."
PYTHON=$(command -v python3 2>/dev/null || command -v python)

# Docker Compose v2 (docker compose) vs v1 (docker-compose)
if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
else
    fail "docker compose not found. Install Docker Compose v2."
fi
ok "Docker and Compose found. Using: $DC"

# ── Step 2: Generate data + train model ───────────────────────────────────────
info "Step 2/9 — Generating data and initial model"

if [ ! -f "data/reference.csv" ]; then
    info "Generating reference.csv and production.csv ..."
    $PYTHON data/generate_data.py
    ok "Data generated."
else
    ok "data/reference.csv already exists — skipping generation."
fi

if [ ! -f "models/model.joblib" ]; then
    info "Training initial XGBoost model ..."
    $PYTHON -m app.train
    ok "Model trained and saved to models/model.joblib"
else
    ok "models/model.joblib already exists — skipping training."
fi

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from .env.example. Set SLACK_WEBHOOK_URL / ANTHROPIC_API_KEY if needed."
fi

# ── Step 3: Build and start Docker Compose ────────────────────────────────────
info "Step 3/9 — Building images and starting stack (this may take 2-3 minutes on first run)"

$DC up -d --build
ok "Docker Compose stack started."

# ── Step 4: Wait for FastAPI health check ─────────────────────────────────────
info "Step 4/9 — Waiting for FastAPI server to be healthy"

MAX_WAIT=120
ELAPSED=0
until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        fail "FastAPI server did not become healthy within ${MAX_WAIT}s. Check: $DC logs fastapi-server"
    fi
    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo ""
ok "FastAPI server is healthy at http://localhost:8000"

# ── Step 5: Warm up predictions log ───────────────────────────────────────────
info "Step 5/9 — Sending 150 baseline predictions to warm up predictions.log"

$PYTHON - <<'PYEOF'
import random
import sys

try:
    import requests
except ImportError:
    print("requests not installed — skipping warmup (run: pip install requests)")
    sys.exit(0)

url = "http://localhost:8000/predict"
sent = 0
for _ in range(150):
    try:
        resp = requests.post(url, json={
            "amount":            round(random.uniform(100, 5000), 2),
            "hour":              random.randint(9, 21),
            "merchant_cat":      random.randint(0, 4),
            "device_type":       random.randint(0, 2),
            "sender_age_days":   random.randint(30, 2000),
            "receiver_age_days": random.randint(30, 2000),
            "txn_count_1h":      random.randint(1, 5),
            "same_device":       random.randint(0, 1),
        }, timeout=5)
        if resp.status_code == 200:
            sent += 1
    except Exception:
        pass

print(f"Sent {sent}/150 predictions. predictions.log has at least {sent} rows.")
PYEOF

ok "Predictions log warmed up."

# ── Step 6: Inject medium-severity drift ──────────────────────────────────────
info "Step 6/9 — Injecting MEDIUM-severity drift (amount +1.0, night hours, 2x velocity)"

$PYTHON data/inject_drift.py --severity medium
ok "Drifted data written to data/drifted_data.csv"

# Force the scheduler to use drifted_data.csv as the production distribution
# by copying it over production.csv (scheduler reads predictions.log,
# but for the demo we trigger a manual run with the drifted data)
info "Running manual drift check against drifted data ..."
$PYTHON - <<'PYEOF'
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
from monitor.drift_check import FEATURE_COLUMNS, generate_drift_report

ref_df     = pd.read_csv("data/reference.csv")
drifted_df = pd.read_csv("data/drifted_data.csv")

report = generate_drift_report(
    reference_df=ref_df,
    production_df=drifted_df,
    feature_columns=FEATURE_COLUMNS,
)

Path("monitor").mkdir(exist_ok=True)
Path("monitor/latest_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"Drift report written | drift_score={report['drift_score']:.1%} | recommendation={report['recommendation']}")
PYEOF

ok "Drift report written to monitor/latest_report.json"

# ── Step 7: Wait for agent to pick up the report ──────────────────────────────
info "Step 7/9 — Waiting up to 120s for agent-runner to fire (interval=300s in Docker, but first run is immediate)"

# The agent-runner in Docker runs every 300s, but the first run fires immediately.
# Wait up to 120s for a decision to appear in logs.
MAX_WAIT=120
ELAPSED=0
DECISION_FOUND=false
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if $DC logs agent-runner 2>/dev/null | grep -q "Decision="; then
        DECISION_FOUND=true
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo -n "."
done
echo ""

if [ "$DECISION_FOUND" = "true" ]; then
    DECISION_LINE=$($DC logs agent-runner 2>/dev/null | grep "Decision=" | tail -1)
    ok "Agent fired! $DECISION_LINE"
else
    warn "Agent decision not seen in logs yet (may still be running). Check: $DC logs agent-runner"
fi

# ── Step 8: Show Prometheus metrics ───────────────────────────────────────────
info "Step 8/9 — Checking Prometheus metrics from FastAPI /metrics"
echo ""
curl -s http://localhost:8000/metrics | grep -E "^(model_drift|model_accuracy|agent_decisions|agent_retrains|predictions_total)" | sort
echo ""

# ── Step 9: Open Grafana ───────────────────────────────────────────────────────
info "Step 9/9 — Opening Grafana dashboard"
GRAFANA_URL="http://localhost:3000"

# Cross-platform browser open
if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$GRAFANA_URL" >/dev/null 2>&1 &   # Linux
elif command -v open >/dev/null 2>&1; then
    open "$GRAFANA_URL"                           # macOS
else
    warn "Could not open browser automatically. Open manually: $GRAFANA_URL"
fi

ok "Grafana: $GRAFANA_URL (login: admin / admin)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Demo running! Key URLs:${NC}"
echo -e "  FastAPI docs  : http://localhost:8000/docs"
echo -e "  FastAPI metrics: http://localhost:8000/metrics"
echo -e "  MLflow UI     : http://localhost:5000"
echo -e "  Prometheus    : http://localhost:9090"
echo -e "  Grafana       : http://localhost:3000  (admin/admin)"
echo -e "${GREEN}============================================================${NC}"
echo ""
info "Tailing agent-runner logs. Press Ctrl+C to stop tailing (stack keeps running)."
info "To stop the full stack: $DC down"
echo ""

# Tail agent-runner logs so the user sees decisions in real time
$DC logs -f agent-runner
