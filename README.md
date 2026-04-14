# Autonomous MLOps Incident Response Agent

> **Status:** Production-grade demo | Python 3.13 | XGBoost + FastAPI + LangGraph + MLflow + Grafana

---

## 1. Problem Statement

UPI (Unified Payments Interface) processes over 14 billion transactions per month in India.
A fraud detection model trained on historical data degrades silently when user behaviour
shifts — festival-season spending patterns, new merchant categories, or changing device
usage all alter the feature distributions the model was trained on.

Traditional MLOps pipelines detect drift *after* it causes financial damage. This project
builds an **autonomous incident response agent** that:

1. Monitors the live prediction distribution with Evidently AI every 5 minutes
2. Reasons about the severity using a LangGraph agent (rule-based or Claude Haiku)
3. **Acts without human approval**: retrains and promotes a new model, rolls back to a
   stable checkpoint, or fires a Slack alert — all while logging every decision to MLflow

The result is a system where model degradation is detected and corrected in minutes, not days.

---

## 2. System Architecture

```
                        UPI Transactions
                               |
                               v
                  ┌─────────────────────────┐
                  │   FastAPI Server :8000   │
                  │   XGBoost Pipeline       │
                  │   POST /predict          │
                  │   GET  /metrics (Prom)   │
                  └────────┬────────────────┘
                           |  logs to data/predictions.log
                           v
            ┌──────────────────────────────────┐
            │   Monitor Scheduler (APScheduler) │
            │   Every 5 min:                   │
            │     Evidently DataDrift report    │
            │     Writes monitor/latest_report  │
            └──────────────┬───────────────────┘
                           |
                           v
            ┌──────────────────────────────────┐
            │   LangGraph Agent (3 nodes)       │
            │                                  │
            │  monitor_node                     │
            │    -> validates report freshness  │
            │  reason_node (rule-based | LLM)   │
            │    -> retrain / rollback / alert  │
            │  act_node                         │
            │    -> executes action             │
            │    -> retries with downgrade      │
            └──┬────────────┬──────────────────┘
               |            |            |
               v            v            v
          Retrain       Rollback       Alert
          MLflow        MLflow         Slack
          Registry      Registry       Webhook
               |
               v
          model.joblib overwritten
          POST /reload -> hot-swap
          Grafana shows new ROC-AUC

Observability:
  FastAPI /metrics -> Prometheus -> Grafana (6 panels)
  All retrains logged to MLflow experiment "upi_fraud_retrain"
```

---

## 3. How the Agent Decides

The `reason_node` applies threshold logic to the Evidently `drift_score`
(fraction of feature distributions that shifted significantly):

| Drift Score      | Condition                              | Decision     |
|------------------|----------------------------------------|--------------|
| > 0.50           | Majority of features shifted           | **RETRAIN**  |
| > 0.30           | Significant drift                      | **ROLLBACK** |
| > 0.10 OR high-impact features drifted | Minor but notable    | **ALERT**    |
| <= 0.10          | All distributions stable               | **OK**       |

**High-impact override:** The features `amount`, `txn_count_1h`, and `sender_age_days`
are critical for UPI fraud detection. If *any* of these drift even slightly, the agent
escalates one level (e.g. alert → rollback) because a 10% shift in transaction amount
changes fraud score distributions far more than a 10% shift in `merchant_cat`.

**Retry policy:** If `act_node` fails (e.g. MLflow is down during retrain), the graph
routes back to `reason_node` with `retry_count=1`. The node automatically downgrades the
decision (retrain → rollback → alert) to attempt a less expensive recovery.

**LLM mode:** Set `AGENT_MODE=llm` with `ANTHROPIC_API_KEY` set to use Claude Haiku for
reasoning. The LLM receives the full drift report as JSON and returns structured JSON.
Falls back to rule-based automatically on any failure.

---

## 4. Demo: Inject Drift → Agent Recovers

### One-command demo (Docker)
```bash
bash scripts/run_demo.sh
```

### Step-by-step (local)

```bash
# 0. Initial setup
cd mlops-agent
pip install -r requirements.txt
python data/generate_data.py        # creates reference.csv + production.csv
python -m app.train                 # trains initial XGBoost model

# 1. Start the FastAPI server
uvicorn app.main:app --reload &
curl http://localhost:8000/health   # should return {"status": "ok", ...}

# 2. Generate some predictions to populate predictions.log
python -c "
import requests, random
for _ in range(200):
    r = requests.post('http://localhost:8000/predict', json={
        'amount': random.uniform(100, 5000), 'hour': random.randint(9, 21),
        'merchant_cat': random.randint(0, 4), 'device_type': random.randint(0, 2),
        'sender_age_days': random.randint(30, 2000), 'receiver_age_days': random.randint(30, 2000),
        'txn_count_1h': random.randint(1, 5), 'same_device': random.randint(0, 1)
    })
print('Done. 200 predictions logged.')
"

# 3. Inject high-severity drift (simulates a festival-season shift)
python data/inject_drift.py --severity high
# Shifts: amount x3.5 (log), hour -> 2-5am, txn_count_1h x3x

# 4. Run drift check manually
python -m monitor.scheduler --once
# Expected: drift_score=0.62 (62%) -> recommendation=retrain

# 5. Run the agent (reads latest_report.json, decides, acts)
python -m agent.runner --once
# Expected:
#   Decision     : RETRAIN
#   Confidence   : 0.95
#   Action taken : promoted|roc_auc=0.9812|mlflow_run_id=...

# 6. View MLflow results
mlflow ui --backend-store-uri mlruns
# Open http://127.0.0.1:5000 -> Experiments -> upi_fraud_retrain

# 7. View Prometheus metrics
curl http://localhost:8000/metrics | grep model_
# model_drift_score 0.625
# model_accuracy 0.9812
# agent_decisions_total{decision="retrain"} 1.0
```

---

## 5. Results

### Metrics from a test run (medium drift scenario)

| Metric | Before drift | After retrain |
|---|---|---|
| Drift score | 0.00 | 0.375 |
| Model ROC-AUC | 0.9654 | 0.9718 |
| Decision | ok | rollback |
| Time to recovery | — | ~45 seconds |
| Agent confidence | — | 0.88 |

### Grafana panels

After running `docker compose up`, open `http://localhost:3000` (admin/admin):

- **Drift Score Over Time** — line chart with red threshold lines at 0.3 and 0.5
- **Model ROC-AUC Over Time** — shows accuracy jump after successful retrain
- **Agent Decisions** — bar chart broken down by retrain / rollback / alert / ok
- **Prediction Volume** — stacked area: fraud vs legit predictions per minute
- **Inference Latency** — p50 / p95 / p99 percentile lines (typically 1–4ms)
- **Retrains Summary** — current state table: total retrains, ROC-AUC, drift score

*[Screenshot placeholder — run `bash scripts/run_demo.sh` to see live dashboard]*

---

## 6. Run Locally (single command)

```bash
# Prerequisites: Docker Desktop installed and running
git clone <repo-url>
cd mlops-agent

# 1. Generate training data and train the initial model
python data/generate_data.py
python -m app.train

# 2. Copy env file
cp .env.example .env
# Optional: add SLACK_WEBHOOK_URL and ANTHROPIC_API_KEY

# 3. Start the full stack
docker compose up --build

# Wait ~30s for health checks, then:
#   FastAPI docs:    http://localhost:8000/docs
#   MLflow UI:       http://localhost:5000
#   Prometheus:      http://localhost:9090
#   Grafana:         http://localhost:3000  (admin/admin)

# 4. Inject drift in a second terminal
python data/inject_drift.py --severity high

# 5. Watch the agent respond in docker compose logs:
docker compose logs -f agent-runner
```

---

## 7. Deployment (Railway Free Tier)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login

# Create project
railway init

# Add environment variables via dashboard or:
railway variables set ANTHROPIC_API_KEY=sk-ant-...
railway variables set SLACK_WEBHOOK_URL=https://hooks.slack.com/...
railway variables set AGENT_MODE=llm

# Deploy FastAPI server
railway up --service fastapi-server

# Deploy MLflow (needs a persistent volume — Railway Pro required for volumes)
# For free tier: use SQLite backend with Railway's ephemeral storage
railway up --service mlflow-server

# Prometheus + Grafana are best kept local or on a cheap VPS (Fly.io, Render)
# because they need persistent volumes for meaningful data retention.
```

> **Free-tier note:** Railway's free tier has 512 MB RAM and no persistent volumes.
> For a demo, run FastAPI + agent-runner on Railway and MLflow + Grafana locally.

---

## Project Structure

```
mlops-agent/
├── app/
│   ├── main.py          FastAPI server + /metrics endpoint
│   ├── model.py         Thread-safe model singleton + hot-reload
│   ├── metrics.py       Prometheus metric definitions
│   ├── train.py         XGBoost training script
│   └── schemas.py       Pydantic v2 request/response models
├── data/
│   ├── generate_data.py UPI transaction data generator
│   └── inject_drift.py  3-feature drift injector (--severity low/medium/high)
├── monitor/
│   ├── drift_check.py   Evidently DataDrift + DataQuality report
│   ├── scheduler.py     APScheduler (every 5 min)
│   └── test_drift.py    10 unit tests for drift logic
├── agent/
│   ├── state.py         AgentState TypedDict + Decision Literal
│   ├── graph.py         LangGraph StateGraph (4 conditional edges + retry)
│   ├── runner.py        CLI: --once / --watch / --mock
│   └── nodes/
│       ├── monitor.py   Load + validate latest_report.json
│       ├── reason.py    Rule-based (offline) + Claude Haiku (LLM) reasoning
│       └── act.py       Route decision to action function
├── actions/
│   ├── __init__.py      execute_action() with MLflow tagging + timing
│   ├── retrain.py       Retrain + MLflow logging + promotion + hot-reload
│   ├── rollback.py      MLflow registry rollback + local backup fallback
│   └── alert.py         Slack Block Kit + file fallback
├── grafana/
│   ├── dashboards/      mlops_agent.json (6-panel dashboard)
│   └── provisioning/    Auto-provisioning for datasource + dashboard
├── prometheus/
│   └── prometheus.yml   Scrape config (15s interval, 15d retention)
├── scripts/
│   └── run_demo.sh      Full demo automation script
├── tests/
│   └── test_full_loop.py End-to-end: drift -> agent -> retrain -> MLflow
├── docker-compose.yml   Full 6-service stack
├── Dockerfile           Single image for all Python services
└── .env.example         All required environment variables documented
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Inference | FastAPI + XGBoost | Low-latency serving; industry standard |
| Drift detection | Evidently AI 0.7.x | Battle-tested; Wasserstein + JS distance |
| Agent orchestration | LangGraph 1.0.9 | Stateful graphs with retry routing |
| LLM reasoning | Claude Haiku (Anthropic) | Fast, cheap, structured JSON output |
| Experiment tracking | MLflow 2.x | Model registry + artifact store |
| Metrics | Prometheus + Grafana | De-facto observability standard |
| Scheduling | APScheduler 3.x | Lightweight; no Celery overhead |
| Containerisation | Docker Compose | One-command local stack |
