# mlops-agent/Dockerfile
# Single image used by fastapi-server, monitor-scheduler, agent-runner, mlflow-server.
# Keeping one image avoids version skew between services sharing the same codebase.

FROM python:3.13-slim

# Install curl for Docker health checks (not in slim by default)
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so Docker can cache this layer independently
# of code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default port — override per-service in docker-compose.yml
EXPOSE 8000

# Default command: FastAPI server.
# Each service overrides this in docker-compose.yml via `command:`.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
