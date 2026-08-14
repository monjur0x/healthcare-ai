#!/usr/bin/env bash
# One-command runner for the Healthcare AI system (CPU-only friendly).
#
#   scripts/run_system.sh start   # train + backend + dashboard + n8n (docker)
#   scripts/run_system.sh status  # show what is running
#   scripts/run_system.sh stop    # stop everything started here
#
# Everything is configured through environment variables (see defaults
# below). No GPU is required: all models are small and run on CPU.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${VENV_PY:-$ROOT/backend/CrewAI/.venv-opencode/bin/python}"
DATASET_DIR="${DATASET_DIR:-/home/monjur0x0/dataset}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$ROOT/backend/artifacts}"
API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"
N8N_PORT="${N8N_PORT:-5678}"
PRESET="${PRESET:-diabetes}"
N8N_ENABLED="${N8N_ENABLED:-1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
PID_DIR="${PID_DIR:-/tmp/healthcare-ai}"

API_URL="http://localhost:${API_PORT}"
mkdir -p "$LOG_DIR" "$PID_DIR"

log() { printf '[healthcare] %s\n' "$*"; }

require_venv() {
  if [ ! -x "$VENV_PY" ]; then
    log "error: venv not found at $VENV_PY (create it or set VENV_PY)"
    exit 1
  fi
}

start_backend() {
  if curl -s -m 2 "$API_URL/health" >/dev/null 2>&1; then
    log "backend already running at $API_URL/health"
    return 0
  fi
  log "starting FastAPI backend on port $API_PORT ..."
  (cd "$ROOT/backend" && \
   API_DATASET_DIR="$DATASET_DIR" \
   API_ARTIFACTS_DIR="$ARTIFACTS_DIR" \
   setsid "$VENV_PY" -m uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" \
     > "$LOG_DIR/api.log" 2>&1 &
   echo $! > "$PID_DIR/backend.pid")
  for _ in $(seq 1 30); do
    curl -s -m 1 "$API_URL/health" >/dev/null 2>&1 && { log "backend ready"; return 0; }
    sleep 1
  done
  log "error: backend did not become healthy; see $LOG_DIR/api.log"
  exit 1
}

train_default_model() {
  log "training default model (preset='$PRESET') via /api/v1/train ..."
  curl -s -X POST "$API_URL/api/v1/train" \
    -H "Content-Type: application/json" \
    -d "{\"preset\": \"$PRESET\", \"model\": \"mlp\"}" | "$VENV_PY" -c \
    'import json,sys; d=json.load(sys.stdin);
     if "model_path" in d:
       print("trained:", d["model_path"], "| accuracy: %.3f" % d["accuracy"], "| federated:", d["federated"])
     else:
       print("train failed:", json.dumps(d)); sys.exit(1)'
}

start_dashboard() {
  if curl -s -m 2 -o /dev/null "http://localhost:${DASHBOARD_PORT}/"; then
    log "dashboard already running on port $DASHBOARD_PORT"
    return 0
  fi
  log "starting Streamlit dashboard on port $DASHBOARD_PORT ..."
  (cd "$ROOT/frontend" && \
   setsid "$VENV_PY" -m streamlit run "$ROOT/frontend/streamlit_app.py" \
     --server.port "$DASHBOARD_PORT" --server.address 0.0.0.0 --server.headless true \
     > "$LOG_DIR/dashboard.log" 2>&1 &
   echo $! > "$PID_DIR/dashboard.pid")
  sleep 4
  curl -s -m 2 -o /dev/null "http://localhost:${DASHBOARD_PORT}/" \
    && log "dashboard ready at http://localhost:${DASHBOARD_PORT}" \
    || log "warning: dashboard not responding yet; see $LOG_DIR/dashboard.log"
}

start_n8n() {
  if ! command -v docker >/dev/null 2>&1; then
    log "docker not found; skipping n8n (run it manually if desired)"
    return 0
  fi
  if curl -s -m 2 -o /dev/null "http://localhost:${N8N_PORT}/"; then
    log "n8n already running on port $N8N_PORT"
    return 0
  fi
  log "starting n8n (docker) on port $N8N_PORT ..."
  docker run -d --rm --name healthcare-n8n \
    -p "${N8N_PORT}:5678" \
    -v healthcare_n8n_data:/home/node/.n8n \
    n8nio/n8n >/dev/null 2>&1 \
    && log "n8n container started: http://localhost:${N8N_PORT} (activate the workflow, then import n8n/healthcare-endtoend.json)" \
    || log "warning: n8n docker start failed (image pull on first run can take a while; retry with 'scripts/run_system.sh start')"
}

cmd_start() {
  require_venv
  start_backend
  train_default_model
  start_dashboard
  if [ "$N8N_ENABLED" = "1" ]; then
    start_n8n
  else
    log "n8n skipped (N8N_ENABLED=0)"
  fi
  log ""
  log "==================================================================="
  log "  System ready (no GPU needed):"
  log "  - Dashboard : http://localhost:${DASHBOARD_PORT}"
  log "  - API docs  : http://localhost:${API_PORT}/docs"
  log "  - n8n       : http://localhost:${N8N_PORT}"
  log ""
  log "  Try the dashboard, or drive the full pipeline via n8n:"
  log "  curl -X POST http://localhost:${N8N_PORT}/webhook/healthcare-endtoend \\"
  log "       -H 'Content-Type: application/json' -d @payload.json"
  log "==================================================================="
}

cmd_status() {
  for name in backend dashboard; do
    pid_file="$PID_DIR/$name.pid"
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      log "$name: running (pid $(cat "$pid_file"))"
    else
      log "$name: not running"
    fi
  done
  log "n8n: $(docker inspect -f '{{.State.Running}}' healthcare-n8n 2>/dev/null || echo 'not running')"
}

cmd_stop() {
  for name in backend dashboard; do
    pid_file="$PID_DIR/$name.pid"
    if [ -f "$pid_file" ]; then
      kill "$(cat "$pid_file")" 2>/dev/null && log "stopped $name"
      rm -f "$pid_file"
    fi
  done
  docker rm -f healthcare-n8n >/dev/null 2>&1 && log "stopped n8n container"
  log "done"
}

case "${1:-}" in
  start)  cmd_start ;;
  status) cmd_status ;;
  stop)   cmd_stop ;;
  *) echo "usage: $0 {start|status|stop}"; exit 2 ;;
esac