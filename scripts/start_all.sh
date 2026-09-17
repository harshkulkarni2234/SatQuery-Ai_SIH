#!/usr/bin/env bash
# Phase C9: starts the full stack for a demo (backend, VQA worker, frontend)
# in the background and reports each PID/log file. Actually run and verified
# on this project's macOS dev machine — see scripts/start_all.ps1 for the
# Windows mirror (authored to match this script step-for-step, but not
# executed on a real Windows machine in this session; see
# docs/DEMO_RUNBOOK.md section 6).
#
# Usage: bash scripts/start_all.sh [--no-vqa]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/scripts/.logs"
mkdir -p "$LOG_DIR"

SKIP_VQA=false
if [[ "${1:-}" == "--no-vqa" ]]; then
  SKIP_VQA=true
fi

echo "== SatQuery AI: starting all services =="

# 1. PostgreSQL check (does not start it — assumes a system service; this
# repo has no bundled Postgres to launch, only to check).
if command -v pg_isready >/dev/null 2>&1; then
  if pg_isready -q; then
    echo "[OK]   PostgreSQL is accepting connections"
  else
    echo "[WARN] PostgreSQL does not appear to be running — start it first"
    echo "       (macOS: brew services start postgresql@17)"
  fi
else
  echo "[WARN] pg_isready not found — cannot verify PostgreSQL is running"
fi

# 2. Backend
echo "[..]   Starting backend (uvicorn) on :8000"
(
  cd "$REPO_ROOT/backend"
  VENV_PY="venv_mac/bin/uvicorn"
  if [[ ! -x "$VENV_PY" ]]; then
    VENV_PY="venv/bin/uvicorn"
  fi
  "$VENV_PY" app.main:app --host 127.0.0.1 --port 8000 \
    > "$LOG_DIR/backend.log" 2>&1 &
  echo $! > "$LOG_DIR/backend.pid"
)
sleep 2
if curl -fsS http://127.0.0.1:8000/health > /dev/null 2>&1; then
  echo "[OK]   Backend is up (pid $(cat "$LOG_DIR/backend.pid")) — log: $LOG_DIR/backend.log"
else
  echo "[FAIL] Backend did not respond on :8000 — check $LOG_DIR/backend.log"
fi

# 3. VQA worker (optional — the system degrades honestly without it)
if [[ "$SKIP_VQA" == false ]]; then
  echo "[..]   Starting VQA worker on :8001"
  (
    cd "$REPO_ROOT/ml/vqa-worker"
    PYBIN="python3"
    if [[ -x "venv/bin/python" ]]; then
      PYBIN="venv/bin/python"
    fi
    "$PYBIN" -m uvicorn worker_service:app --host 127.0.0.1 --port 8001 \
      > "$LOG_DIR/vqa_worker.log" 2>&1 &
    echo $! > "$LOG_DIR/vqa_worker.pid"
  )
  echo "[..]   (VQA worker takes longer to become healthy — model load in progress)"
else
  echo "[SKIP] VQA worker (--no-vqa passed) — description queries will use the honest fallback"
fi

# 4. Frontend
echo "[..]   Starting frontend (vite dev server) on :5173"
(
  cd "$REPO_ROOT/frontend"
  npm run dev -- --port 5173 > "$LOG_DIR/frontend.log" 2>&1 &
  echo $! > "$LOG_DIR/frontend.pid"
)
sleep 2
echo "[OK]   Frontend starting (pid $(cat "$LOG_DIR/frontend.pid")) — log: $LOG_DIR/frontend.log"

echo ""
echo "== Done. Open http://localhost:5173 =="
echo "PIDs recorded in $LOG_DIR/*.pid — stop each with: kill \$(cat $LOG_DIR/<name>.pid)"
echo "Run scripts/smoke_test.sh once all services report healthy."
