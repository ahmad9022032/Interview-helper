#!/usr/bin/env bash
# Starts Ollama (if needed), the backend and the frontend, and waits.
# Ctrl-C stops everything.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"

[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example"; }
set -a; . ./.env; set +a
MODEL="${MODEL:-qwen2.5:3b-instruct}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

command -v ollama >/dev/null || { echo "Ollama is not installed. See the README."; exit 1; }

if ! curl -sf -m 2 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null; then
  echo "starting ollama..."
  ollama serve >/tmp/interview-copilot-ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf -m 2 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null && break
    sleep 1
  done
fi

if ! ollama list | awk '{print $1}' | grep -qx "$MODEL"; then
  echo "pulling $MODEL (first run only)..."
  ollama pull "$MODEL"
fi

[ -d backend/.venv ] || {
  echo "creating backend virtualenv..."
  python3 -m venv backend/.venv
  backend/.venv/bin/pip install -q --upgrade pip
  backend/.venv/bin/pip install -q -r backend/requirements.txt
}
[ -d frontend/node_modules ] || { echo "installing frontend deps..."; (cd frontend && npm install --no-audit --no-fund); }

cleanup() { echo; echo "stopping..."; kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "starting backend on :$BACKEND_PORT ..."
(cd backend && exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT") &

echo "starting frontend on :$FRONTEND_PORT ..."
(cd frontend && exec npm run dev -- -p "$FRONTEND_PORT") &

echo
echo "Interview Copilot starting up (first run downloads the Whisper model)."
echo "  UI      http://localhost:$FRONTEND_PORT"
echo "  health  http://localhost:$BACKEND_PORT/api/health"
echo "  model   $MODEL"
echo
wait
