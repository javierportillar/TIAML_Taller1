#!/usr/bin/env bash
# Lanza la API FastAPI del agente en modo desarrollo con auto-reload.
# Uso:  ./scripts/run_api.sh           -> escucha en 127.0.0.1:8000
#       API_HOST=0.0.0.0 API_PORT=8080 ./scripts/run_api.sh   -> expone a la LAN

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "[!] No existe .venv. Crea uno con: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

HOST="${API_HOST:-127.0.0.1}"
PORT="${API_PORT:-8000}"

echo "[*] Lanzando uvicorn en ${HOST}:${PORT} con --reload"
exec .venv/bin/uvicorn api.main:app --host "${HOST}" --port "${PORT}" --reload
