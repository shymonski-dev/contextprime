#!/usr/bin/env bash
set -euo pipefail

# If the first argument looks like a flag, prepend uvicorn to the command
if [[ $# -eq 0 ]]; then
  set -- uvicorn
fi

if [[ "${1:-}" == -* ]]; then
  set -- uvicorn "$@"
fi

if [[ "${1:-}" == "uvicorn" ]]; then
  HOST="${HOST:-0.0.0.0}"
  PORT="${PORT:-8000}"
  WORKERS="${UVICORN_WORKERS:-1}"

  shift 1
  set -- uvicorn src.api.main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    "$@"
fi

exec "$@"
