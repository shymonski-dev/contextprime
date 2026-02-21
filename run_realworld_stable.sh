#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/doctags_rag/.env"
APP_DIR="$ROOT_DIR/doctags_rag"
SETUP_SCRIPT="$APP_DIR/scripts/setup_databases.py"
RUN_SCRIPT="$APP_DIR/scripts/run_pdf_realworld_e2e.py"
PYTHON_BIN="$APP_DIR/venv/bin/python"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: missing environment file: $ENV_FILE"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: missing python virtual environment interpreter: $PYTHON_BIN"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" =~ ^replace_ ]]; then
  echo "ERROR: set a valid provider key in $ENV_FILE before running this command."
  exit 1
fi

if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
  echo "ERROR: NEO4J_PASSWORD must be set in $ENV_FILE."
  exit 1
fi

echo "Step 1: ensuring graph and vector storage objects..."
(
  export QDRANT_HOST=127.0.0.1
  export QDRANT__HOST=127.0.0.1
  export QDRANT_PORT=6333
  export QDRANT__PORT=6333
  export NEO4J_URI=bolt://127.0.0.1:7687
  export NEO4J__URI=bolt://127.0.0.1:7687
  "$PYTHON_BIN" "$SETUP_SCRIPT"
)

echo "Step 2: restarting app service..."
docker compose --env-file "$ENV_FILE" restart app >/dev/null

echo "Step 3: waiting for readiness..."
for _ in $(seq 1 40); do
  if curl -sS --max-time 5 http://127.0.0.1:8000/api/readiness >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -sS --max-time 5 http://127.0.0.1:8000/api/readiness >/dev/null 2>&1; then
  echo "ERROR: app readiness check failed after restart."
  docker compose --env-file "$ENV_FILE" logs --tail 120 app || true
  exit 1
fi

echo "Step 4: running real-world document flow..."
"$PYTHON_BIN" -u "$RUN_SCRIPT" "$@"

echo "Done."
