#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/doctags_rag"
SCRIPT_PATH="$APP_DIR/scripts/check_european_union_artificial_intelligence_act_readiness.py"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: missing readiness script at $SCRIPT_PATH"
  exit 1
fi

if command -v python3 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
then
  echo "Running readiness check with local python environment..."
  python3 "$SCRIPT_PATH"
  exit $?
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "Running readiness check with docker compose app service..."
  docker compose run --rm --no-deps app python scripts/check_european_union_artificial_intelligence_act_readiness.py
  exit $?
fi

echo "ERROR: python with yaml support or docker is required."
exit 1
