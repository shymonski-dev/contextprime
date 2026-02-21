#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "Running security release gate..."

if git ls-files | rg -q '(^|/)\.env$'; then
  echo "ERROR: tracked .env file detected."
  git ls-files | rg '(^|/)\.env$' || true
  exit 1
fi

if ! rg -q '^SECURITY__AUTH_MODE=jwt$' doctags_rag/.env.example; then
  echo "ERROR: doctags_rag/.env.example must default to SECURITY__AUTH_MODE=jwt."
  exit 1
fi

if ! rg -q '^SECURITY__JWT_SECRET=' doctags_rag/.env.example; then
  echo "ERROR: doctags_rag/.env.example must include SECURITY__JWT_SECRET."
  exit 1
fi

if rg -n --hidden --glob '!.git' -S \
  '(jwt_support_legacy_token_fallback|SECURITY__JWT_SUPPORT_LEGACY_TOKEN_FALLBACK)' \
  doctags_rag/src doctags_rag/tests >/dev/null; then
  echo "ERROR: legacy signed token fallback settings are still present."
  rg -n --hidden --glob '!.git' -S \
    '(jwt_support_legacy_token_fallback|SECURITY__JWT_SUPPORT_LEGACY_TOKEN_FALLBACK)' \
    doctags_rag/src doctags_rag/tests || true
  exit 1
fi

if rg -n --hidden --glob '!.git' -S \
  '^[[:space:]]*NEO4J_PASSWORD=password[[:space:]]*$|^[[:space:]]*SECURITY__AUTH_MODE=token[[:space:]]*$' \
  README.md QUICK_REFERENCE.md DUAL_INDEXING_SETUP.md doctags_rag/.env.example >/dev/null; then
  echo "ERROR: weak documented defaults detected."
  rg -n --hidden --glob '!.git' -S \
    '^[[:space:]]*NEO4J_PASSWORD=password[[:space:]]*$|^[[:space:]]*SECURITY__AUTH_MODE=token[[:space:]]*$' \
    README.md QUICK_REFERENCE.md DUAL_INDEXING_SETUP.md doctags_rag/.env.example || true
  exit 1
fi

if rg -n --hidden --glob '!.git' --glob '!run_security_release_gate.sh' -S \
  '(BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY|BEGIN EC PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|sk-[A-Za-z0-9]{20,})' >/dev/null; then
  echo "ERROR: potential secret material detected in tracked files."
  rg -n --hidden --glob '!.git' --glob '!run_security_release_gate.sh' -S \
    '(BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY|BEGIN EC PRIVATE KEY|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{20,}|sk-[A-Za-z0-9]{20,})' || true
  exit 1
fi

if [ -x doctags_rag/venv/bin/python3 ]; then
  doctags_rag/venv/bin/python3 -m pip_audit --local
elif command -v pip-audit >/dev/null 2>&1; then
  pip-audit --local
else
  echo "ERROR: pip-audit is required for dependency scanning."
  exit 1
fi

echo "Security release gate passed."
