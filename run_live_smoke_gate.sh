#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_FILE="doctags_rag/.env"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
SAMPLE_FILE="${SMOKE_SAMPLE_FILE:-doctags_rag/data/samples/sample_text.txt}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: missing environment file: $ENV_FILE"
  exit 1
fi

if [[ ! -f "$SAMPLE_FILE" ]]; then
  echo "ERROR: missing sample file for upload smoke test: $SAMPLE_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

if [[ -z "${SECURITY__JWT_SECRET:-${SECURITY_JWT_SECRET:-}}" ]]; then
  echo "ERROR: SECURITY__JWT_SECRET is required for signed token smoke checks."
  exit 1
fi

if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
  echo "ERROR: NEO4J_PASSWORD is required."
  exit 1
fi

echo "Starting services..."
docker compose --env-file "$ENV_FILE" up -d neo4j qdrant app >/dev/null

echo "Waiting for readiness..."
ready=0
for _ in $(seq 1 60); do
  if curl -sS --max-time 5 "$API_BASE_URL/api/readiness" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "$ready" -ne 1 ]]; then
  echo "ERROR: application readiness timed out."
  docker compose --env-file "$ENV_FILE" ps
  docker compose --env-file "$ENV_FILE" logs --tail 120 app || true
  exit 1
fi

health_code="$(curl -sS -o /tmp/live_smoke_health.json -w "%{http_code}" "$API_BASE_URL/api/health")"
readiness_code="$(curl -sS -o /tmp/live_smoke_readiness.json -w "%{http_code}" "$API_BASE_URL/api/readiness")"
unauth_search_code="$(
  curl -sS -o /tmp/live_smoke_unauth_search.json -w "%{http_code}" \
    -X POST "$API_BASE_URL/api/search/hybrid" \
    -H "Content-Type: application/json" \
    -d '{"query":"smoke test query","top_k":3}' \
    || true
)"

echo "health_code=$health_code"
echo "readiness_code=$readiness_code"
echo "unauth_search_code=$unauth_search_code"

if [[ "$health_code" != "200" || "$readiness_code" != "200" || "$unauth_search_code" != "401" ]]; then
  echo "ERROR: base smoke checks failed."
  exit 1
fi

openai_state="$(
  awk -F= '/^OPENAI_API_KEY=/{if ($2 == "") {print "empty"} else if ($2 ~ /^replace_/) {print "placeholder"} else {print "set"}}' "$ENV_FILE"
)"

if [[ "$openai_state" != "set" ]]; then
  echo "BLOCKED: OPENAI_API_KEY is $openai_state in $ENV_FILE."
  echo "Set a valid provider key, then rerun this command."
  exit 2
fi

token="$(
python3 - <<'PY'
import base64
import hashlib
import hmac
import json
import os

def b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

secret = (os.getenv("SECURITY__JWT_SECRET") or os.getenv("SECURITY_JWT_SECRET") or "").encode("utf-8")
header = {"alg": "HS256", "typ": "JWT"}
payload = {"sub": "smoke-runner", "roles": ["admin"], "scopes": ["api:read", "api:write"]}
parts = [
    b64url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
    b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
]
signature = hmac.new(secret, ".".join(parts).encode("ascii"), hashlib.sha256).digest()
print(".".join(parts + [b64url(signature)]))
PY
)"

upload_settings='{"enable_ocr":false,"chunk_size":800,"chunk_overlap":120,"chunking_method":"structure","extract_entities":false,"build_raptor":false}'
upload_code="$(
  curl -sS -o /tmp/live_smoke_upload.json -w "%{http_code}" \
    -X POST "$API_BASE_URL/api/documents" \
    -H "Authorization: Bearer $token" \
    -F "file=@$SAMPLE_FILE" \
    -F "settings=$upload_settings"
)"

auth_search_code="$(
  curl -sS -o /tmp/live_smoke_auth_search.json -w "%{http_code}" \
    -X POST "$API_BASE_URL/api/search/hybrid" \
    -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" \
    -d '{"query":"what topics are covered in this document?","top_k":3,"strategy":"hybrid"}'
)"

query_id="$(
python3 - <<'PY'
import json
from pathlib import Path

path = Path("/tmp/live_smoke_auth_search.json")
if not path.exists():
    print("")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    print(str(metadata.get("query_id") or ""))
PY
)"

feedback_code="not_run"
if [[ -n "$query_id" ]]; then
  feedback_code="$(
    curl -sS -o /tmp/live_smoke_feedback.json -w "%{http_code}" \
      -X POST "$API_BASE_URL/api/feedback/retrieval" \
      -H "Authorization: Bearer $token" \
      -H "Content-Type: application/json" \
      -d "{\"query_id\":\"$query_id\",\"helpful\":true,\"selected_result_ids\":[],\"result_labels\":[],\"comment\":\"live smoke feedback\"}"
  )"
fi

echo "upload_code=$upload_code"
echo "auth_search_code=$auth_search_code"
echo "query_id=$query_id"
echo "feedback_code=$feedback_code"

if [[ ( "$upload_code" == "200" || "$upload_code" == "201" ) && "$auth_search_code" == "200" && "$feedback_code" == "200" ]]; then
  echo "PASS: live smoke gate completed."
  exit 0
fi

echo "ERROR: live smoke gate failed."
exit 1
