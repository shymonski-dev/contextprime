#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BUILD_APP=1
if [[ "${1:-}" == "--skip-build" ]]; then
  BUILD_APP=0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker command is not available."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker daemon is not reachable."
  echo "Start Docker Desktop and try again."
  exit 1
fi

if [[ -z "${NEO4J_PASSWORD:-}" && -f "doctags_rag/.env" ]]; then
  NEO4J_PASSWORD="$(rg '^NEO4J_PASSWORD=' doctags_rag/.env -N -m 1 | cut -d'=' -f2- || true)"
  export NEO4J_PASSWORD
fi

if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
  echo "ERROR: NEO4J_PASSWORD must be set (shell env or doctags_rag/.env)."
  exit 1
fi

if [[ "$BUILD_APP" -eq 1 ]]; then
  echo "Building application image..."
  docker compose build app
fi

echo "Starting required data services..."
docker compose up -d neo4j qdrant

wait_for_http() {
  local label="$1"
  local url="$2"
  local timeout_seconds="$3"
  local start_ts
  start_ts="$(date +%s)"
  local now_ts

  echo "Waiting for ${label} readiness at ${url}..."
  while true; do
    local probe_ok=1
    if command -v curl >/dev/null 2>&1; then
      if curl --silent --show-error --fail --max-time 5 "${url}" >/dev/null 2>&1; then
        probe_ok=0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget --no-verbose --tries=1 --timeout=5 --spider "${url}" >/dev/null 2>&1; then
        probe_ok=0
      fi
    else
      echo "ERROR: curl or wget is required for readiness checks."
      return 1
    fi
    if [[ "$probe_ok" -eq 0 ]]; then
      echo "${label} is ready."
      return 0
    fi

    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_seconds )); then
      echo "ERROR: ${label} did not become ready within ${timeout_seconds}s."
      docker compose ps
      docker compose logs --tail 120 neo4j qdrant || true
      return 1
    fi
    sleep 3
  done
}

wait_for_http "Qdrant" "http://127.0.0.1:6333/readyz" 180
wait_for_http "Neo4j" "http://127.0.0.1:7474" 240

append_unique_candidate() {
  local value="${1:-}"
  if [[ -z "$value" ]]; then
    return
  fi
  local existing
  for existing in "${NEO4J_PASSWORD_CANDIDATES[@]:-}"; do
    if [[ "$existing" == "$value" ]]; then
      return
    fi
  done
  NEO4J_PASSWORD_CANDIDATES+=("$value")
}

detect_neo4j_password() {
  NEO4J_PASSWORD_CANDIDATES=()
  append_unique_candidate "${NEO4J_PASSWORD:-}"

  if [[ -f "doctags_rag/.env" ]]; then
    local env_password
    env_password="$(rg '^NEO4J_PASSWORD=' doctags_rag/.env -N -m 1 | cut -d'=' -f2- || true)"
    append_unique_candidate "$env_password"
  fi

  append_unique_candidate "password"
  append_unique_candidate "change_this_neo4j_password"
  append_unique_candidate "neo4j"

  local candidate
  for candidate in "${NEO4J_PASSWORD_CANDIDATES[@]}"; do
    if docker exec doctags-neo4j cypher-shell -u neo4j -p "$candidate" "RETURN 1;" >/dev/null 2>&1; then
      RESOLVED_NEO4J_PASSWORD="$candidate"
      return 0
    fi
  done

  return 1
}

if ! detect_neo4j_password; then
  echo "ERROR: Unable to authenticate to Neo4j with known password candidates."
  echo "Set NEO4J_PASSWORD in your shell or doctags_rag/.env and retry."
  docker compose ps
  docker compose logs --tail 80 neo4j || true
  exit 1
fi

echo "Running full test suite (stable mode)..."
docker compose run --rm --no-deps \
  -e SECURITY__REQUIRE_ACCESS_TOKEN=false \
  -e SECURITY__ACCESS_TOKEN= \
  -e NEO4J__PASSWORD="$RESOLVED_NEO4J_PASSWORD" \
  -e NEO4J_PASSWORD="$RESOLVED_NEO4J_PASSWORD" \
  app bash -lc '
set +e
failed=""
run_count=0

run_target() {
  local target="$1"
  echo "RUN:$target"
  pytest "$target" -q
  local rc=$?
  if [ $rc -eq 137 ]; then
    echo "RETRY:$target (first attempt exited with 137)"
    pytest "$target" -q
    rc=$?
  fi
  run_count=$((run_count + 1))
  if [ $rc -ne 0 ]; then
    failed="$failed $target"
  fi
}

run_knowledge_graph_split() {
  local targets=(
    "tests/test_knowledge_graph.py::test_pipeline_handles_unavailable_neo4j"
    "tests/test_knowledge_graph.py::TestEntityExtractor::test_basic_entity_extraction"
    "tests/test_knowledge_graph.py::TestEntityExtractor::test_entity_confidence_filtering"
    "tests/test_knowledge_graph.py::TestEntityExtractor::test_entity_context_extraction"
    "tests/test_knowledge_graph.py::TestEntityExtractor::test_batch_entity_extraction"
    "tests/test_knowledge_graph.py::TestEntityExtractor::test_entity_deduplication"
    "tests/test_knowledge_graph.py::TestRelationshipExtractor::test_basic_relationship_extraction"
    "tests/test_knowledge_graph.py::TestRelationshipExtractor::test_relationship_confidence"
    "tests/test_knowledge_graph.py::TestRelationshipExtractor::test_relationship_types"
    "tests/test_knowledge_graph.py::test_entity_extractor_handles_missing_model"
    "tests/test_knowledge_graph.py::TestEntityResolver::test_exact_match_resolution"
    "tests/test_knowledge_graph.py::TestEntityResolver::test_fuzzy_match_resolution"
    "tests/test_knowledge_graph.py::TestEntityResolver::test_type_based_resolution"
    "tests/test_knowledge_graph.py::TestEntityResolver::test_cross_document_resolution"
    "tests/test_knowledge_graph.py::TestGraphBuilder::test_document_node_creation"
    "tests/test_knowledge_graph.py::TestGraphBuilder::test_entity_node_creation"
    "tests/test_knowledge_graph.py::TestKnowledgeGraphPipeline::test_pipeline_initialization"
    "tests/test_knowledge_graph.py::TestKnowledgeGraphPipeline::test_pipeline_config"
    "tests/test_knowledge_graph.py::TestGraphQueryInterface::test_entity_search"
    "tests/test_knowledge_graph.py::TestGraphQueryInterface::test_entity_statistics"
    "tests/test_knowledge_graph.py::TestKnowledgeGraphIntegration::test_end_to_end_pipeline"
    "tests/test_knowledge_graph.py::TestKnowledgeGraphIntegration::test_cross_document_linking"
    "tests/test_knowledge_graph.py::TestKnowledgeGraphIntegration::test_query_after_build"
  )

  local target
  for target in "${targets[@]}"; do
    run_target "$target"
  done
}

for file in tests/test_*.py; do
  if [ "$file" = "tests/test_knowledge_graph.py" ]; then
    run_knowledge_graph_split
  else
    run_target "$file"
  fi
done

echo "TOTAL_RUNS:$run_count"
if [ -n "$failed" ]; then
  echo "FAILED:$failed"
  exit 1
fi

echo "FAILED:"
exit 0
'
