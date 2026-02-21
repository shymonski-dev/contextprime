# Reset Recovery Note

Date: 2026-02-20  
Workspace: `/Volumes/SSD2/SUPER_RAG`

## Purpose

This note captures the exact restart path after a machine reset and the latest end of day stop state.

## End of day stop state

- Development is paused by request.
- Duplicate full test runs were stopped.
- No active `run_full_tests_stable.sh` or `docker compose build app` process remains.
- Docker compose services were stopped with `docker compose down`.
- Full stable test gate must be re run from a clean morning start.

## First steps after reset

1. Start Docker Desktop and wait for healthy state.
2. Open terminal and move to workspace root:
   - `cd /Volumes/SSD2/SUPER_RAG`
3. Check Docker daemon:
   - `docker info`
4. Run security gate first:
   - `./run_security_release_gate.sh`
5. Run compliance readiness gate:
   - `./run_european_union_artificial_intelligence_act_readiness.sh`
6. Run full stable suite:
   - `./run_full_tests_stable.sh`

## Required environment values

Set these before test and deployment gates:

```bash
NEO4J_PASSWORD=<strong_password>
OPENAI_API_KEY=<provider_key>
SECURITY__REQUIRE_ACCESS_TOKEN=true
SECURITY__AUTH_MODE=jwt
SECURITY__JWT_SECRET=<long_random_secret>
```

If `NEO4J_PASSWORD` is not exported, the full test runner also checks `doctags_rag/.env`.

## If Docker is unavailable

1. Confirm Docker Desktop is open.
2. Recheck with `docker info`.
3. Retry `docker compose up -d neo4j qdrant`.

## Done condition

Work is unblocked when:

- `./run_security_release_gate.sh` passes.
- `./run_european_union_artificial_intelligence_act_readiness.sh` passes.
- `./run_full_tests_stable.sh` completes with empty `FAILED:` output.
