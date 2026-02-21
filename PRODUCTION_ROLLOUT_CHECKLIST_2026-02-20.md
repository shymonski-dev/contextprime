# Production Rollout Checklist (February 20, 2026)

Use this checklist for the current hardened release.

Current checkpoint:
- Code, policy, and live production smoke gates are passing.
- Real-world validation is passing with artifact evidence recorded.

Latest verified run evidence:
- Stable suite: `TOTAL_RUNS:46`, empty `FAILED:`
- Live smoke gate: `./run_live_smoke_gate.sh` passed
- Real-world gate: `./run_realworld_stable.sh` passed
- Real-world artifacts: `/Volumes/SSD2/SUPER_RAG/doctags_rag/reports/realworld_e2e_20260221_124554`

## 1. Release scope lock

- [ ] Confirm the release branch and commit are frozen.
- [ ] Confirm the updated files are included:
`doctags_rag/src/api/middleware.py`,
`doctags_rag/src/api/services/retrieval_service.py`,
`doctags_rag/src/retrieval/hybrid_retriever.py`,
`doctags_rag/src/api/services/metrics_store.py`,
`doctags_rag/src/retrieval/feedback_capture_store.py`,
`doctags_rag/config/config.yaml`.
- [ ] Confirm no unreviewed local edits are included in production.

## 2. Secrets and runtime configuration

- [ ] Set required secrets in `doctags_rag/.env`.
- [ ] Use signed token mode for production access control.
- [ ] Confirm strong secret lengths and no default passwords.

Minimum secure values:

```bash
NEO4J_PASSWORD=<strong_password>
OPENAI_API_KEY=<provider_key>
SECURITY__REQUIRE_ACCESS_TOKEN=true
SECURITY__AUTH_MODE=jwt
SECURITY__JWT_SECRET=<at_least_32_characters>
SECURITY__JWT_ALGORITHM=HS256
SECURITY__JWT_ENFORCE_PERMISSIONS=true
SECURITY__JWT_REQUIRED_READ_SCOPES=api:read
SECURITY__JWT_REQUIRED_WRITE_SCOPES=api:write
SECURITY__JWT_ADMIN_ROLES=admin,owner
```

Optional signed token checks:

```bash
SECURITY__JWT_ISSUER=<issuer_value>
SECURITY__JWT_AUDIENCE=<audience_value>
```

## 3. Retrieval budget policy confirmation

- [ ] Confirm request budget limits in `doctags_rag/config/config.yaml`.
- [ ] Confirm current values are acceptable for cost and latency.

Current defaults:

```yaml
retrieval:
  hybrid_search:
    request_budget:
      max_top_k: 12
      max_query_variants: 3
      max_corrective_variants: 2
      max_total_variant_searches: 5
      max_search_time_ms: 4500
```

## 4. Data protection before deploy

- [ ] Create backup folder.
- [ ] Back up application data bind mount.
- [ ] Back up graph database and vector database named volumes.
- [ ] Confirm volume names if your project name is different.

```bash
docker volume ls | grep -E "neo4j_data|qdrant_storage"
mkdir -p backups
tar -czf "backups/doctags_data_$(date +%F_%H%M%S).tgz" doctags_rag/data

docker run --rm -v super_rag_neo4j_data:/from -v "$PWD/backups:/to" alpine \
  sh -c 'cd /from && tar -czf "/to/neo4j_data_$(date +%F_%H%M%S).tgz" .'

docker run --rm -v super_rag_qdrant_storage:/from -v "$PWD/backups:/to" alpine \
  sh -c 'cd /from && tar -czf "/to/qdrant_storage_$(date +%F_%H%M%S).tgz" .'
```

## 5. Build and test gate

- [ ] Build the production application image.
- [ ] Run European Union Artificial Intelligence Act readiness check.
- [ ] Run full stable test command.
- [ ] Confirm no failed targets.

```bash
docker compose --env-file doctags_rag/.env build app
./run_european_union_artificial_intelligence_act_readiness.sh
./run_full_tests_stable.sh
```

Release gate:

- [ ] `TOTAL_RUNS` is present.
- [ ] `FAILED:` is empty.

## 6. Deployment sequence

- [ ] Start data services first.
- [ ] Wait for readiness.
- [ ] Start application service.
- [ ] Confirm all services are healthy.

```bash
docker compose --env-file doctags_rag/.env up -d neo4j qdrant
docker compose --env-file doctags_rag/.env up -d app
docker compose --env-file doctags_rag/.env ps
docker compose --env-file doctags_rag/.env logs --tail 200 app
```

## 7. Smoke checks after deployment

- [ ] Health endpoint returns success.
- [ ] Readiness endpoint returns success.
- [ ] Status endpoint shows healthy dependencies.
- [ ] Protected route denies missing token.
- [ ] Protected route allows valid signed token.

```bash
curl -sS http://localhost:8000/api/health
curl -sS http://localhost:8000/api/readiness
curl -sS http://localhost:8000/api/status
```

Before the authenticated search check, confirm provider key is not placeholder:

```bash
awk -F= '/^OPENAI_API_KEY=/{if($2 ~ /^replace_/ || $2 == \"\"){print \"INVALID\"}else{print \"OK\"}}' doctags_rag/.env
```

Then run one real search and one feedback submission from the user interface and confirm:

- [ ] Search response metadata includes request budget fields.
- [ ] Query identifier is present in metadata.
- [ ] Feedback write succeeds.

Single-command live smoke gate:

```bash
./run_live_smoke_gate.sh
```

Single-command Real-world stable validation:

```bash
./run_realworld_stable.sh
```

## 8. First hour production watch

- [ ] Watch application logs for internal server errors.
- [ ] Watch for repeated forbidden or rate limit responses.
- [ ] Track query latency from search response metadata.
- [ ] Confirm upload count and query count continue to increase after application restarts.

## 9. Rollback plan

Rollback trigger examples:

- [ ] Repeated readiness failures.
- [ ] Sustained internal server errors.
- [ ] Sustained latency above agreed limit.

Rollback actions:

```bash
docker compose --env-file doctags_rag/.env down
# restore previous image or previous commit
docker compose --env-file doctags_rag/.env up -d neo4j qdrant app
```

If data rollback is needed, restore from backups created in section 4.

## 10. Sign-off record

- [ ] Engineering sign-off
- [ ] Operations sign-off
- [ ] Security sign-off
- [ ] Product sign-off
- [ ] Deployment time recorded
- [ ] Post deployment review time scheduled
