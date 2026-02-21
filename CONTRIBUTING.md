# Contributing

Thank you for helping improve Contextprime.

## Before opening a pull request

1. Run security checks:
   - `./run_security_release_gate.sh`
2. Run readiness checks:
   - `./run_european_union_artificial_intelligence_act_readiness.sh`
3. Run stable test suite:
   - `./run_full_tests_stable.sh --skip-build`

## Pull request rules

1. Keep changes focused and explain why they are needed.
2. Add or update tests for behavior changes.
3. Update documentation when user behavior changes.
4. Never commit secrets, keys, or local environment files.

## Local development

1. Copy environment template:
   - `cp doctags_rag/.env.example doctags_rag/.env`
2. Start services:
   - `docker compose up -d neo4j qdrant app`
3. Run tests:
   - `cd doctags_rag && pytest tests/ -q`
