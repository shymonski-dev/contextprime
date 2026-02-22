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
5. By submitting a contribution, you agree that your contribution may be
   distributed under the project dual license model in `LICENSE`.

## Local development

1. Copy environment template:
   - `cp doctags_rag/.env.example doctags_rag/.env`
2. Start services:
   - `docker compose up -d neo4j qdrant app`
3. Run tests:
   - `cd doctags_rag && pytest tests/ -q`

## Testing conventions

- **Admin routes**: use `app.dependency_overrides[require_admin_identity]` in test
  fixtures to bypass the role gate. Restore the override after each test.
- **Retriever mocks**: `_search_vector` and `_search_graph` return
  `Tuple[List, Optional[str]]`. Mocks must follow the same contract:
  `lambda *args, **kwargs: ([], None)` for success, or `([], "error message")` to
  simulate a backend failure.
- **Rate limiter with Redis mock**: pass a fake client via `limiter._redis_client`
  after construction. The Lua eval signature is
  `eval(script, num_keys, redis_key, now_ms, window_ms, max_requests, member, cost_units)`;
  cost is `ARGV[5]` (0-indexed: `args[5]` in Python).
- **JWT tests**: `_build_settings()` must include `jwt_require_expiry=True` (the
  default) so exp-enforcement tests are not masked by a missing attribute.
