from src.api.services.request_limit_store import SharedSlidingWindowRateLimiter


def test_sqlite_backed_limiter_denies_after_limit(tmp_path):
    limiter = SharedSlidingWindowRateLimiter(
        max_requests=2,
        window_seconds=60,
        redis_url=None,
        sqlite_path=tmp_path / "rate_limit.db",
    )

    assert limiter.check("ip:127.0.0.1").allowed is True
    assert limiter.check("ip:127.0.0.1").allowed is True
    decision = limiter.check("ip:127.0.0.1")
    assert decision.allowed is False
    assert decision.retry_after_seconds >= 1


def test_sqlite_backed_limiter_shared_across_instances(tmp_path):
    db_path = tmp_path / "rate_limit.db"
    limiter_a = SharedSlidingWindowRateLimiter(
        max_requests=2,
        window_seconds=60,
        redis_url=None,
        sqlite_path=db_path,
    )
    limiter_b = SharedSlidingWindowRateLimiter(
        max_requests=2,
        window_seconds=60,
        redis_url=None,
        sqlite_path=db_path,
    )

    assert limiter_a.check("ip:shared").allowed is True
    assert limiter_b.check("ip:shared").allowed is True
    assert limiter_a.check("ip:shared").allowed is False


def test_redis_failure_falls_back_to_sqlite(tmp_path):
    class BrokenRedis:
        def eval(self, *_args, **_kwargs):
            raise RuntimeError("redis down")

    limiter = SharedSlidingWindowRateLimiter(
        max_requests=1,
        window_seconds=60,
        redis_url=None,
        sqlite_path=tmp_path / "rate_limit.db",
    )
    limiter._redis_client = BrokenRedis()

    first = limiter.check("ip:fallback")
    second = limiter.check("ip:fallback")
    assert first.allowed is True
    assert second.allowed is False
    assert limiter._redis_client is None


def test_sqlite_backed_limiter_supports_weighted_cost(tmp_path):
    limiter = SharedSlidingWindowRateLimiter(
        max_requests=5,
        window_seconds=60,
        redis_url=None,
        sqlite_path=tmp_path / "rate_limit.db",
    )

    assert limiter.check("ip:weighted", cost=3).allowed is True
    assert limiter.check("ip:weighted", cost=2).allowed is True
    decision = limiter.check("ip:weighted", cost=1)
    assert decision.allowed is False


def test_redis_path_used_for_cost_greater_than_one(tmp_path):
    """Redis path must be exercised even when cost > 1 (no bypass)."""
    lua_calls = []

    class FakeRedis:
        def eval(self, script, num_keys, *args):
            # args layout: redis_key, now_ms, window_ms, max_requests, member, cost_units
            lua_calls.append(args)
            cost = int(args[5])  # ARGV[5] = cost_units (0-indexed: args[5])
            if len(lua_calls) == 1:
                return [1, 0]   # allow
            return [0, 10]      # deny

        def ping(self):
            return True

    limiter = SharedSlidingWindowRateLimiter(
        max_requests=5,
        window_seconds=60,
        redis_url=None,
        sqlite_path=tmp_path / "rate_limit.db",
    )
    limiter._redis_client = FakeRedis()

    first = limiter.check("ip:redis-cost", cost=3)
    second = limiter.check("ip:redis-cost", cost=5)

    assert first.allowed is True
    assert second.allowed is False
    # Both calls must have gone through Redis (not the old SQLite bypass)
    assert len(lua_calls) == 2
    # Each Lua call must have received the correct cost as ARGV[5]
    assert int(lua_calls[0][5]) == 3
    assert int(lua_calls[1][5]) == 5
