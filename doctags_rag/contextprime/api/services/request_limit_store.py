"""Shared request limiting with Redis primary and SQLite fallback."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from time import monotonic, time
from typing import Optional
import secrets
import sqlite3

from loguru import logger

try:
    import redis
except Exception:  # pragma: no cover - optional dependency guard
    redis = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RateLimitDecision:
    """Decision returned by the request limiter."""

    allowed: bool
    retry_after_seconds: int = 0


_RATE_LIMIT_SCHEMA_VERSION = 2

_RATE_LIMIT_MIGRATIONS = [
    (
        1,
        """CREATE TABLE IF NOT EXISTS rate_limit_events (
            subject TEXT NOT NULL,
            event_time REAL NOT NULL,
            cost INTEGER NOT NULL DEFAULT 1
        )""",
    ),
    (
        2,
        # cost column already present in v1 DDL; this migration is a no-op guard
        # for databases upgraded from a version that pre-dated the cost column.
        "SELECT 1",
    ),
]


class SQLiteSlidingWindowRateLimiter:
    """SQLite-backed sliding window limiter shared across workers."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        max_requests: int,
        window_seconds: int,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self._lock = Lock()
        self._last_cleanup = monotonic()
        self._initialize_database()

    def check(self, subject: str, cost: int = 1) -> RateLimitDecision:
        """Return allow or deny decision for one subject key."""
        key = str(subject).strip() or "unknown"
        cost_units = max(1, int(cost))
        now = float(time())
        window_start = now - self.window_seconds

        with self._lock:
            with self._connection(write=True) as conn:
                conn.execute(
                    "DELETE FROM rate_limit_events WHERE subject = ? AND event_time <= ?",
                    (key, window_start),
                )
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost), 0) FROM rate_limit_events WHERE subject = ?",
                    (key,),
                ).fetchone()
                consumed = int(row[0]) if row else 0
                if consumed + cost_units > self.max_requests:
                    oldest_row = conn.execute(
                        """
                        SELECT event_time
                        FROM rate_limit_events
                        WHERE subject = ?
                        ORDER BY event_time ASC
                        LIMIT 1
                        """,
                        (key,),
                    ).fetchone()
                    retry_after = 1
                    if oldest_row:
                        retry_after = max(
                            1,
                            int((float(oldest_row[0]) + self.window_seconds) - now),
                        )
                    return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)

                conn.execute(
                    "INSERT INTO rate_limit_events (subject, event_time, cost) VALUES (?, ?, ?)",
                    (key, now, cost_units),
                )
                self._cleanup_stale_rows(conn=conn, window_start=window_start)

        return RateLimitDecision(allowed=True, retry_after_seconds=0)

    def _cleanup_stale_rows(self, *, conn, window_start: float) -> None:
        now_monotonic = monotonic()
        if (now_monotonic - self._last_cleanup) < 60:
            return
        self._last_cleanup = now_monotonic
        conn.execute(
            "DELETE FROM rate_limit_events WHERE event_time <= ?",
            (window_start,),
        )

    def _initialize_database(self) -> None:
        with self._lock:
            with self._connection(write=False) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                self._apply_migrations(conn)
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rate_limit_subject_time
                    ON rate_limit_events(subject, event_time)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rate_limit_time
                    ON rate_limit_events(event_time)
                    """
                )

    def _apply_migrations(self, conn) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
        )
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current = row[0] if row else 0
        for version, sql in _RATE_LIMIT_MIGRATIONS:
            if current < version:
                conn.execute(sql)
                if current == 0:
                    conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
                else:
                    conn.execute("UPDATE schema_version SET version = ?", (version,))
                current = version

    @contextmanager
    def _connection(self, *, write: bool):
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class SharedSlidingWindowRateLimiter:
    """Redis-backed limiter with SQLite fallback when Redis is unavailable."""

    _LUA_CHECK_SCRIPT = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local member = ARGV[4]
local cost = tonumber(ARGV[5])
local window_start = now_ms - window_ms

redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
local count = redis.call('ZCARD', key)
if count + cost > max_requests then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry_after = 1
  if oldest[2] ~= nil then
    retry_after = math.ceil(((tonumber(oldest[2]) + window_ms) - now_ms) / 1000)
    if retry_after < 1 then
      retry_after = 1
    end
  end
  return {0, retry_after}
end

redis.call('ZADD', key, now_ms, member)
redis.call('EXPIRE', key, math.ceil(window_ms / 1000) + 2)
return {1, 0}
"""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: int = 60,
        redis_url: Optional[str] = None,
        sqlite_path: str | Path = "data/storage/rate_limit.db",
    ) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1, int(window_seconds))
        self._sqlite = SQLiteSlidingWindowRateLimiter(
            db_path=sqlite_path,
            max_requests=self.max_requests,
            window_seconds=self.window_seconds,
        )
        self._redis_url = (redis_url or "").strip() or None
        self._redis_client = self._build_redis_client(self._redis_url)

    def check(self, subject: str, cost: int = 1) -> RateLimitDecision:
        """Check request allowance using Redis or SQLite fallback."""
        cost_units = max(1, int(cost))
        client = self._redis_client
        if client is None:
            return self._sqlite.check(subject, cost=cost_units)

        redis_key = self._redis_subject_key(subject)
        now_ms = int(time() * 1000)
        window_ms = int(self.window_seconds * 1000)
        member = f"{now_ms}:{secrets.token_hex(6)}"

        try:
            result = client.eval(
                self._LUA_CHECK_SCRIPT,
                1,
                redis_key,
                now_ms,
                window_ms,
                self.max_requests,
                member,
                cost_units,
            )
            allowed = bool(int(result[0])) if isinstance(result, (list, tuple)) else False
            retry_after = int(result[1]) if isinstance(result, (list, tuple)) else 1
            return RateLimitDecision(
                allowed=allowed,
                retry_after_seconds=0 if allowed else max(1, retry_after),
            )
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            logger.warning(
                "Redis limiter is unavailable, falling back to SQLite limiter: {}",
                exc,
            )
            self._redis_client = None
            return self._sqlite.check(subject, cost=cost_units)

    def _build_redis_client(self, redis_url: Optional[str]):
        if not redis_url:
            return None
        if redis is None:
            logger.warning("Redis package is unavailable; request limiting uses SQLite only.")
            return None
        try:
            client = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
                decode_responses=False,
            )
            client.ping()
            logger.info("Request limiter using Redis backend at {}", redis_url)
            return client
        except Exception as exc:  # pragma: no cover - depends on runtime availability
            logger.warning(
                "Redis is not reachable at {}. Using SQLite request limiter. {}",
                redis_url,
                exc,
            )
            return None

    def _redis_subject_key(self, subject: str) -> str:
        key = str(subject).strip() or "unknown"
        key_hash = sha256(key.encode("utf-8")).hexdigest()[:32]
        return f"doctags:rate_limit:{key_hash}"
