"""Persistent store for API runtime counters."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Dict
import sqlite3


class OperationalMetricsStore:
    """SQLite-backed counter store shared across workers."""

    def __init__(self, db_path: str | Path = "data/storage/metrics.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize_database()

    def increment_uploads(self, amount: int = 1) -> int:
        """Increment and return total document upload count."""
        return self.increment("total_uploads", amount=amount)

    def increment_queries(self, amount: int = 1) -> int:
        """Increment and return total query count."""
        return self.increment("total_queries", amount=amount)

    def increment(self, metric_key: str, amount: int = 1) -> int:
        """Increment one metric atomically and return the new value."""
        key = str(metric_key).strip().lower()
        if not key:
            raise ValueError("metric_key must not be empty")
        delta = int(amount)
        if delta == 0:
            return self.get(key)

        with self._lock:
            with self._connection(write=True) as conn:
                conn.execute(
                    """
                    INSERT INTO metrics (metric_key, value)
                    VALUES (?, 0)
                    ON CONFLICT(metric_key) DO NOTHING
                    """,
                    (key,),
                )
                conn.execute(
                    "UPDATE metrics SET value = value + ? WHERE metric_key = ?",
                    (delta, key),
                )
                row = conn.execute(
                    "SELECT value FROM metrics WHERE metric_key = ?",
                    (key,),
                ).fetchone()

        return int(row[0]) if row else 0

    def get(self, metric_key: str) -> int:
        """Read one metric value."""
        key = str(metric_key).strip().lower()
        if not key:
            return 0

        with self._lock:
            with self._connection(write=False) as conn:
                row = conn.execute(
                    "SELECT value FROM metrics WHERE metric_key = ?",
                    (key,),
                ).fetchone()
        return int(row[0]) if row else 0

    def get_snapshot(self) -> Dict[str, int]:
        """Return all tracked metric values."""
        with self._lock:
            with self._connection(write=False) as conn:
                rows = conn.execute(
                    "SELECT metric_key, value FROM metrics ORDER BY metric_key ASC"
                ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def _initialize_database(self) -> None:
        with self._lock:
            with self._connection(write=False) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metrics (
                        metric_key TEXT PRIMARY KEY,
                        value INTEGER NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO metrics (metric_key, value)
                    VALUES ('total_uploads', 0)
                    ON CONFLICT(metric_key) DO NOTHING
                    """
                )
                conn.execute(
                    """
                    INSERT INTO metrics (metric_key, value)
                    VALUES ('total_queries', 0)
                    ON CONFLICT(metric_key) DO NOTHING
                    """
                )

    @contextmanager
    def _connection(self, write: bool):
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
