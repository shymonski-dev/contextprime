"""
Production retrieval feedback capture store.

This module records:
- Retrieval query events with returned candidates
- User feedback events tied to a query identifier
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional
import hashlib
import json
import sqlite3
import uuid

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None


class FeedbackCaptureStore:
    """SQLite-backed store for retrieval query and feedback events."""

    def __init__(
        self,
        root_dir: str | Path = "data/feedback",
        *,
        db_name: str = "retrieval_feedback.db",
        mirror_jsonl: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root_dir / db_name
        self.mirror_jsonl = bool(mirror_jsonl)
        self.query_events_path = self.root_dir / "retrieval_query_events.jsonl"
        self.feedback_events_path = self.root_dir / "retrieval_feedback_events.jsonl"
        self._lock = Lock()
        self._initialize_database()

    def record_query_event(
        self,
        *,
        query: str,
        request_payload: Dict[str, Any],
        results: Iterable[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a retrieval query event and return a generated query identifier."""
        recorded_at = self._utc_now()
        query_id = self._build_event_id(prefix="qry", seed=f"{query}|{recorded_at}")
        normalized_results = self._normalize_results(results)
        payload = {
            "event_type": "retrieval_query",
            "query_id": query_id,
            "recorded_at": recorded_at,
            "query": str(query),
            "request": dict(request_payload or {}),
            "results": normalized_results,
            "metadata": dict(metadata or {}),
        }

        with self._lock:
            with self._db_connection(write=True) as conn:
                conn.execute(
                    """
                    INSERT INTO query_events (
                        query_id,
                        recorded_at,
                        query,
                        request_json,
                        results_json,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        recorded_at,
                        payload["query"],
                        json.dumps(payload["request"], ensure_ascii=True),
                        json.dumps(payload["results"], ensure_ascii=True),
                        json.dumps(payload["metadata"], ensure_ascii=True),
                    ),
                )
            if self.mirror_jsonl:
                self._append_jsonl(self.query_events_path, payload)
        return query_id

    def record_feedback_event(
        self,
        *,
        query_id: str,
        helpful: Optional[bool] = None,
        selected_result_ids: Optional[Iterable[str]] = None,
        result_labels: Optional[Iterable[Dict[str, Any]]] = None,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist a retrieval feedback event and return a generated feedback identifier."""
        normalized_query_id = str(query_id).strip()
        if not normalized_query_id:
            raise ValueError("query_id is required")

        selected_ids = self._normalize_ids(selected_result_ids or [])
        labels = self._normalize_labels(result_labels or [])
        feedback_id = self._build_event_id(
            prefix="fbk",
            seed=f"{normalized_query_id}|{helpful}|{selected_ids}|{labels}",
        )
        payload = {
            "event_type": "retrieval_feedback",
            "feedback_id": feedback_id,
            "query_id": normalized_query_id,
            "recorded_at": self._utc_now(),
            "helpful": helpful if helpful is None else bool(helpful),
            "selected_result_ids": selected_ids,
            "result_labels": labels,
            "comment": (comment or "").strip(),
            "user_id": (user_id or "").strip() or None,
            "metadata": dict(metadata or {}),
        }

        with self._lock:
            try:
                with self._db_connection(write=True) as conn:
                    conn.execute(
                        """
                        INSERT INTO feedback_events (
                            feedback_id,
                            query_id,
                            recorded_at,
                            helpful,
                            selected_result_ids_json,
                            result_labels_json,
                            comment,
                            user_id,
                            metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["feedback_id"],
                            payload["query_id"],
                            payload["recorded_at"],
                            self._coerce_helpful(payload["helpful"]),
                            json.dumps(payload["selected_result_ids"], ensure_ascii=True),
                            json.dumps(payload["result_labels"], ensure_ascii=True),
                            payload["comment"],
                            payload["user_id"],
                            json.dumps(payload["metadata"], ensure_ascii=True),
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                if "FOREIGN KEY" in str(exc).upper():
                    raise ValueError(f"Unknown query_id: {normalized_query_id}") from exc
                raise
            if self.mirror_jsonl:
                self._append_jsonl(self.feedback_events_path, payload)
        return feedback_id

    def query_exists(self, query_id: str) -> bool:
        """Return True when a query identifier exists in the query event store."""
        target = str(query_id).strip()
        if not target:
            return False
        with self._lock:
            with self._db_connection(write=False) as conn:
                row = conn.execute(
                    "SELECT 1 FROM query_events WHERE query_id = ? LIMIT 1",
                    (target,),
                ).fetchone()
        return row is not None

    def get_statistics(self) -> Dict[str, Any]:
        """Return basic event counters for monitoring."""
        with self._lock:
            with self._db_connection(write=False) as conn:
                query_events = conn.execute("SELECT COUNT(*) FROM query_events").fetchone()[0]
                feedback_events = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]

        return {
            "query_events": int(query_events),
            "feedback_events": int(feedback_events),
            "root_dir": str(self.root_dir),
            "database_path": str(self.db_path),
        }

    def load_query_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load query events from indexed storage."""
        limit_value = max(1, int(limit)) if limit is not None else None
        sql = (
            "SELECT query_id, recorded_at, query, request_json, results_json, metadata_json "
            "FROM query_events ORDER BY recorded_at DESC"
        )
        params: tuple[Any, ...] = ()
        if limit_value is not None:
            sql += " LIMIT ?"
            params = (limit_value,)

        rows: List[tuple[Any, ...]]
        with self._lock:
            with self._db_connection(write=False) as conn:
                rows = conn.execute(sql, params).fetchall()

        events: List[Dict[str, Any]] = []
        for row in rows:
            events.append(
                {
                    "event_type": "retrieval_query",
                    "query_id": row[0],
                    "recorded_at": row[1],
                    "query": row[2],
                    "request": self._decode_json(row[3], default={}),
                    "results": self._decode_json(row[4], default=[]),
                    "metadata": self._decode_json(row[5], default={}),
                }
            )
        return events

    def load_feedback_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load feedback events from indexed storage."""
        limit_value = max(1, int(limit)) if limit is not None else None
        sql = (
            "SELECT feedback_id, query_id, recorded_at, helpful, selected_result_ids_json, "
            "result_labels_json, comment, user_id, metadata_json "
            "FROM feedback_events ORDER BY recorded_at DESC"
        )
        params: tuple[Any, ...] = ()
        if limit_value is not None:
            sql += " LIMIT ?"
            params = (limit_value,)

        rows: List[tuple[Any, ...]]
        with self._lock:
            with self._db_connection(write=False) as conn:
                rows = conn.execute(sql, params).fetchall()

        events: List[Dict[str, Any]] = []
        for row in rows:
            helpful_raw = row[3]
            helpful_value = None if helpful_raw is None else bool(int(helpful_raw))
            events.append(
                {
                    "event_type": "retrieval_feedback",
                    "feedback_id": row[0],
                    "query_id": row[1],
                    "recorded_at": row[2],
                    "helpful": helpful_value,
                    "selected_result_ids": self._decode_json(row[4], default=[]),
                    "result_labels": self._decode_json(row[5], default=[]),
                    "comment": row[6] or "",
                    "user_id": row[7],
                    "metadata": self._decode_json(row[8], default={}),
                }
            )
        return events

    def _normalize_results(self, results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in results:
            row = dict(item or {})
            result_id = str(row.get("id", "")).strip()
            if not result_id:
                continue
            content = str(row.get("content", ""))
            normalized.append(
                {
                    "id": result_id,
                    "content": self._clip_text(content, max_chars=2400),
                    "score": float(row.get("score", 0.0)),
                    "confidence": float(row.get("confidence", 0.0)),
                    "source": str(row.get("source", "")),
                    "metadata": dict(row.get("metadata") or {}),
                }
            )
        return normalized

    def _normalize_ids(self, values: Iterable[str]) -> List[str]:
        seen = set()
        normalized: List[str] = []
        for value in values:
            token = str(value).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)
        return normalized

    def _normalize_labels(self, labels: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for row in labels:
            payload = dict(row or {})
            result_id = str(payload.get("result_id", "")).strip()
            if not result_id:
                continue
            label = int(payload.get("label", 0))
            if label not in (0, 1):
                continue
            normalized.append(
                {
                    "result_id": result_id,
                    "label": label,
                    "note": self._clip_text(str(payload.get("note", "")), max_chars=500),
                }
            )
        return normalized

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True, default=str)
        with path.open("a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line + "\n")
            handle.flush()
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _clip_text(self, value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3] + "..."

    def _build_event_id(self, prefix: str, seed: str) -> str:
        now = self._utc_now()
        digest = hashlib.sha1(f"{seed}|{uuid.uuid4().hex}".encode("utf-8")).hexdigest()[:12]
        compact_time = now.replace("-", "").replace(":", "").replace(".", "")
        return f"{prefix}_{compact_time}_{digest}"

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _coerce_helpful(self, helpful: Optional[bool]) -> Optional[int]:
        if helpful is None:
            return None
        return 1 if bool(helpful) else 0

    def _decode_json(self, raw: Optional[str], default: Any) -> Any:
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def _initialize_database(self) -> None:
        with self._lock:
            with self._db_connection(write=False) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS query_events (
                        query_id TEXT PRIMARY KEY,
                        recorded_at TEXT NOT NULL,
                        query TEXT NOT NULL,
                        request_json TEXT NOT NULL,
                        results_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback_events (
                        feedback_id TEXT PRIMARY KEY,
                        query_id TEXT NOT NULL,
                        recorded_at TEXT NOT NULL,
                        helpful INTEGER NULL,
                        selected_result_ids_json TEXT NOT NULL,
                        result_labels_json TEXT NOT NULL,
                        comment TEXT NOT NULL,
                        user_id TEXT NULL,
                        metadata_json TEXT NOT NULL,
                        FOREIGN KEY (query_id) REFERENCES query_events(query_id)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_query_events_recorded_at ON query_events(recorded_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_feedback_events_query_id ON feedback_events(query_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_feedback_events_recorded_at ON feedback_events(recorded_at DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_feedback_events_user_id ON feedback_events(user_id)"
                )

    @contextmanager
    def _db_connection(self, *, write: bool = False):
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
