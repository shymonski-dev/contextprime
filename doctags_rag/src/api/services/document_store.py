"""Persistent store for processed documents."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
import json
import sqlite3


@dataclass
class StoredDocumentPayload:
    """Serialized representation of a processed document."""

    info: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    markdown: Optional[str]
    text_preview: Optional[str]
    doctags: Optional[Dict[str, Any]]
    message: Optional[str]


class PersistentDocumentStore:
    """SQLite store for processed documents."""

    def __init__(self, db_path: str | Path = "data/storage/documents.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize_database()

    def upsert(self, document_id: str, payload: StoredDocumentPayload) -> None:
        uploaded_at = str(payload.info.get("uploaded_at", ""))
        with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        uploaded_at,
                        info_json,
                        chunks_json,
                        markdown,
                        text_preview,
                        doctags_json,
                        message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        uploaded_at = excluded.uploaded_at,
                        info_json = excluded.info_json,
                        chunks_json = excluded.chunks_json,
                        markdown = excluded.markdown,
                        text_preview = excluded.text_preview,
                        doctags_json = excluded.doctags_json,
                        message = excluded.message
                    """,
                    (
                        document_id,
                        uploaded_at,
                        json.dumps(payload.info, ensure_ascii=True),
                        json.dumps(payload.chunks, ensure_ascii=True),
                        payload.markdown,
                        payload.text_preview,
                        json.dumps(payload.doctags, ensure_ascii=True) if payload.doctags is not None else None,
                        payload.message,
                    ),
                )

    def get(self, document_id: str) -> Optional[StoredDocumentPayload]:
        """Load one persisted document by identifier."""
        document_id_value = str(document_id).strip()
        if not document_id_value:
            return None

        with self._lock:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT
                        document_id,
                        info_json,
                        chunks_json,
                        markdown,
                        text_preview,
                        doctags_json,
                        message
                    FROM documents
                    WHERE document_id = ?
                    LIMIT 1
                    """,
                    (document_id_value,),
                ).fetchone()
        if row is None:
            return None
        return self._row_to_payload(row)

    def count_documents(self) -> int:
        """Return persisted document count."""
        with self._lock:
            with self._connection() as conn:
                value = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return int(value)

    def list_document_infos(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load persisted document info records sorted by newest first."""
        limit_value = max(1, int(limit)) if limit is not None else None
        sql = (
            "SELECT info_json FROM documents "
            "ORDER BY uploaded_at DESC"
        )
        params: tuple[Any, ...] = ()
        if limit_value is not None:
            sql += " LIMIT ?"
            params = (limit_value,)

        with self._lock:
            with self._connection() as conn:
                rows = conn.execute(sql, params).fetchall()
        return [self._decode_json(row[0], default={}) for row in rows]

    def load_all(self, limit: Optional[int] = None) -> Dict[str, StoredDocumentPayload]:
        """Load persisted documents, optionally capped to newest ``limit``."""
        limit_value = max(1, int(limit)) if limit is not None else None
        sql = (
            """
            SELECT
                document_id,
                info_json,
                chunks_json,
                markdown,
                text_preview,
                doctags_json,
                message
            FROM documents
            ORDER BY uploaded_at DESC
            """
        )
        params: tuple[Any, ...] = ()
        if limit_value is not None:
            sql += " LIMIT ?"
            params = (limit_value,)

        results: Dict[str, StoredDocumentPayload] = {}
        with self._lock:
            with self._connection() as conn:
                rows = conn.execute(sql, params).fetchall()

        for row in rows:
            document_id = str(row[0])
            results[document_id] = self._row_to_payload(row)
        return results

    def _initialize_database(self) -> None:
        with self._lock:
            with self._connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        uploaded_at TEXT NOT NULL,
                        info_json TEXT NOT NULL,
                        chunks_json TEXT NOT NULL,
                        markdown TEXT NULL,
                        text_preview TEXT NULL,
                        doctags_json TEXT NULL,
                        message TEXT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON documents(uploaded_at DESC)"
                )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _decode_json(self, raw: Optional[str], default: Any) -> Any:
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def _row_to_payload(self, row: tuple[Any, ...]) -> StoredDocumentPayload:
        info = self._decode_json(row[1], default={})
        chunks = self._decode_json(row[2], default=[])
        doctags = self._decode_json(row[5], default=None)
        return StoredDocumentPayload(
            info=info,
            chunks=chunks,
            markdown=row[3],
            text_preview=row[4],
            doctags=doctags,
            message=row[6],
        )
