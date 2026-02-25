"""
Qdrant Vector Database Manager for Contextprime.

Provides comprehensive vector database operations including:
- Connection management
- Collection creation and management
- Vector insertion and batch operations
- Similarity search with filtering
- Collection statistics and monitoring
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS
import time
import math
import re
import json
from collections import Counter, defaultdict
from functools import wraps
from urllib import error as urllib_error
from urllib import request as urllib_request

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SearchParams,
    UpdateStatus,
    CollectionInfo,
)
from qdrant_client.http.exceptions import (
    UnexpectedResponse,
    ResponseHandlingException,
)
from qdrant_client.http.models import UpdateResult
from loguru import logger

from ..core.config import QdrantConfig, get_settings


@dataclass
class VectorPoint:
    """Represents a vector point with metadata."""
    id: Union[str, int, UUID]
    vector: List[float]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Represents a search result from Qdrant."""
    id: Union[str, int, UUID]
    score: float
    vector: Optional[List[float]]
    metadata: Dict[str, Any]


def retry_on_connection_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations on connection errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (UnexpectedResponse, ConnectionError, TimeoutError) as e:
                    message = str(e).lower()
                    if (
                        isinstance(e, UnexpectedResponse)
                        and ("404" in message or "doesn't exist" in message or "not found" in message)
                    ):
                        logger.warning(
                            f"Non-retryable Qdrant response in {func.__name__}: {e}"
                        )
                        raise

                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (2 ** attempt)
                        logger.warning(
                            f"Connection error in {func.__name__}, "
                            f"retrying in {sleep_time}s (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Max retries reached for {func.__name__}")
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            raise last_exception
        return wrapper
    return decorator


def normalize_point_id(point_id: Optional[Union[str, int, UUID]] = None) -> Union[str, UUID]:
    """
    Normalize point ID to be compatible with Qdrant v1.14+.

    Qdrant v1.14+ requires IDs to be either:
    - Unsigned integers
    - Valid UUIDs

    Args:
        point_id: Input ID (str, int, UUID, or None)

    Returns:
        Normalized ID (UUID or int)
    """
    if point_id is None:
        return str(uuid4())

    if isinstance(point_id, UUID):
        return str(point_id)

    if isinstance(point_id, int):
        return point_id

    if isinstance(point_id, str):
        # Check if it's already a valid UUID string
        try:
            return str(UUID(point_id))
        except ValueError:
            # Convert arbitrary string to deterministic UUID
            return str(uuid5(NAMESPACE_DNS, point_id))

    # Fallback: generate random UUID
    return str(uuid4())


class QdrantManager:
    """
    Manages Qdrant vector database operations with connection management and error handling.

    Supports:
    - Collection management
    - Vector CRUD operations
    - Similarity search with filtering
    - Batch operations
    - Collection statistics
    """

    _LEXICAL_STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "in",
        "on", "at", "to", "for", "of", "with", "by", "from", "as", "is",
        "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "what", "who", "where", "when", "why", "how", "which", "that",
        "this", "these", "those", "it", "its", "their", "there", "can",
        "could", "would", "should", "may", "might", "will", "about",
    }

    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialize Qdrant manager.

        Args:
            config: Qdrant configuration. If None, loads from global settings.
        """
        if config is None:
            settings = get_settings()
            config = settings.qdrant

        self.config = config
        self.client: Optional[QdrantClient] = None
        self._connected = False

        # Distance metric mapping
        self._distance_metrics = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }

        # Initialize connection
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Qdrant client."""
        try:
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                timeout=30,
                check_compatibility=False,
            )
            # Test connection
            self.client.get_collections()
            self._connected = True
            logger.info(f"Qdrant client initialized: {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if Qdrant connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if not self.client:
                return False
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the Qdrant client and cleanup resources."""
        if self.client:
            try:
                self.client.close()
            except:
                pass
            self._connected = False
            logger.info("Qdrant client closed")

    @retry_on_connection_error(max_retries=3)
    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance_metric: Optional[str] = None,
        recreate: bool = False
    ) -> bool:
        """
        Create a collection for storing vectors.

        Args:
            collection_name: Name of the collection (uses config default if None)
            vector_size: Vector dimension (uses config default if None)
            distance_metric: Distance metric (cosine, euclidean, dot)
            recreate: If True, delete existing collection first

        Returns:
            True if collection was created
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        vector_size = vector_size or self.config.vector_size
        distance_metric = distance_metric or self.config.distance_metric

        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists:
            if recreate:
                logger.info(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
            else:
                logger.info(f"Collection already exists: {collection_name}")
                return True

        # Create collection
        distance = self._distance_metrics.get(distance_metric, Distance.COSINE)

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )
            logger.info(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    @retry_on_connection_error(max_retries=3)
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

    @retry_on_connection_error(max_retries=3)
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection was deleted
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name

        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    @retry_on_connection_error(max_retries=3)
    def insert_vector(
        self,
        vector: List[float],
        metadata: Dict[str, Any],
        vector_id: Optional[Union[str, int, UUID]] = None,
        collection_name: Optional[str] = None
    ) -> Union[str, int, UUID]:
        """
        Insert a single vector with metadata.

        Args:
            vector: Vector embeddings
            metadata: Metadata to store with vector
            vector_id: Optional ID for the vector (generated if None)
            collection_name: Collection name (uses config default if None)

        Returns:
            ID of the inserted vector
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        vector_id = normalize_point_id(vector_id)

        point = PointStruct(
            id=vector_id,
            vector=vector,
            payload=metadata,
        )

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[point],
            )
            return vector_id
        except Exception as e:
            logger.error(f"Failed to insert vector: {e}")
            raise

    @retry_on_connection_error(max_retries=3)
    def insert_vectors_batch(
        self,
        vectors: List[VectorPoint],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> int:
        """
        Insert multiple vectors in batches.

        Args:
            vectors: List of VectorPoint objects
            collection_name: Collection name
            batch_size: Number of vectors per batch

        Returns:
            Number of vectors inserted
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        total_inserted = 0

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]

            points = [
                PointStruct(
                    id=normalize_point_id(v.id),
                    vector=v.vector,
                    payload=v.metadata,
                )
                for v in batch
            ]

            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                total_inserted += len(points)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(points)} vectors")
            except Exception as e:
                logger.error(f"Failed to insert batch: {e}")
                raise

        logger.info(f"Total vectors inserted: {total_inserted}")
        return total_inserted

    @retry_on_connection_error(max_retries=3)
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Perform similarity search.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            collection_name: Collection name
            score_threshold: Minimum score threshold

        Returns:
            List of search results
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name

        # Build filter object
        filter_obj = None
        if filters:
            filter_obj = self._build_filter(filters)

        try:
            if hasattr(self.client, "query_points"):
                query_response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=filter_obj,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                results = query_response.points
            elif hasattr(self.client, "search"):
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=filter_obj,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                raise AttributeError(
                    "Qdrant client does not provide query_points or search"
                )

            return [
                SearchResult(
                    id=r.id,
                    score=r.score,
                    vector=r.vector,
                    metadata=r.payload or {},
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_lexical(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        max_scan_points: int = 1500,
        scan_ratio: float = 0.02,
        max_scan_cap: int = 20000,
        page_size: int = 200,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> List[SearchResult]:
        """
        Perform lexical retrieval with BM25 scoring over indexed payload text.

        Notes:
            - Uses scroll pagination and scores only scanned points.
            - Intended as sparse lexical signal for hybrid fusion.
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        query_terms = self._tokenize_for_lexical(query_text)
        if not query_terms:
            return []

        filter_obj = self._build_filter(filters) if filters else None
        effective_scan_points = self._resolve_lexical_scan_budget(
            collection_name=collection_name,
            filter_obj=filter_obj,
            top_k=top_k,
            page_size=page_size,
            max_scan_points=max_scan_points,
            scan_ratio=scan_ratio,
            max_scan_cap=max_scan_cap,
        )

        scanned: List[SearchResult] = []
        offset = None

        while len(scanned) < effective_scan_points:
            remaining = effective_scan_points - len(scanned)
            limit = max(1, min(page_size, remaining))
            page, next_offset = self.scroll_collection(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_vectors=False,
                filters=filters,
            )
            if not page:
                break

            scanned.extend(page)
            if next_offset is None:
                break
            offset = next_offset

        if not scanned:
            return []

        tokenized_docs: List[Tuple[SearchResult, List[str]]] = []
        term_df: Dict[str, int] = defaultdict(int)

        for item in scanned:
            text = self._extract_text_from_payload(item.metadata)
            if not text:
                continue
            tokens = self._tokenize_for_lexical(text)
            if not tokens:
                continue
            tokenized_docs.append((item, tokens))
            unique_tokens = set(tokens)
            for term in query_terms:
                if term in unique_tokens:
                    term_df[term] += 1

        if not tokenized_docs:
            return []

        total_docs = len(tokenized_docs)
        avg_doc_len = sum(len(tokens) for _, tokens in tokenized_docs) / max(1, total_docs)
        query_tf = Counter(query_terms)

        scored: List[SearchResult] = []
        for item, tokens in tokenized_docs:
            doc_len = len(tokens)
            doc_tf = Counter(tokens)
            score = 0.0

            for term, qtf in query_tf.items():
                tf = doc_tf.get(term, 0)
                if tf <= 0:
                    continue
                df = term_df.get(term, 0)
                idf = math.log(((total_docs - df + 0.5) / (df + 0.5)) + 1.0)
                denom = tf + bm25_k1 * (1.0 - bm25_b + bm25_b * (doc_len / max(1.0, avg_doc_len)))
                score += idf * ((tf * (bm25_k1 + 1.0)) / max(1e-9, denom)) * qtf

            if score <= 0:
                continue

            scored.append(
                SearchResult(
                    id=item.id,
                    score=float(score),
                    vector=None,
                    metadata=item.metadata,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _resolve_lexical_scan_budget(
        self,
        *,
        collection_name: str,
        filter_obj: Optional[Filter],
        top_k: int,
        page_size: int,
        max_scan_points: int,
        scan_ratio: float,
        max_scan_cap: int,
    ) -> int:
        """Choose an adaptive lexical scan budget based on collection size."""
        base_scan = max(100, int(max_scan_points))
        fallback_scan = max(base_scan, int(top_k) * max(20, int(page_size)))

        estimated_count = self._estimate_collection_size(
            collection_name=collection_name,
            filter_obj=filter_obj,
        )
        if estimated_count is None:
            return fallback_scan

        ratio = min(1.0, max(0.0, float(scan_ratio)))
        ratio_scan = int(math.ceil(estimated_count * ratio))
        effective_scan = max(fallback_scan, ratio_scan)

        if max_scan_cap > 0:
            effective_scan = min(effective_scan, int(max_scan_cap))
        return max(100, effective_scan)

    def _estimate_collection_size(
        self,
        *,
        collection_name: str,
        filter_obj: Optional[Filter],
    ) -> Optional[int]:
        """Estimate searchable collection size for lexical budget selection."""
        if not self.client:
            return None
        try:
            count_result = self.client.count(
                collection_name=collection_name,
                count_filter=filter_obj,
                exact=False,
            )
            value = getattr(count_result, "count", None)
            if value is None:
                return None
            return max(0, int(value))
        except Exception as err:
            logger.debug("Unable to estimate Qdrant collection size for lexical scan: %s", err)
            return None

    def _extract_text_from_payload(self, payload: Optional[Dict[str, Any]]) -> str:
        """Extract the best available text field from payload metadata."""
        if not payload:
            return ""
        for key in ("text", "content", "chunk_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _tokenize_for_lexical(self, text: str) -> List[str]:
        """Tokenize text for lexical scoring."""
        tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        return [t for t in tokens if t not in self._LEXICAL_STOPWORDS]

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from dictionary.

        Args:
            filters: Dictionary of filter conditions

        Returns:
            Qdrant Filter object
        """
        conditions = []

        for key, value in filters.items():
            if isinstance(value, (str, int, bool)):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
            elif isinstance(value, list):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value),
                    )
                )
            elif isinstance(value, dict):
                # Handle range queries
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                                gt=value.get("gt"),
                                lt=value.get("lt"),
                            ),
                        )
                    )

        return Filter(must=conditions) if conditions else None

    @retry_on_connection_error(max_retries=3)
    def get_vector(
        self,
        vector_id: Union[str, int, UUID],
        collection_name: Optional[str] = None,
        with_vector: bool = True
    ) -> Optional[SearchResult]:
        """
        Get a vector by ID.

        Args:
            vector_id: Vector ID
            collection_name: Collection name
            with_vector: Include vector embeddings in result

        Returns:
            Search result or None if not found
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        vector_id = normalize_point_id(vector_id)

        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=with_vector,
            )

            if not result:
                return None

            point = result[0]
            return SearchResult(
                id=point.id,
                score=1.0,  # No score for direct retrieval
                vector=point.vector,
                metadata=point.payload or {},
            )
        except Exception as e:
            logger.error(f"Failed to get vector: {e}")
            return None

    @retry_on_connection_error(max_retries=3)
    def update_vector(
        self,
        vector_id: Union[str, int, UUID],
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Update a vector's embeddings or metadata.

        Args:
            vector_id: Vector ID
            vector: New vector embeddings (optional)
            metadata: New metadata (optional)
            collection_name: Collection name

        Returns:
            True if vector was updated
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        if vector is None and metadata is None:
            logger.warning("No updates provided")
            return False

        collection_name = collection_name or self.config.collection_name
        vector_id = normalize_point_id(vector_id)

        try:
            # If updating vector, use upsert
            if vector is not None:
                point = PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=metadata or {},
                )
                self.client.upsert(
                    collection_name=collection_name,
                    points=[point],
                )
            # If only updating metadata, use set_payload
            elif metadata is not None:
                self.client.set_payload(
                    collection_name=collection_name,
                    payload=metadata,
                    points=[vector_id],
                )

            return True
        except Exception as e:
            logger.error(f"Failed to update vector: {e}")
            return False

    @retry_on_connection_error(max_retries=3)
    def delete_vector(
        self,
        vector_id: Union[str, int, UUID],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete a vector by ID.

        Args:
            vector_id: Vector ID
            collection_name: Collection name

        Returns:
            True if vector was deleted
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        vector_id = normalize_point_id(vector_id)

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=[vector_id],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector: {e}")
            return False

    @retry_on_connection_error(max_retries=3)
    def delete_vectors_by_filter(
        self,
        filters: Dict[str, Any],
        collection_name: Optional[str] = None
    ) -> int:
        """
        Delete vectors matching filter criteria.

        Args:
            filters: Filter conditions
            collection_name: Collection name

        Returns:
            Number of vectors deleted
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        filter_obj = self._build_filter(filters)

        try:
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=filter_obj,
            )
            # Note: Qdrant doesn't return count of deleted items
            logger.info(f"Deleted vectors matching filter")
            return 0  # Return 0 as we don't know the count
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return 0

    @retry_on_connection_error(max_retries=3)
    def get_collection_info(
        self,
        collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get collection information and statistics.

        Args:
            collection_name: Collection name

        Returns:
            Dictionary with collection information
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name

        try:
            info = self.client.get_collection(collection_name)
            parsed = self._parse_collection_info_payload(
                collection_name=collection_name,
                payload=info,
            )
            if parsed:
                return parsed
        except ResponseHandlingException as err:
            logger.warning(
                f"Qdrant client could not parse collection info for {collection_name}: {err}. "
                "Falling back to HTTP."
            )
        except Exception as e:
            logger.warning(
                f"Client collection info lookup failed for {collection_name}: {e}. "
                "Falling back to HTTP."
            )

        return self._get_collection_info_via_http(collection_name)

    def _parse_collection_info_payload(
        self,
        collection_name: str,
        payload: Any,
    ) -> Optional[Dict[str, Any]]:
        """Parse collection info returned by the client or HTTP fallback."""
        if payload is None:
            return None

        vectors_count = self._coerce_int(
            self._lookup_nested(payload, ["vectors_count"])
        )
        points_count = self._coerce_int(
            self._lookup_nested(payload, ["points_count"])
        )
        segments_count = self._coerce_int(
            self._lookup_nested(payload, ["segments_count"])
        )
        status = self._stringify(
            self._lookup_nested(payload, ["status"])
        )
        optimizer_status = self._stringify(
            self._lookup_nested(payload, ["optimizer_status"])
        )

        vectors_cfg = (
            self._lookup_nested(payload, ["config", "params", "vectors"])
            or self._lookup_nested(payload, ["result", "config", "params", "vectors"])
        )
        vector_size, distance = self._extract_vector_config(vectors_cfg)

        return {
            "name": collection_name,
            "vectors_count": vectors_count,
            "points_count": points_count,
            "segments_count": segments_count,
            "status": status,
            "optimizer_status": optimizer_status,
            "vector_size": vector_size,
            "distance": distance,
        }

    def _get_collection_info_via_http(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Fetch collection info directly over HTTP for server-client compatibility."""
        base_url = self._build_base_http_url()
        url = f"{base_url}/collections/{collection_name}"
        request = urllib_request.Request(url, method="GET")
        if self.config.api_key:
            request.add_header("api-key", self.config.api_key)

        try:
            with urllib_request.urlopen(request, timeout=10) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body)
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as err:
            logger.error(f"Failed to fetch collection info via HTTP ({url}): {err}")
            return None

        parsed = self._parse_collection_info_payload(
            collection_name=collection_name,
            payload=payload.get("result", payload),
        )
        if parsed is None:
            logger.error(f"Collection info HTTP payload missing expected fields for {collection_name}")
            return None
        return parsed

    def _build_base_http_url(self) -> str:
        host = str(self.config.host).strip()
        if host.startswith("http://") or host.startswith("https://"):
            return host.rstrip("/")
        return f"http://{host}:{self.config.port}"

    def _lookup_nested(self, payload: Any, path: List[str]) -> Any:
        current = payload
        for key in path:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    def _extract_vector_config(self, vectors_cfg: Any) -> Tuple[Optional[int], Optional[str]]:
        if vectors_cfg is None:
            return None, None

        if isinstance(vectors_cfg, dict):
            if "size" in vectors_cfg:
                return self._coerce_int(vectors_cfg.get("size")), self._stringify(vectors_cfg.get("distance"))

            default_cfg = vectors_cfg.get("default")
            if isinstance(default_cfg, dict):
                return self._coerce_int(default_cfg.get("size")), self._stringify(default_cfg.get("distance"))

            for value in vectors_cfg.values():
                if isinstance(value, dict) and "size" in value:
                    return self._coerce_int(value.get("size")), self._stringify(value.get("distance"))
            return None, None

        size = getattr(vectors_cfg, "size", None)
        distance = getattr(vectors_cfg, "distance", None)
        if size is not None:
            return self._coerce_int(size), self._stringify(distance)

        for attr in ("default", "text", "image"):
            nested_cfg = getattr(vectors_cfg, attr, None)
            if nested_cfg is not None:
                nested_size = getattr(nested_cfg, "size", None)
                if nested_size is not None:
                    nested_distance = getattr(nested_cfg, "distance", None)
                    return self._coerce_int(nested_size), self._stringify(nested_distance)

        return None, None

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _stringify(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        candidate = getattr(value, "value", value)
        if candidate is None:
            return None
        return str(candidate)

    @retry_on_connection_error(max_retries=3)
    def scroll_collection(
        self,
        collection_name: Optional[str] = None,
        limit: int = 100,
        offset: Optional[Union[str, int, UUID]] = None,
        with_vectors: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[SearchResult], Optional[Union[str, int, UUID]]]:
        """
        Scroll through collection points.

        Args:
            collection_name: Collection name
            limit: Number of points to return
            offset: Offset for pagination
            with_vectors: Include vectors in results
            filters: Optional filters

        Returns:
            Tuple of (results, next_offset)
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = collection_name or self.config.collection_name
        filter_obj = self._build_filter(filters) if filters else None

        try:
            result = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
                scroll_filter=filter_obj,
            )

            points, next_offset = result

            search_results = [
                SearchResult(
                    id=p.id,
                    score=1.0,
                    vector=p.vector,
                    metadata=p.payload or {},
                )
                for p in points
            ]

            return search_results, next_offset
        except Exception as e:
            logger.error(f"Failed to scroll collection: {e}")
            return [], None

    def get_statistics(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed statistics for the collection.

        Args:
            collection_name: Collection name

        Returns:
            Dictionary with statistics
        """
        info = self.get_collection_info(collection_name)

        if not info:
            return {}

        return {
            "collection_name": info["name"],
            "total_vectors": info["vectors_count"],
            "total_points": info["points_count"],
            "segments": info["segments_count"],
            "status": info["status"],
            "vector_dimension": info["vector_size"],
            "distance_metric": info["distance"],
        }

    def clear_collection(
        self,
        collection_name: Optional[str] = None,
        confirm: bool = False
    ) -> bool:
        """
        Clear all vectors from a collection.

        Args:
            collection_name: Collection name
            confirm: Must be True to execute

        Returns:
            True if collection was cleared
        """
        if not confirm:
            logger.warning("Collection clear requires confirmation")
            return False

        collection_name = collection_name or self.config.collection_name

        try:
            # Delete and recreate collection
            info = self.get_collection_info(collection_name)
            if not info:
                return False

            self.delete_collection(collection_name)
            self.create_collection(
                collection_name=collection_name,
                vector_size=info.get("vector_size"),
                distance_metric=info.get("distance"),
            )

            logger.warning(f"Collection cleared: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
