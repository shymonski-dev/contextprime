"""
Qdrant Vector Database Manager for DocTags RAG System.

Provides comprehensive vector database operations including:
- Connection management
- Collection creation and management
- Vector insertion and batch operations
- Similarity search with filtering
- Collection statistics and monitoring
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from uuid import UUID, uuid4
import time
from functools import wraps

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
        vector_id = vector_id or str(uuid4())

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
                    id=v.id or str(uuid4()),
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
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_obj,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
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

            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None

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
                vector_size=info["vector_dimension"],
            )

            logger.warning(f"Collection cleared: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
