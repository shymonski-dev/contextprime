"""
Intelligent Caching Layer for DocTags RAG.

Implements multi-level caching:
- Query result caching
- Embedding caching
- Semantic similarity based cache matching
- LRU eviction with TTL
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib
import json
import pickle
from pathlib import Path
import threading

import numpy as np
from loguru import logger

try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    logger.warning("diskcache not available, using in-memory cache only")


@dataclass
class CacheEntry:
    """Represents a cache entry."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None

            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            self.hits += 1
            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set value in cache."""
        with self.lock:
            now = datetime.now()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=0,
                ttl_seconds=ttl or self.default_ttl,
                metadata=metadata or {}
            )

            self.cache[key] = entry
            self.cache.move_to_end(key)

            # Evict if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate
            }


class SemanticQueryCache:
    """Cache with semantic similarity matching for queries."""

    def __init__(
        self,
        max_size: int = 500,
        similarity_threshold: float = 0.95,
        default_ttl: int = 3600
    ):
        """
        Initialize semantic query cache.

        Args:
            max_size: Maximum cache entries
            similarity_threshold: Minimum similarity for cache hit
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl

        self.cache: Dict[str, CacheEntry] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.lock = threading.RLock()

        self.hits = 0
        self.misses = 0
        self.semantic_hits = 0

    def get(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> Optional[Any]:
        """
        Get cached results for query.

        Args:
            query: Query text
            query_embedding: Optional query embedding for semantic matching

        Returns:
            Cached results or None
        """
        with self.lock:
            # Exact match
            query_hash = self._hash_query(query)
            if query_hash in self.cache:
                entry = self.cache[query_hash]
                if not entry.is_expired():
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    self.hits += 1
                    return entry.value
                else:
                    del self.cache[query_hash]

            # Semantic match if embedding provided
            if query_embedding is not None:
                similar_key = self._find_similar_query(query_embedding)
                if similar_key:
                    entry = self.cache[similar_key]
                    if not entry.is_expired():
                        entry.last_accessed = datetime.now()
                        entry.access_count += 1
                        self.semantic_hits += 1
                        self.hits += 1
                        logger.debug(f"Semantic cache hit for query: {query}")
                        return entry.value

            self.misses += 1
            return None

    def set(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache results for query.

        Args:
            query: Query text
            query_embedding: Query embedding for semantic matching
            value: Results to cache
            ttl: Time to live in seconds
        """
        with self.lock:
            query_hash = self._hash_query(query)

            entry = CacheEntry(
                key=query_hash,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                ttl_seconds=ttl or self.default_ttl,
                metadata={"query": query}
            )

            self.cache[query_hash] = entry

            if query_embedding is not None:
                self.query_embeddings[query_hash] = query_embedding

            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                self._evict_oldest()

    def _find_similar_query(
        self,
        query_embedding: np.ndarray
    ) -> Optional[str]:
        """Find similar cached query using embeddings."""
        if not self.query_embeddings:
            return None

        max_similarity = 0.0
        most_similar_key = None

        for key, cached_embedding in self.query_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity > max_similarity and similarity >= self.similarity_threshold:
                max_similarity = similarity
                most_similar_key = key

        return most_similar_key

    def _evict_oldest(self) -> None:
        """Evict oldest entry."""
        if not self.cache:
            return

        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )

        del self.cache[oldest_key]
        if oldest_key in self.query_embeddings:
            del self.query_embeddings[oldest_key]

    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        return hashlib.sha256(query.encode()).hexdigest()

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.query_embeddings.clear()
            self.hits = 0
            self.misses = 0
            self.semantic_hits = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            semantic_hit_rate = self.semantic_hits / self.hits if self.hits > 0 else 0.0

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "semantic_hits": self.semantic_hits,
                "hit_rate": hit_rate,
                "semantic_hit_rate": semantic_hit_rate
            }


class CacheManager:
    """
    Comprehensive cache manager for RAG system.

    Features:
    - Query result caching with semantic matching
    - Embedding caching
    - Persistent disk cache
    - Multi-level caching strategy
    - Cache statistics and monitoring
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_size: int = 1000,
        max_query_cache_size: int = 500,
        enable_disk_cache: bool = True,
        query_ttl: int = 3600,  # 1 hour
        embedding_ttl: int = 86400,  # 24 hours
        semantic_similarity_threshold: float = 0.95
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for persistent cache
            max_memory_size: Maximum memory cache entries
            max_query_cache_size: Maximum query cache entries
            enable_disk_cache: Enable persistent disk cache
            query_ttl: TTL for query results in seconds
            embedding_ttl: TTL for embeddings in seconds
            semantic_similarity_threshold: Threshold for semantic cache hits
        """
        self.cache_dir = cache_dir or Path.home() / ".doctags_cache"
        self.enable_disk_cache = enable_disk_cache and DISKCACHE_AVAILABLE

        # Initialize caches
        self.query_cache = SemanticQueryCache(
            max_size=max_query_cache_size,
            similarity_threshold=semantic_similarity_threshold,
            default_ttl=query_ttl
        )

        self.embedding_cache = LRUCache(
            max_size=max_memory_size,
            default_ttl=embedding_ttl
        )

        self.result_cache = LRUCache(
            max_size=max_memory_size,
            default_ttl=query_ttl
        )

        # Disk cache
        self.disk_cache = None
        if self.enable_disk_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.disk_cache = diskcache.Cache(str(self.cache_dir))
                logger.info(f"Disk cache initialized at {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to initialize disk cache: {e}")
                self.enable_disk_cache = False

        logger.info("Cache manager initialized")

    def get_query_results(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached query results."""
        return self.query_cache.get(query, query_embedding)

    def cache_query_results(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> None:
        """Cache query results."""
        self.query_cache.set(query, query_embedding, results, ttl)

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._hash_text(text)
        cached = self.embedding_cache.get(key)

        if cached is not None:
            return cached

        # Check disk cache
        if self.enable_disk_cache and self.disk_cache:
            try:
                disk_cached = self.disk_cache.get(f"emb_{key}")
                if disk_cached is not None:
                    # Restore to memory cache
                    self.embedding_cache.set(key, disk_cached)
                    return disk_cached
            except:
                pass

        return None

    def cache_embedding(
        self,
        text: str,
        embedding: np.ndarray,
        persist: bool = True
    ) -> None:
        """Cache embedding."""
        key = self._hash_text(text)

        # Memory cache
        self.embedding_cache.set(key, embedding)

        # Disk cache
        if persist and self.enable_disk_cache and self.disk_cache:
            try:
                self.disk_cache.set(f"emb_{key}", embedding)
            except:
                pass

    def get_result(self, key: str) -> Optional[Any]:
        """Get cached result by key."""
        return self.result_cache.get(key)

    def cache_result(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Cache arbitrary result."""
        self.result_cache.set(key, value, ttl)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.result_cache.clear()

        if self.enable_disk_cache and self.disk_cache:
            try:
                self.disk_cache.clear()
            except:
                pass

        logger.info("All caches cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "query_cache": self.query_cache.get_stats(),
            "embedding_cache": self.embedding_cache.get_stats(),
            "result_cache": self.result_cache.get_stats(),
            "disk_cache_enabled": self.enable_disk_cache
        }

        if self.enable_disk_cache and self.disk_cache:
            try:
                stats["disk_cache"] = {
                    "size": len(self.disk_cache),
                    "volume": self.disk_cache.volume()
                }
            except:
                pass

        return stats

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def preload_embeddings(
        self,
        texts: List[str],
        embeddings: List[np.ndarray]
    ) -> int:
        """
        Preload embeddings in batch.

        Args:
            texts: List of texts
            embeddings: Corresponding embeddings

        Returns:
            Number of embeddings cached
        """
        count = 0
        for text, embedding in zip(texts, embeddings):
            try:
                self.cache_embedding(text, embedding, persist=True)
                count += 1
            except:
                pass

        logger.info(f"Preloaded {count} embeddings")
        return count

    def warm_up(self, common_queries: List[str]) -> None:
        """
        Warm up cache with common queries.

        Args:
            common_queries: List of frequently used queries
        """
        logger.info(f"Warming up cache with {len(common_queries)} queries")
        # This would be used with actual retrieval results
        # For now, just log the intent
        pass

    def export_stats(self, output_path: Path) -> None:
        """Export cache statistics to file."""
        stats = self.get_statistics()

        try:
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Cache statistics exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
