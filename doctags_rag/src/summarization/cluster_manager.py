"""
Intelligent Clustering Manager for RAPTOR System.

Implements multiple clustering algorithms:
- UMAP + HDBSCAN for density-based clustering
- K-means for centroid-based clustering
- Semantic clustering based on embeddings
- Topic-based grouping
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available, will use PCA for dimensionality reduction")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN not available, will use K-means as fallback")

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


class ClusteringMethod(Enum):
    """Available clustering methods."""
    UMAP_HDBSCAN = "umap_hdbscan"
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    AUTO = "auto"


@dataclass
class Cluster:
    """Represents a cluster of items."""
    cluster_id: int
    item_indices: List[int]
    centroid: Optional[np.ndarray] = None
    coherence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Get cluster size."""
        return len(self.item_indices)

    def __repr__(self) -> str:
        return f"Cluster(id={self.cluster_id}, size={self.size}, coherence={self.coherence_score:.3f})"


@dataclass
class ClusteringResult:
    """Results from clustering operation."""
    clusters: List[Cluster]
    labels: np.ndarray
    method: ClusteringMethod
    num_clusters: int
    outliers: List[int] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    reduced_embeddings: Optional[np.ndarray] = None

    def get_cluster_by_id(self, cluster_id: int) -> Optional[Cluster]:
        """Get cluster by ID."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None


class ClusterManager:
    """
    Manages clustering operations for hierarchical tree construction.

    Supports multiple clustering algorithms with automatic method selection
    based on data characteristics.
    """

    def __init__(
        self,
        method: ClusteringMethod = ClusteringMethod.AUTO,
        min_cluster_size: int = 3,
        max_cluster_size: int = 50,
        similarity_threshold: float = 0.7,
        random_state: int = 42
    ):
        """
        Initialize cluster manager.

        Args:
            method: Clustering method to use
            min_cluster_size: Minimum cluster size
            max_cluster_size: Maximum cluster size (soft constraint)
            similarity_threshold: Threshold for semantic similarity
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.similarity_threshold = similarity_threshold
        self.random_state = random_state

        logger.info(
            f"ClusterManager initialized: method={method.value}, "
            f"min_size={min_cluster_size}, max_size={max_cluster_size}"
        )

    def cluster(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int] = None,
        item_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> ClusteringResult:
        """
        Cluster embeddings using configured method.

        Args:
            embeddings: Embedding vectors (N, D)
            num_clusters: Target number of clusters (if applicable)
            item_metadata: Optional metadata for each item

        Returns:
            ClusteringResult with clusters and metrics
        """
        if len(embeddings) == 0:
            logger.warning("Empty embeddings provided")
            return ClusteringResult(
                clusters=[],
                labels=np.array([]),
                method=self.method,
                num_clusters=0
            )

        # Normalize embeddings
        embeddings = self._normalize_embeddings(embeddings)

        # Select clustering method
        method = self._select_method(embeddings, num_clusters)

        logger.info(f"Clustering {len(embeddings)} items using {method.value}")

        # Perform clustering
        if method == ClusteringMethod.UMAP_HDBSCAN:
            result = self._cluster_umap_hdbscan(embeddings)
        elif method == ClusteringMethod.KMEANS:
            result = self._cluster_kmeans(embeddings, num_clusters)
        elif method == ClusteringMethod.HIERARCHICAL:
            result = self._cluster_hierarchical(embeddings, num_clusters)
        elif method == ClusteringMethod.SEMANTIC:
            result = self._cluster_semantic(embeddings, item_metadata)
        else:
            # Fallback to K-means
            result = self._cluster_kmeans(embeddings, num_clusters)

        # Compute cluster metrics
        result.metrics = self._compute_metrics(embeddings, result.labels)

        # Compute centroids
        self._compute_centroids(embeddings, result)

        # Compute coherence scores
        self._compute_coherence_scores(embeddings, result)

        logger.info(
            f"Clustering complete: {result.num_clusters} clusters, "
            f"{len(result.outliers)} outliers"
        )

        return result

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def _select_method(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int]
    ) -> ClusteringMethod:
        """
        Select clustering method based on data characteristics.

        Args:
            embeddings: Embedding vectors
            num_clusters: Target number of clusters

        Returns:
            Selected clustering method
        """
        if self.method != ClusteringMethod.AUTO:
            return self.method

        n_samples = len(embeddings)

        # Use UMAP+HDBSCAN for large datasets if available
        if UMAP_AVAILABLE and HDBSCAN_AVAILABLE and n_samples > 100:
            return ClusteringMethod.UMAP_HDBSCAN

        # Use K-means if number of clusters is specified
        if num_clusters is not None:
            return ClusteringMethod.KMEANS

        # Use hierarchical for small datasets
        if n_samples < 100:
            return ClusteringMethod.HIERARCHICAL

        # Default to K-means
        return ClusteringMethod.KMEANS

    def _cluster_umap_hdbscan(self, embeddings: np.ndarray) -> ClusteringResult:
        """
        Cluster using UMAP for dimensionality reduction and HDBSCAN.

        Args:
            embeddings: Embedding vectors

        Returns:
            ClusteringResult
        """
        if not UMAP_AVAILABLE or not HDBSCAN_AVAILABLE:
            logger.warning("UMAP or HDBSCAN not available, falling back to K-means")
            return self._cluster_kmeans(embeddings, None)

        # Reduce dimensionality with UMAP
        n_components = min(50, embeddings.shape[1], len(embeddings) - 1)

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(embeddings) - 1),
            metric='cosine',
            random_state=self.random_state
        )

        reduced = reducer.fit_transform(embeddings)

        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(reduced)

        # Identify outliers (label = -1)
        outliers = np.where(labels == -1)[0].tolist()

        # Create clusters
        unique_labels = set(labels) - {-1}
        clusters = []

        for label in sorted(unique_labels):
            cluster_indices = np.where(labels == label)[0].tolist()

            cluster = Cluster(
                cluster_id=len(clusters),
                item_indices=cluster_indices,
                metadata={
                    'original_label': int(label),
                    'hdbscan_probability': clusterer.probabilities_[cluster_indices].tolist()
                }
            )
            clusters.append(cluster)

        return ClusteringResult(
            clusters=clusters,
            labels=labels,
            method=ClusteringMethod.UMAP_HDBSCAN,
            num_clusters=len(clusters),
            outliers=outliers,
            reduced_embeddings=reduced
        )

    def _cluster_kmeans(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int]
    ) -> ClusteringResult:
        """
        Cluster using K-means algorithm.

        Args:
            embeddings: Embedding vectors
            num_clusters: Number of clusters

        Returns:
            ClusteringResult
        """
        # Determine number of clusters
        if num_clusters is None:
            num_clusters = self._estimate_num_clusters(embeddings)

        num_clusters = max(2, min(num_clusters, len(embeddings) // self.min_cluster_size))

        # Apply K-means
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=self.random_state,
            n_init=10
        )

        labels = kmeans.fit_predict(embeddings)

        # Create clusters
        clusters = []
        for label in range(num_clusters):
            cluster_indices = np.where(labels == label)[0].tolist()

            if len(cluster_indices) >= self.min_cluster_size:
                cluster = Cluster(
                    cluster_id=len(clusters),
                    item_indices=cluster_indices,
                    centroid=kmeans.cluster_centers_[label],
                    metadata={'original_label': int(label)}
                )
                clusters.append(cluster)
            else:
                # Mark as outliers
                labels[cluster_indices] = -1

        outliers = np.where(labels == -1)[0].tolist()

        return ClusteringResult(
            clusters=clusters,
            labels=labels,
            method=ClusteringMethod.KMEANS,
            num_clusters=len(clusters),
            outliers=outliers
        )

    def _cluster_hierarchical(
        self,
        embeddings: np.ndarray,
        num_clusters: Optional[int]
    ) -> ClusteringResult:
        """
        Cluster using hierarchical/agglomerative clustering.

        Args:
            embeddings: Embedding vectors
            num_clusters: Number of clusters

        Returns:
            ClusteringResult
        """
        # Determine number of clusters
        if num_clusters is None:
            num_clusters = self._estimate_num_clusters(embeddings)

        num_clusters = max(2, min(num_clusters, len(embeddings) // self.min_cluster_size))

        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage='ward'
        )

        labels = clustering.fit_predict(embeddings)

        # Create clusters
        clusters = []
        for label in range(num_clusters):
            cluster_indices = np.where(labels == label)[0].tolist()

            if len(cluster_indices) >= self.min_cluster_size:
                cluster = Cluster(
                    cluster_id=len(clusters),
                    item_indices=cluster_indices,
                    metadata={'original_label': int(label)}
                )
                clusters.append(cluster)

        return ClusteringResult(
            clusters=clusters,
            labels=labels,
            method=ClusteringMethod.HIERARCHICAL,
            num_clusters=len(clusters)
        )

    def _cluster_semantic(
        self,
        embeddings: np.ndarray,
        item_metadata: Optional[List[Dict[str, Any]]]
    ) -> ClusteringResult:
        """
        Cluster using semantic similarity thresholds.

        Args:
            embeddings: Embedding vectors
            item_metadata: Item metadata

        Returns:
            ClusteringResult
        """
        # Compute pairwise similarities
        similarities = np.dot(embeddings, embeddings.T)

        # Initialize clusters
        labels = -np.ones(len(embeddings), dtype=int)
        cluster_id = 0

        for i in range(len(embeddings)):
            if labels[i] != -1:
                continue  # Already assigned

            # Find similar items
            similar_indices = np.where(similarities[i] >= self.similarity_threshold)[0]

            # Create new cluster
            labels[similar_indices] = cluster_id
            cluster_id += 1

        # Create clusters
        clusters = []
        for cid in range(cluster_id):
            cluster_indices = np.where(labels == cid)[0].tolist()

            if len(cluster_indices) >= self.min_cluster_size:
                cluster = Cluster(
                    cluster_id=len(clusters),
                    item_indices=cluster_indices,
                    metadata={'original_label': int(cid)}
                )
                clusters.append(cluster)
            else:
                # Mark as outliers
                labels[cluster_indices] = -1

        outliers = np.where(labels == -1)[0].tolist()

        return ClusteringResult(
            clusters=clusters,
            labels=labels,
            method=ClusteringMethod.SEMANTIC,
            num_clusters=len(clusters),
            outliers=outliers
        )

    def _estimate_num_clusters(self, embeddings: np.ndarray) -> int:
        """
        Estimate optimal number of clusters using heuristics.

        Args:
            embeddings: Embedding vectors

        Returns:
            Estimated number of clusters
        """
        n_samples = len(embeddings)

        # Use sqrt(n/2) heuristic
        estimated = int(np.sqrt(n_samples / 2))

        # Ensure reasonable bounds
        min_clusters = max(2, n_samples // self.max_cluster_size)
        max_clusters = min(n_samples // self.min_cluster_size, 20)

        return max(min_clusters, min(estimated, max_clusters))

    def _compute_centroids(
        self,
        embeddings: np.ndarray,
        result: ClusteringResult
    ):
        """
        Compute centroids for each cluster.

        Args:
            embeddings: Embedding vectors
            result: ClusteringResult to update
        """
        for cluster in result.clusters:
            cluster_embeddings = embeddings[cluster.item_indices]
            cluster.centroid = np.mean(cluster_embeddings, axis=0)

    def _compute_coherence_scores(
        self,
        embeddings: np.ndarray,
        result: ClusteringResult
    ):
        """
        Compute coherence scores for each cluster.

        Args:
            embeddings: Embedding vectors
            result: ClusteringResult to update
        """
        for cluster in result.clusters:
            if len(cluster.item_indices) < 2:
                cluster.coherence_score = 1.0
                continue

            cluster_embeddings = embeddings[cluster.item_indices]

            # Compute pairwise similarities within cluster
            similarities = np.dot(cluster_embeddings, cluster_embeddings.T)

            # Average similarity (excluding diagonal)
            mask = ~np.eye(len(similarities), dtype=bool)
            avg_similarity = similarities[mask].mean()

            cluster.coherence_score = float(avg_similarity)

    def _compute_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.

        Args:
            embeddings: Embedding vectors
            labels: Cluster labels

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Filter out outliers for metrics
        valid_mask = labels >= 0
        if valid_mask.sum() < 2:
            return metrics

        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]

        # Check if we have at least 2 clusters
        n_clusters = len(set(valid_labels))
        if n_clusters < 2:
            return metrics

        try:
            # Silhouette score
            metrics['silhouette_score'] = silhouette_score(
                valid_embeddings,
                valid_labels,
                metric='cosine'
            )
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")

        try:
            # Calinski-Harabasz score
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                valid_embeddings,
                valid_labels
            )
        except Exception as e:
            logger.warning(f"Could not compute Calinski-Harabasz score: {e}")

        return metrics

    def merge_small_clusters(
        self,
        result: ClusteringResult,
        embeddings: np.ndarray,
        min_size: Optional[int] = None
    ) -> ClusteringResult:
        """
        Merge clusters that are too small.

        Args:
            result: ClusteringResult
            embeddings: Embedding vectors
            min_size: Minimum cluster size (uses self.min_cluster_size if None)

        Returns:
            Updated ClusteringResult
        """
        if min_size is None:
            min_size = self.min_cluster_size

        # Identify small clusters
        small_clusters = [c for c in result.clusters if c.size < min_size]
        large_clusters = [c for c in result.clusters if c.size >= min_size]

        if not small_clusters:
            return result

        logger.info(f"Merging {len(small_clusters)} small clusters")

        # For each small cluster, merge with most similar large cluster
        for small in small_clusters:
            if not large_clusters:
                # All clusters are small, keep as is
                result.outliers.extend(small.item_indices)
                continue

            small_centroid = embeddings[small.item_indices].mean(axis=0)

            # Find most similar large cluster
            best_similarity = -1
            best_cluster = None

            for large in large_clusters:
                similarity = np.dot(small_centroid, large.centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = large

            # Merge
            if best_cluster:
                best_cluster.item_indices.extend(small.item_indices)
                # Recompute centroid
                best_cluster.centroid = embeddings[best_cluster.item_indices].mean(axis=0)

        # Update result
        result.clusters = large_clusters
        result.num_clusters = len(large_clusters)

        # Recompute labels
        new_labels = -np.ones(len(embeddings), dtype=int)
        for i, cluster in enumerate(result.clusters):
            new_labels[cluster.item_indices] = i
            cluster.cluster_id = i

        result.labels = new_labels

        return result
