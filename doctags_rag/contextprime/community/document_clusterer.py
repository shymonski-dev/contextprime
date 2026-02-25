"""
Document Clusterer for clustering documents based on various criteria.

Supports:
- Entity-based clustering
- Semantic clustering
- Temporal clustering
- Hierarchical clustering
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger


class ClusteringMethod(Enum):
    """Document clustering methods."""
    ENTITY_BASED = "entity_based"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


@dataclass
class DocumentCluster:
    """Represents a document cluster."""
    cluster_id: int
    document_ids: Set[str]
    centroid: Optional[np.ndarray] = None
    representative_docs: List[str] = None
    shared_entities: Set[str] = None
    themes: List[str] = None
    temporal_range: Optional[Tuple[Any, Any]] = None
    metadata: Dict[str, Any] = None


@dataclass
class ClusteringResult:
    """Results from document clustering."""
    clusters: Dict[int, DocumentCluster]
    doc_to_cluster: Dict[str, int]
    method: str
    num_clusters: int
    silhouette_score: Optional[float] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class DocumentClusterer:
    """
    Clusters documents based on various criteria.

    Supports entity overlap, semantic similarity, temporal grouping,
    and hierarchical clustering methods.
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        max_clusters: int = 50,
        random_seed: int = 42
    ):
        """
        Initialize document clusterer.

        Args:
            min_cluster_size: Minimum documents per cluster
            max_clusters: Maximum number of clusters
            random_seed: Random seed for reproducibility
        """
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.random_seed = random_seed

    def cluster_by_entities(
        self,
        doc_entities: Dict[str, Set[str]],
        similarity_threshold: float = 0.3
    ) -> ClusteringResult:
        """
        Cluster documents based on shared entities.

        Args:
            doc_entities: Dictionary mapping doc_id to set of entity names
            similarity_threshold: Minimum Jaccard similarity to cluster together

        Returns:
            ClusteringResult
        """
        logger.info(f"Clustering {len(doc_entities)} documents by entity overlap")
        start_time = time.time()

        # Build similarity matrix
        doc_ids = list(doc_entities.keys())
        n_docs = len(doc_ids)

        similarity_matrix = np.zeros((n_docs, n_docs))

        for i, doc_i in enumerate(doc_ids):
            entities_i = doc_entities[doc_i]
            for j, doc_j in enumerate(doc_ids):
                if i <= j:
                    entities_j = doc_entities[doc_j]
                    # Jaccard similarity
                    intersection = len(entities_i & entities_j)
                    union = len(entities_i | entities_j)
                    sim = intersection / union if union > 0 else 0.0
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        # Perform clustering
        n_clusters = self._estimate_num_clusters(distance_matrix)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )

        labels = clustering.fit_predict(distance_matrix)

        # Build clusters
        clusters = defaultdict(set)
        doc_to_cluster = {}

        for idx, label in enumerate(labels):
            doc_id = doc_ids[idx]
            clusters[label].add(doc_id)
            doc_to_cluster[doc_id] = label

        # Compute shared entities for each cluster
        cluster_objects = {}
        for cluster_id, doc_set in clusters.items():
            # Find entities shared by documents in cluster
            if doc_set:
                shared_entities = set.intersection(*[doc_entities[doc_id] for doc_id in doc_set])
            else:
                shared_entities = set()

            cluster_objects[cluster_id] = DocumentCluster(
                cluster_id=cluster_id,
                document_ids=doc_set,
                shared_entities=shared_entities,
                representative_docs=list(doc_set)[:3]
            )

        execution_time = time.time() - start_time

        # Compute silhouette score
        silhouette = None
        if n_clusters > 1 and n_docs > n_clusters:
            try:
                silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score: {e}")

        return ClusteringResult(
            clusters=cluster_objects,
            doc_to_cluster=doc_to_cluster,
            method="entity_based",
            num_clusters=len(cluster_objects),
            silhouette_score=silhouette,
            execution_time=execution_time
        )

    def cluster_by_semantics(
        self,
        doc_embeddings: Dict[str, np.ndarray],
        n_clusters: Optional[int] = None
    ) -> ClusteringResult:
        """
        Cluster documents based on semantic embeddings.

        Args:
            doc_embeddings: Dictionary mapping doc_id to embedding vector
            n_clusters: Number of clusters (auto-estimated if None)

        Returns:
            ClusteringResult
        """
        logger.info(f"Clustering {len(doc_embeddings)} documents by semantic similarity")
        start_time = time.time()

        doc_ids = list(doc_embeddings.keys())
        embeddings = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])

        # Estimate number of clusters if not provided
        if n_clusters is None:
            n_clusters = min(
                self.max_clusters,
                max(2, int(np.sqrt(len(doc_ids))))
            )

        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            n_init=10
        )

        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # Build clusters
        clusters = defaultdict(set)
        doc_to_cluster = {}

        for idx, label in enumerate(labels):
            doc_id = doc_ids[idx]
            clusters[label].add(doc_id)
            doc_to_cluster[doc_id] = label

        # Build cluster objects
        cluster_objects = {}
        for cluster_id, doc_set in clusters.items():
            # Find representative documents (closest to centroid)
            centroid = centroids[cluster_id]
            doc_distances = []

            for doc_id in doc_set:
                embedding = doc_embeddings[doc_id]
                distance = np.linalg.norm(embedding - centroid)
                doc_distances.append((doc_id, distance))

            doc_distances.sort(key=lambda x: x[1])
            representative_docs = [doc_id for doc_id, _ in doc_distances[:3]]

            cluster_objects[cluster_id] = DocumentCluster(
                cluster_id=cluster_id,
                document_ids=doc_set,
                centroid=centroid,
                representative_docs=representative_docs
            )

        execution_time = time.time() - start_time

        # Compute silhouette score
        silhouette = None
        if n_clusters > 1 and len(doc_ids) > n_clusters:
            try:
                silhouette = silhouette_score(embeddings, labels)
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score: {e}")

        return ClusteringResult(
            clusters=cluster_objects,
            doc_to_cluster=doc_to_cluster,
            method="semantic",
            num_clusters=len(cluster_objects),
            silhouette_score=silhouette,
            execution_time=execution_time
        )

    def cluster_by_time(
        self,
        doc_timestamps: Dict[str, Any],
        time_window: Optional[float] = None
    ) -> ClusteringResult:
        """
        Cluster documents based on temporal proximity.

        Args:
            doc_timestamps: Dictionary mapping doc_id to timestamp
            time_window: Time window for clustering (auto-estimated if None)

        Returns:
            ClusteringResult
        """
        logger.info(f"Clustering {len(doc_timestamps)} documents by temporal proximity")
        start_time = time.time()

        # Sort documents by timestamp
        sorted_docs = sorted(doc_timestamps.items(), key=lambda x: x[1])

        if not sorted_docs:
            return ClusteringResult(
                clusters={},
                doc_to_cluster={},
                method="temporal",
                num_clusters=0,
                execution_time=time.time() - start_time
            )

        # Estimate time window if not provided
        if time_window is None:
            timestamps = [ts for _, ts in sorted_docs]
            time_range = max(timestamps) - min(timestamps)
            time_window = time_range / min(10, len(sorted_docs))

        # Cluster documents within time windows
        clusters = {}
        doc_to_cluster = {}
        current_cluster_id = 0
        current_cluster_docs = set()
        cluster_start_time = sorted_docs[0][1]

        for doc_id, timestamp in sorted_docs:
            if timestamp - cluster_start_time <= time_window:
                # Add to current cluster
                current_cluster_docs.add(doc_id)
            else:
                # Save current cluster and start new one
                if current_cluster_docs:
                    clusters[current_cluster_id] = DocumentCluster(
                        cluster_id=current_cluster_id,
                        document_ids=current_cluster_docs.copy(),
                        temporal_range=(cluster_start_time, timestamp)
                    )
                    for d in current_cluster_docs:
                        doc_to_cluster[d] = current_cluster_id

                current_cluster_id += 1
                current_cluster_docs = {doc_id}
                cluster_start_time = timestamp

        # Save last cluster
        if current_cluster_docs:
            clusters[current_cluster_id] = DocumentCluster(
                cluster_id=current_cluster_id,
                document_ids=current_cluster_docs,
                temporal_range=(cluster_start_time, sorted_docs[-1][1])
            )
            for d in current_cluster_docs:
                doc_to_cluster[d] = current_cluster_id

        execution_time = time.time() - start_time

        return ClusteringResult(
            clusters=clusters,
            doc_to_cluster=doc_to_cluster,
            method="temporal",
            num_clusters=len(clusters),
            execution_time=execution_time
        )

    def hierarchical_cluster(
        self,
        doc_embeddings: Dict[str, np.ndarray],
        linkage: str = 'ward',
        distance_threshold: Optional[float] = None
    ) -> ClusteringResult:
        """
        Perform hierarchical clustering on documents.

        Args:
            doc_embeddings: Dictionary mapping doc_id to embedding vector
            linkage: Linkage criterion (ward, complete, average, single)
            distance_threshold: Distance threshold for cutting the tree

        Returns:
            ClusteringResult
        """
        logger.info(f"Hierarchical clustering of {len(doc_embeddings)} documents")
        start_time = time.time()

        doc_ids = list(doc_embeddings.keys())
        embeddings = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])

        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage=linkage
        )

        labels = clustering.fit_predict(embeddings)

        # Build clusters
        clusters = defaultdict(set)
        doc_to_cluster = {}

        for idx, label in enumerate(labels):
            doc_id = doc_ids[idx]
            clusters[label].add(doc_id)
            doc_to_cluster[doc_id] = label

        # Build cluster objects
        cluster_objects = {}
        for cluster_id, doc_set in clusters.items():
            cluster_objects[cluster_id] = DocumentCluster(
                cluster_id=cluster_id,
                document_ids=doc_set,
                representative_docs=list(doc_set)[:3]
            )

        execution_time = time.time() - start_time

        # Compute silhouette score
        silhouette = None
        n_clusters = len(cluster_objects)
        if n_clusters > 1 and len(doc_ids) > n_clusters:
            try:
                silhouette = silhouette_score(embeddings, labels)
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score: {e}")

        return ClusteringResult(
            clusters=cluster_objects,
            doc_to_cluster=doc_to_cluster,
            method="hierarchical",
            num_clusters=n_clusters,
            silhouette_score=silhouette,
            execution_time=execution_time
        )

    def hybrid_cluster(
        self,
        doc_embeddings: Dict[str, np.ndarray],
        doc_entities: Dict[str, Set[str]],
        entity_weight: float = 0.5
    ) -> ClusteringResult:
        """
        Cluster documents using hybrid approach (semantic + entity-based).

        Args:
            doc_embeddings: Document embeddings
            doc_entities: Document entities
            entity_weight: Weight for entity similarity (0-1)

        Returns:
            ClusteringResult
        """
        logger.info(f"Hybrid clustering of {len(doc_embeddings)} documents")
        start_time = time.time()

        doc_ids = list(doc_embeddings.keys())
        n_docs = len(doc_ids)

        # Build combined similarity matrix
        similarity_matrix = np.zeros((n_docs, n_docs))

        embeddings_array = np.array([doc_embeddings[doc_id] for doc_id in doc_ids])

        for i in range(n_docs):
            for j in range(i, n_docs):
                doc_i = doc_ids[i]
                doc_j = doc_ids[j]

                # Semantic similarity (cosine)
                emb_i = embeddings_array[i]
                emb_j = embeddings_array[j]
                semantic_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-10)

                # Entity similarity (Jaccard)
                entities_i = doc_entities.get(doc_i, set())
                entities_j = doc_entities.get(doc_j, set())
                intersection = len(entities_i & entities_j)
                union = len(entities_i | entities_j)
                entity_sim = intersection / union if union > 0 else 0.0

                # Combined similarity
                combined_sim = (1 - entity_weight) * semantic_sim + entity_weight * entity_sim
                similarity_matrix[i, j] = combined_sim
                similarity_matrix[j, i] = combined_sim

        # Convert to distance
        distance_matrix = 1 - similarity_matrix

        # Clustering
        n_clusters = self._estimate_num_clusters(distance_matrix)

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )

        labels = clustering.fit_predict(distance_matrix)

        # Build clusters
        clusters = defaultdict(set)
        doc_to_cluster = {}

        for idx, label in enumerate(labels):
            doc_id = doc_ids[idx]
            clusters[label].add(doc_id)
            doc_to_cluster[doc_id] = label

        # Build cluster objects
        cluster_objects = {}
        for cluster_id, doc_set in clusters.items():
            # Compute shared entities
            if doc_set:
                shared_entities = set.intersection(*[doc_entities.get(doc_id, set()) for doc_id in doc_set])
            else:
                shared_entities = set()

            cluster_objects[cluster_id] = DocumentCluster(
                cluster_id=cluster_id,
                document_ids=doc_set,
                shared_entities=shared_entities,
                representative_docs=list(doc_set)[:3]
            )

        execution_time = time.time() - start_time

        # Compute silhouette score
        silhouette = None
        if n_clusters > 1 and n_docs > n_clusters:
            try:
                silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score: {e}")

        return ClusteringResult(
            clusters=cluster_objects,
            doc_to_cluster=doc_to_cluster,
            method="hybrid",
            num_clusters=len(cluster_objects),
            silhouette_score=silhouette,
            execution_time=execution_time
        )

    def _estimate_num_clusters(self, distance_matrix: np.ndarray) -> int:
        """Estimate optimal number of clusters using heuristics."""
        n_samples = distance_matrix.shape[0]

        # Use square root heuristic
        n_clusters = int(np.sqrt(n_samples))

        # Apply constraints
        n_clusters = max(2, min(self.max_clusters, n_clusters))

        return n_clusters
