"""
Entity Resolver for Knowledge Graph Construction.

Resolves and disambiguates entities across documents using:
- String similarity (Levenshtein, Jaro-Winkler, fuzzy matching)
- Embedding-based semantic similarity
- Hybrid approaches
- Entity merging strategies
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from loguru import logger

from .entity_extractor import Entity
from ..core.config import get_settings


@dataclass
class EntityCluster:
    """Represents a cluster of resolved entities."""
    canonical_entity: Entity
    variant_entities: List[Entity] = field(default_factory=list)
    confidence: float = 1.0
    merge_reason: str = "exact_match"

    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary."""
        return {
            "canonical": self.canonical_entity.to_dict(),
            "variants": [e.to_dict() for e in self.variant_entities],
            "total_occurrences": 1 + len(self.variant_entities),
            "confidence": self.confidence,
            "merge_reason": self.merge_reason
        }


@dataclass
class ResolutionResult:
    """Result of entity resolution."""
    clusters: List[EntityCluster]
    unique_entities: int
    merged_count: int
    statistics: Dict[str, Any] = field(default_factory=dict)


class EntityResolver:
    """
    Comprehensive entity resolution and disambiguation.

    Features:
    - String similarity matching (Levenshtein, Jaro-Winkler)
    - Fuzzy string matching
    - Embedding-based semantic similarity
    - Hybrid resolution combining multiple approaches
    - Entity merging with provenance tracking
    - Cross-document entity linking
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_embeddings: bool = True,
        algorithm: str = "hybrid"  # levenshtein, embedding, or hybrid
    ):
        """
        Initialize entity resolver.

        Args:
            similarity_threshold: Minimum similarity for entity matching
            embedding_model: Sentence transformer model for embeddings
            use_embeddings: Whether to use embedding-based similarity
            algorithm: Resolution algorithm to use
        """
        self.similarity_threshold = similarity_threshold
        self.algorithm = algorithm
        self.use_embeddings = use_embeddings
        self.settings = get_settings()

        # Initialize embedding model if needed
        self.embedding_model = None
        if use_embeddings and algorithm in ["embedding", "hybrid"]:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.use_embeddings = False

        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def resolve_entities(
        self,
        entities: List[Entity],
        entity_type: Optional[str] = None
    ) -> ResolutionResult:
        """
        Resolve entities within a single document or entity list.

        Args:
            entities: List of entities to resolve
            entity_type: Optional filter by entity type

        Returns:
            ResolutionResult with entity clusters
        """
        if not entities:
            return ResolutionResult(
                clusters=[],
                unique_entities=0,
                merged_count=0
            )

        # Filter by type if specified
        if entity_type:
            entities = [e for e in entities if e.type == entity_type]

        # Group entities by type for more accurate resolution
        entities_by_type = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.type].append(entity)

        all_clusters = []

        # Resolve each type separately
        for ent_type, ent_list in entities_by_type.items():
            if self.algorithm == "levenshtein":
                clusters = self._resolve_string_similarity(ent_list)
            elif self.algorithm == "embedding":
                clusters = self._resolve_embedding_similarity(ent_list)
            elif self.algorithm == "hybrid":
                clusters = self._resolve_hybrid(ent_list)
            else:
                clusters = self._resolve_string_similarity(ent_list)

            all_clusters.extend(clusters)

        # Generate statistics
        statistics = self._generate_statistics(all_clusters, len(entities))

        result = ResolutionResult(
            clusters=all_clusters,
            unique_entities=len(all_clusters),
            merged_count=len(entities) - len(all_clusters),
            statistics=statistics
        )

        logger.debug(
            f"Resolved {len(entities)} entities to {len(all_clusters)} unique entities "
            f"(merged {result.merged_count})"
        )

        return result

    def resolve_cross_document(
        self,
        entity_sets: List[Tuple[str, List[Entity]]]  # (doc_id, entities)
    ) -> ResolutionResult:
        """
        Resolve entities across multiple documents.

        Args:
            entity_sets: List of (document_id, entities) tuples

        Returns:
            ResolutionResult with cross-document entity clusters
        """
        # Flatten all entities while tracking provenance
        all_entities = []
        entity_to_doc = {}

        for doc_id, entities in entity_sets:
            for entity in entities:
                all_entities.append(entity)
                # Track which document this entity came from
                entity_key = id(entity)
                entity_to_doc[entity_key] = doc_id

        # Resolve all entities together
        result = self.resolve_entities(all_entities)

        # Add document provenance to clusters
        for cluster in result.clusters:
            docs = set()
            # Add document for canonical entity
            if id(cluster.canonical_entity) in entity_to_doc:
                docs.add(entity_to_doc[id(cluster.canonical_entity)])

            # Add documents for variants
            for variant in cluster.variant_entities:
                if id(variant) in entity_to_doc:
                    docs.add(entity_to_doc[id(variant)])

            cluster.canonical_entity.attributes["source_documents"] = list(docs)

        logger.info(
            f"Cross-document resolution: {len(all_entities)} entities "
            f"from {len(entity_sets)} documents resolved to {result.unique_entities} unique entities"
        )

        return result

    def _resolve_string_similarity(
        self,
        entities: List[Entity]
    ) -> List[EntityCluster]:
        """Resolve entities using string similarity."""
        if not entities:
            return []

        clusters = []
        processed = set()

        # Sort by confidence (process high-confidence entities first)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)

        for i, entity in enumerate(sorted_entities):
            if i in processed:
                continue

            # Create new cluster with this entity as canonical
            cluster = EntityCluster(
                canonical_entity=entity,
                variant_entities=[],
                confidence=entity.confidence,
                merge_reason="canonical"
            )

            # Find similar entities
            for j, other in enumerate(sorted_entities[i+1:], start=i+1):
                if j in processed:
                    continue

                similarity = self._calculate_string_similarity(
                    entity.text,
                    other.text
                )

                if similarity >= self.similarity_threshold:
                    cluster.variant_entities.append(other)
                    processed.add(j)

            processed.add(i)
            clusters.append(cluster)

        return clusters

    def _resolve_embedding_similarity(
        self,
        entities: List[Entity]
    ) -> List[EntityCluster]:
        """Resolve entities using embedding-based similarity."""
        if not entities or not self.embedding_model:
            return self._resolve_string_similarity(entities)

        # Get embeddings for all entities
        entity_texts = []
        for entity in entities:
            # Use entity text with context if available
            text = entity.context if entity.context else entity.text
            entity_texts.append(text)

        embeddings = self.embedding_model.encode(entity_texts, show_progress_bar=False)

        clusters = []
        processed = set()

        # Sort by confidence
        sorted_entities = sorted(enumerate(entities), key=lambda x: x[1].confidence, reverse=True)

        for idx, (i, entity) in enumerate(sorted_entities):
            if i in processed:
                continue

            cluster = EntityCluster(
                canonical_entity=entity,
                variant_entities=[],
                confidence=entity.confidence,
                merge_reason="embedding_similarity"
            )

            entity_emb = embeddings[i]

            # Find similar entities
            for other_idx, (j, other) in enumerate(sorted_entities[idx+1:], start=idx+1):
                if j in processed:
                    continue

                other_emb = embeddings[j]

                # Calculate cosine similarity
                similarity = self._cosine_similarity(entity_emb, other_emb)

                if similarity >= self.similarity_threshold:
                    cluster.variant_entities.append(other)
                    processed.add(j)

            processed.add(i)
            clusters.append(cluster)

        return clusters

    def _resolve_hybrid(
        self,
        entities: List[Entity]
    ) -> List[EntityCluster]:
        """Resolve entities using hybrid approach (string + embeddings)."""
        if not entities:
            return []

        # First pass: string similarity
        string_clusters = self._resolve_string_similarity(entities)

        # If embeddings not available, return string clusters
        if not self.embedding_model:
            return string_clusters

        # Second pass: check embedding similarity for remaining entities
        # Extract canonical entities from string clusters
        canonical_entities = [cluster.canonical_entity for cluster in string_clusters]

        if len(canonical_entities) <= 1:
            return string_clusters

        # Get embeddings for canonical entities
        entity_texts = [
            e.context if e.context else e.text
            for e in canonical_entities
        ]
        embeddings = self.embedding_model.encode(entity_texts, show_progress_bar=False)

        # Merge clusters with high embedding similarity
        final_clusters = []
        processed = set()

        for i, cluster in enumerate(string_clusters):
            if i in processed:
                continue

            # Check against other clusters
            merged = False
            for j, other_cluster in enumerate(string_clusters[i+1:], start=i+1):
                if j in processed:
                    continue

                # Calculate embedding similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])

                if similarity >= self.similarity_threshold:
                    # Merge clusters
                    cluster.variant_entities.append(other_cluster.canonical_entity)
                    cluster.variant_entities.extend(other_cluster.variant_entities)
                    cluster.merge_reason = "hybrid_similarity"
                    processed.add(j)
                    merged = True

            processed.add(i)
            final_clusters.append(cluster)

        return final_clusters

    def _calculate_string_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calculate string similarity using multiple metrics."""
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Exact match
        if text1 == text2:
            return 1.0

        # Calculate multiple similarity scores
        # 1. Ratio (Levenshtein-based)
        ratio_score = fuzz.ratio(text1, text2) / 100.0

        # 2. Token sort ratio (handles word order differences)
        token_sort_score = fuzz.token_sort_ratio(text1, text2) / 100.0

        # 3. Partial ratio (handles substring matches)
        partial_score = fuzz.partial_ratio(text1, text2) / 100.0

        # Use the maximum score
        similarity = max(ratio_score, token_sort_score, partial_score)

        return similarity

    def _cosine_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def merge_entities(
        self,
        entity1: Entity,
        entity2: Entity,
        keep_all_attributes: bool = True
    ) -> Entity:
        """
        Merge two entities, preserving information.

        Args:
            entity1: First entity (will be used as base)
            entity2: Second entity
            keep_all_attributes: Whether to merge all attributes

        Returns:
            Merged entity
        """
        # Use entity with higher confidence as base
        if entity2.confidence > entity1.confidence:
            entity1, entity2 = entity2, entity1

        merged = Entity(
            text=entity1.text,
            type=entity1.type,
            start_char=entity1.start_char,
            end_char=entity1.end_char,
            confidence=max(entity1.confidence, entity2.confidence),
            attributes=entity1.attributes.copy(),
            context=entity1.context or entity2.context,
            source=entity1.source
        )

        # Merge attributes
        if keep_all_attributes:
            for key, value in entity2.attributes.items():
                if key not in merged.attributes:
                    merged.attributes[key] = value
                elif isinstance(merged.attributes[key], list):
                    if isinstance(value, list):
                        merged.attributes[key].extend(value)
                    else:
                        merged.attributes[key].append(value)

        # Track merge history
        if "merged_from" not in merged.attributes:
            merged.attributes["merged_from"] = []
        merged.attributes["merged_from"].append({
            "text": entity2.text,
            "type": entity2.type,
            "confidence": entity2.confidence
        })

        return merged

    def find_canonical_form(
        self,
        entity_variants: List[str]
    ) -> str:
        """
        Find the canonical form for entity variants.

        Args:
            entity_variants: List of entity text variants

        Returns:
            Canonical form (most common or longest form)
        """
        if not entity_variants:
            return ""

        if len(entity_variants) == 1:
            return entity_variants[0]

        # Count occurrences
        from collections import Counter
        counts = Counter(entity_variants)

        # Get most common
        most_common = counts.most_common(1)[0][0]

        # If there's a tie, prefer the longest form
        max_count = counts[most_common]
        candidates = [text for text, count in counts.items() if count == max_count]

        if len(candidates) > 1:
            canonical = max(candidates, key=len)
        else:
            canonical = most_common

        return canonical

    def resolve_abbreviations(
        self,
        entities: List[Entity],
        abbreviation_map: Optional[Dict[str, str]] = None
    ) -> List[Entity]:
        """
        Resolve abbreviations to full forms.

        Args:
            entities: List of entities
            abbreviation_map: Optional mapping of abbreviations to full forms

        Returns:
            List of entities with resolved abbreviations
        """
        if abbreviation_map is None:
            abbreviation_map = {}

        # Auto-detect abbreviations
        full_forms = {}
        for entity in entities:
            if len(entity.text) > 3 and entity.text.isupper():
                # Likely an abbreviation
                # Look for full form in context
                if entity.context:
                    words = entity.context.split()
                    # Simple heuristic: look for capitalized phrase before/after
                    for i, word in enumerate(words):
                        if word == entity.text and i > 0:
                            # Check previous words
                            potential_full = []
                            for j in range(max(0, i-5), i):
                                if words[j][0].isupper():
                                    potential_full.append(words[j])
                            if potential_full:
                                full_forms[entity.text] = " ".join(potential_full)

        # Merge with provided map
        full_forms.update(abbreviation_map)

        # Resolve entities
        resolved = []
        for entity in entities:
            if entity.text in full_forms:
                resolved_entity = Entity(
                    text=full_forms[entity.text],
                    type=entity.type,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    confidence=entity.confidence,
                    attributes={**entity.attributes, "abbreviation": entity.text},
                    context=entity.context,
                    source=entity.source
                )
                resolved.append(resolved_entity)
            else:
                resolved.append(entity)

        return resolved

    def _generate_statistics(
        self,
        clusters: List[EntityCluster],
        original_count: int
    ) -> Dict[str, Any]:
        """Generate statistics about entity resolution."""
        stats = {
            "original_entities": original_count,
            "unique_entities": len(clusters),
            "merged_entities": original_count - len(clusters),
            "merge_ratio": (original_count - len(clusters)) / original_count if original_count > 0 else 0,
            "clusters_by_size": defaultdict(int),
            "merge_reasons": defaultdict(int)
        }

        for cluster in clusters:
            cluster_size = 1 + len(cluster.variant_entities)
            stats["clusters_by_size"][cluster_size] += 1
            stats["merge_reasons"][cluster.merge_reason] += 1

        stats["clusters_by_size"] = dict(stats["clusters_by_size"])
        stats["merge_reasons"] = dict(stats["merge_reasons"])

        return stats

    def get_entity_variants(
        self,
        canonical_text: str,
        clusters: List[EntityCluster]
    ) -> List[str]:
        """
        Get all variants of an entity.

        Args:
            canonical_text: Canonical entity text
            clusters: Entity clusters

        Returns:
            List of variant texts
        """
        for cluster in clusters:
            if cluster.canonical_entity.text.lower() == canonical_text.lower():
                variants = [cluster.canonical_entity.text]
                variants.extend([e.text for e in cluster.variant_entities])
                return list(set(variants))

        return []
