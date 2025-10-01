"""
Cross-Document Analyzer for analyzing relationships and patterns across documents.

Provides:
- Entity co-occurrence analysis
- Theme evolution tracking
- Document similarity computation
- Knowledge synthesis across documents
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

import numpy as np
import networkx as nx
from loguru import logger


@dataclass
class CooccurrencePattern:
    """Entity co-occurrence pattern."""
    entity_pair: Tuple[str, str]
    cooccurrence_count: int
    documents: Set[str]
    confidence: float
    contexts: List[str] = None


@dataclass
class ThemeEvolution:
    """Theme evolution over documents."""
    theme: str
    timeline: List[Tuple[Any, float]]  # (timestamp, strength)
    documents: Set[str]
    trend: str  # "increasing", "decreasing", "stable"


@dataclass
class DocumentSimilarity:
    """Similarity between two documents."""
    doc_id_1: str
    doc_id_2: str
    semantic_similarity: float
    entity_overlap: float
    structural_similarity: float
    combined_similarity: float


class CrossDocumentAnalyzer:
    """
    Analyzes relationships and patterns across multiple documents.

    Supports co-occurrence analysis, theme tracking, similarity computation,
    and knowledge synthesis.
    """

    def __init__(
        self,
        min_cooccurrence: int = 2,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize cross-document analyzer.

        Args:
            min_cooccurrence: Minimum co-occurrences to report
            similarity_threshold: Minimum similarity for document pairs
        """
        self.min_cooccurrence = min_cooccurrence
        self.similarity_threshold = similarity_threshold

    def analyze_entity_cooccurrence(
        self,
        doc_entities: Dict[str, Set[str]],
        top_k: int = 50
    ) -> List[CooccurrencePattern]:
        """
        Analyze entity co-occurrence patterns across documents.

        Args:
            doc_entities: Dictionary mapping doc_id to set of entities
            top_k: Number of top patterns to return

        Returns:
            List of co-occurrence patterns
        """
        logger.info(f"Analyzing entity co-occurrence across {len(doc_entities)} documents")
        start_time = time.time()

        # Build co-occurrence matrix
        cooccurrence = defaultdict(lambda: {"count": 0, "docs": set()})

        for doc_id, entities in doc_entities.items():
            entity_list = sorted(list(entities))
            # Check all pairs
            for i, entity1 in enumerate(entity_list):
                for entity2 in entity_list[i+1:]:
                    pair = tuple(sorted([entity1, entity2]))
                    cooccurrence[pair]["count"] += 1
                    cooccurrence[pair]["docs"].add(doc_id)

        # Filter and create patterns
        patterns = []
        for pair, data in cooccurrence.items():
            if data["count"] >= self.min_cooccurrence:
                # Compute confidence based on frequency
                confidence = min(1.0, data["count"] / (len(doc_entities) * 0.1))

                pattern = CooccurrencePattern(
                    entity_pair=pair,
                    cooccurrence_count=data["count"],
                    documents=data["docs"],
                    confidence=confidence
                )
                patterns.append(pattern)

        # Sort by count and take top k
        patterns.sort(key=lambda x: x.cooccurrence_count, reverse=True)
        top_patterns = patterns[:top_k]

        logger.info(f"Found {len(top_patterns)} co-occurrence patterns in {time.time() - start_time:.2f}s")
        return top_patterns

    def build_cooccurrence_graph(
        self,
        doc_entities: Dict[str, Set[str]],
        min_cooccurrence: Optional[int] = None
    ) -> nx.Graph:
        """
        Build a graph of entity co-occurrences.

        Args:
            doc_entities: Dictionary mapping doc_id to set of entities
            min_cooccurrence: Minimum co-occurrences for edge (uses default if None)

        Returns:
            NetworkX graph with entities as nodes
        """
        min_co = min_cooccurrence or self.min_cooccurrence

        graph = nx.Graph()

        # Build co-occurrence counts
        cooccurrence = defaultdict(int)

        for doc_id, entities in doc_entities.items():
            entity_list = sorted(list(entities))
            for i, entity1 in enumerate(entity_list):
                # Add node if not exists
                if not graph.has_node(entity1):
                    graph.add_node(entity1, entity=entity1)

                for entity2 in entity_list[i+1:]:
                    if not graph.has_node(entity2):
                        graph.add_node(entity2, entity=entity2)

                    pair = tuple(sorted([entity1, entity2]))
                    cooccurrence[pair] += 1

        # Add edges for significant co-occurrences
        for (entity1, entity2), count in cooccurrence.items():
            if count >= min_co:
                graph.add_edge(entity1, entity2, weight=count, cooccurrence=count)

        logger.info(
            f"Built co-occurrence graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        return graph

    def track_theme_evolution(
        self,
        doc_themes: Dict[str, Dict[str, float]],
        doc_timestamps: Dict[str, Any]
    ) -> List[ThemeEvolution]:
        """
        Track how themes evolve across documents over time.

        Args:
            doc_themes: Dictionary mapping doc_id to theme scores
            doc_timestamps: Dictionary mapping doc_id to timestamps

        Returns:
            List of theme evolution patterns
        """
        logger.info("Tracking theme evolution")

        # Organize themes by timeline
        theme_timelines = defaultdict(list)
        theme_docs = defaultdict(set)

        for doc_id, themes in doc_themes.items():
            timestamp = doc_timestamps.get(doc_id)
            if timestamp is None:
                continue

            for theme, score in themes.items():
                theme_timelines[theme].append((timestamp, score))
                theme_docs[theme].add(doc_id)

        # Analyze each theme
        evolutions = []

        for theme, timeline in theme_timelines.items():
            # Sort by timestamp
            timeline.sort(key=lambda x: x[0])

            # Determine trend
            if len(timeline) >= 3:
                scores = [score for _, score in timeline]
                # Simple linear trend
                x = np.arange(len(scores))
                slope = np.polyfit(x, scores, 1)[0]

                if slope > 0.1:
                    trend = "increasing"
                elif slope < -0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"

            evolution = ThemeEvolution(
                theme=theme,
                timeline=timeline,
                documents=theme_docs[theme],
                trend=trend
            )
            evolutions.append(evolution)

        # Sort by number of documents
        evolutions.sort(key=lambda x: len(x.documents), reverse=True)

        return evolutions

    def compute_document_similarity(
        self,
        doc_id_1: str,
        doc_id_2: str,
        doc_embeddings: Dict[str, np.ndarray],
        doc_entities: Dict[str, Set[str]],
        doc_structure: Optional[Dict[str, Any]] = None
    ) -> DocumentSimilarity:
        """
        Compute multi-faceted similarity between two documents.

        Args:
            doc_id_1: First document ID
            doc_id_2: Second document ID
            doc_embeddings: Document embeddings
            doc_entities: Document entities
            doc_structure: Document structure information (optional)

        Returns:
            DocumentSimilarity object
        """
        # Semantic similarity from embeddings
        if doc_id_1 in doc_embeddings and doc_id_2 in doc_embeddings:
            emb1 = doc_embeddings[doc_id_1]
            emb2 = doc_embeddings[doc_id_2]
            semantic_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
        else:
            semantic_sim = 0.0

        # Entity overlap (Jaccard similarity)
        entities1 = doc_entities.get(doc_id_1, set())
        entities2 = doc_entities.get(doc_id_2, set())
        intersection = len(entities1 & entities2)
        union = len(entities1 | entities2)
        entity_overlap = intersection / union if union > 0 else 0.0

        # Structural similarity (if available)
        structural_sim = 0.0
        if doc_structure:
            struct1 = doc_structure.get(doc_id_1, {})
            struct2 = doc_structure.get(doc_id_2, {})
            # Compare structure features (e.g., section counts, depth)
            if struct1 and struct2:
                # Simple comparison of section counts
                sections1 = struct1.get("num_sections", 0)
                sections2 = struct2.get("num_sections", 0)
                max_sections = max(sections1, sections2, 1)
                structural_sim = 1.0 - abs(sections1 - sections2) / max_sections

        # Combined similarity (weighted average)
        weights = [0.5, 0.4, 0.1]  # semantic, entity, structural
        combined = (
            weights[0] * semantic_sim +
            weights[1] * entity_overlap +
            weights[2] * structural_sim
        )

        return DocumentSimilarity(
            doc_id_1=doc_id_1,
            doc_id_2=doc_id_2,
            semantic_similarity=semantic_sim,
            entity_overlap=entity_overlap,
            structural_similarity=structural_sim,
            combined_similarity=combined
        )

    def find_similar_documents(
        self,
        doc_id: str,
        doc_embeddings: Dict[str, np.ndarray],
        doc_entities: Dict[str, Set[str]],
        top_k: int = 10
    ) -> List[DocumentSimilarity]:
        """
        Find documents similar to the given document.

        Args:
            doc_id: Target document ID
            doc_embeddings: Document embeddings
            doc_entities: Document entities
            top_k: Number of similar documents to return

        Returns:
            List of DocumentSimilarity objects
        """
        similarities = []

        for other_doc_id in doc_embeddings.keys():
            if other_doc_id == doc_id:
                continue

            sim = self.compute_document_similarity(
                doc_id,
                other_doc_id,
                doc_embeddings,
                doc_entities
            )

            if sim.combined_similarity >= self.similarity_threshold:
                similarities.append(sim)

        # Sort by combined similarity
        similarities.sort(key=lambda x: x.combined_similarity, reverse=True)

        return similarities[:top_k]

    def detect_contradictions(
        self,
        doc_claims: Dict[str, List[str]],
        claim_embeddings: Dict[str, np.ndarray],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Detect potential contradictions across documents.

        Args:
            doc_claims: Dictionary mapping doc_id to list of claims
            claim_embeddings: Embeddings for each claim
            threshold: Similarity threshold for contradiction detection

        Returns:
            List of potential contradictions
        """
        logger.info("Detecting contradictions across documents")

        contradictions = []

        all_claims = []
        for doc_id, claims in doc_claims.items():
            for claim in claims:
                all_claims.append({"doc_id": doc_id, "claim": claim})

        # Compare all claim pairs
        for i, claim1_data in enumerate(all_claims):
            claim1 = claim1_data["claim"]
            doc1 = claim1_data["doc_id"]

            if claim1 not in claim_embeddings:
                continue

            emb1 = claim_embeddings[claim1]

            for claim2_data in all_claims[i+1:]:
                claim2 = claim2_data["claim"]
                doc2 = claim2_data["doc_id"]

                if doc1 == doc2:  # Skip same document
                    continue

                if claim2 not in claim_embeddings:
                    continue

                emb2 = claim_embeddings[claim2]

                # High similarity but from different documents suggests potential contradiction
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)

                if similarity >= threshold:
                    contradictions.append({
                        "claim1": claim1,
                        "claim2": claim2,
                        "doc1": doc1,
                        "doc2": doc2,
                        "similarity": similarity,
                        "type": "potential_contradiction"
                    })

        logger.info(f"Found {len(contradictions)} potential contradictions")
        return contradictions

    def identify_consensus(
        self,
        doc_claims: Dict[str, List[str]],
        min_support: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Identify claims with consensus across multiple documents.

        Args:
            doc_claims: Dictionary mapping doc_id to list of claims
            min_support: Minimum documents supporting a claim

        Returns:
            List of consensus claims
        """
        logger.info("Identifying consensus across documents")

        # Group similar claims
        claim_groups = defaultdict(lambda: {"claims": [], "docs": set()})

        # Simple grouping by exact match (could be improved with embedding similarity)
        for doc_id, claims in doc_claims.items():
            for claim in claims:
                normalized_claim = claim.lower().strip()
                claim_groups[normalized_claim]["claims"].append(claim)
                claim_groups[normalized_claim]["docs"].add(doc_id)

        # Filter by minimum support
        consensus_claims = []

        for normalized_claim, data in claim_groups.items():
            if len(data["docs"]) >= min_support:
                consensus_claims.append({
                    "claim": data["claims"][0],  # Use first occurrence
                    "support_count": len(data["docs"]),
                    "supporting_docs": list(data["docs"]),
                    "confidence": len(data["docs"]) / len(doc_claims)
                })

        # Sort by support count
        consensus_claims.sort(key=lambda x: x["support_count"], reverse=True)

        logger.info(f"Found {len(consensus_claims)} consensus claims")
        return consensus_claims

    def aggregate_knowledge(
        self,
        doc_knowledge: Dict[str, Dict[str, Any]],
        aggregation_method: str = "union"
    ) -> Dict[str, Any]:
        """
        Aggregate knowledge from multiple documents.

        Args:
            doc_knowledge: Dictionary mapping doc_id to knowledge structures
            aggregation_method: Method to aggregate (union, intersection, voting)

        Returns:
            Aggregated knowledge structure
        """
        logger.info(f"Aggregating knowledge from {len(doc_knowledge)} documents")

        if aggregation_method == "union":
            # Combine all unique facts
            aggregated = {
                "entities": set(),
                "relationships": [],
                "facts": set()
            }

            for doc_id, knowledge in doc_knowledge.items():
                aggregated["entities"].update(knowledge.get("entities", []))
                aggregated["relationships"].extend(knowledge.get("relationships", []))
                aggregated["facts"].update(knowledge.get("facts", []))

            # Convert sets to lists for JSON serialization
            aggregated["entities"] = list(aggregated["entities"])
            aggregated["facts"] = list(aggregated["facts"])

        elif aggregation_method == "intersection":
            # Only include facts present in all documents
            if not doc_knowledge:
                return {"entities": [], "relationships": [], "facts": []}

            all_entities = [set(k.get("entities", [])) for k in doc_knowledge.values()]
            all_facts = [set(k.get("facts", [])) for k in doc_knowledge.values()]

            aggregated = {
                "entities": list(set.intersection(*all_entities)) if all_entities else [],
                "facts": list(set.intersection(*all_facts)) if all_facts else [],
                "relationships": []
            }

        else:  # voting
            # Include facts with majority support
            entity_votes = defaultdict(int)
            fact_votes = defaultdict(int)

            for doc_id, knowledge in doc_knowledge.items():
                for entity in knowledge.get("entities", []):
                    entity_votes[entity] += 1
                for fact in knowledge.get("facts", []):
                    fact_votes[tuple(fact) if isinstance(fact, list) else fact] += 1

            threshold = len(doc_knowledge) / 2

            aggregated = {
                "entities": [e for e, v in entity_votes.items() if v > threshold],
                "facts": [f for f, v in fact_votes.items() if v > threshold],
                "relationships": []
            }

        return aggregated
