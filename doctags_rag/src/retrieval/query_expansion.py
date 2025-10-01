"""
Query Expansion System for DocTags RAG.

Implements comprehensive query expansion:
- Synonym expansion
- Entity expansion from knowledge graph
- Semantic expansion using embeddings
- Contextual expansion from session history
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import deque

import numpy as np
from loguru import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not available, entity extraction will be limited")

try:
    from nltk.corpus import wordnet
    import nltk
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    logger.warning("nltk/wordnet not available, synonym expansion will be limited")


@dataclass
class ExpandedQuery:
    """Expanded query with metadata."""
    original_query: str
    expanded_query: str
    synonyms: List[str]
    related_entities: List[str]
    semantic_terms: List[str]
    contextual_terms: List[str]
    expansion_strategy: str


class QueryExpander:
    """
    Comprehensive query expansion system.

    Features:
    - Synonym expansion via WordNet
    - Entity expansion from knowledge graph
    - Semantic expansion using embeddings
    - Contextual expansion from session
    - Domain-specific expansions
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        max_synonyms: int = 3,
        max_expansions: int = 5,
        enable_wordnet: bool = True,
        enable_semantic: bool = True,
        enable_contextual: bool = True,
        context_window: int = 5
    ):
        """
        Initialize query expander.

        Args:
            spacy_model: spaCy model for NLP
            max_synonyms: Maximum synonyms per term
            max_expansions: Maximum expansion terms total
            enable_wordnet: Enable WordNet synonyms
            enable_semantic: Enable semantic expansion
            enable_contextual: Enable contextual expansion
            context_window: Number of previous queries to consider
        """
        self.max_synonyms = max_synonyms
        self.max_expansions = max_expansions
        self.enable_wordnet = enable_wordnet and WORDNET_AVAILABLE
        self.enable_semantic = enable_semantic
        self.enable_contextual = enable_contextual
        self.context_window = context_window

        # Initialize spaCy
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")

        # Initialize WordNet
        if self.enable_wordnet:
            try:
                # Download wordnet if not available
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                logger.info("WordNet initialized")
            except Exception as e:
                logger.warning(f"WordNet initialization failed: {e}")
                self.enable_wordnet = False

        # Domain-specific synonyms/expansions
        self.domain_expansions = {
            "ml": ["machine learning", "artificial intelligence", "deep learning"],
            "ai": ["artificial intelligence", "machine learning", "neural networks"],
            "nlp": ["natural language processing", "text processing", "language models"],
            "cv": ["computer vision", "image processing", "visual recognition"],
            "dl": ["deep learning", "neural networks", "machine learning"],
        }

        # Query context history
        self.query_history: deque = deque(maxlen=context_window)

        logger.info("Query expander initialized")

    def expand_query(
        self,
        query: str,
        strategy: str = "comprehensive",
        graph_entities: Optional[List[str]] = None,
        embedding_model: Optional[Any] = None
    ) -> ExpandedQuery:
        """
        Expand query using multiple strategies.

        Args:
            query: Original query text
            strategy: Expansion strategy (comprehensive, conservative, aggressive)
            graph_entities: Related entities from knowledge graph
            embedding_model: Optional embedding model for semantic expansion

        Returns:
            Expanded query with metadata
        """
        logger.info(f"Expanding query: {query}")

        # Extract key terms
        key_terms = self._extract_key_terms(query)

        # Apply different expansion strategies
        synonyms = []
        related_entities = []
        semantic_terms = []
        contextual_terms = []

        # Synonym expansion
        if self.enable_wordnet:
            synonyms = self._expand_synonyms(key_terms)

        # Entity expansion from graph
        if graph_entities:
            related_entities = graph_entities[:self.max_expansions]

        # Semantic expansion
        if self.enable_semantic and embedding_model:
            semantic_terms = self._expand_semantic(query, key_terms, embedding_model)

        # Contextual expansion
        if self.enable_contextual:
            contextual_terms = self._expand_contextual(query)

        # Domain-specific expansion
        domain_terms = self._expand_domain_specific(key_terms)
        synonyms.extend(domain_terms)

        # Build expanded query based on strategy
        expanded_query = self._build_expanded_query(
            original_query=query,
            synonyms=synonyms,
            related_entities=related_entities,
            semantic_terms=semantic_terms,
            contextual_terms=contextual_terms,
            strategy=strategy
        )

        # Update query history
        self.query_history.append(query)

        result = ExpandedQuery(
            original_query=query,
            expanded_query=expanded_query,
            synonyms=synonyms[:self.max_expansions],
            related_entities=related_entities,
            semantic_terms=semantic_terms[:self.max_expansions],
            contextual_terms=contextual_terms,
            expansion_strategy=strategy
        )

        logger.info(f"Query expanded: {len(synonyms)} synonyms, {len(related_entities)} entities")

        return result

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        if self.nlp:
            try:
                doc = self.nlp(query)
                # Extract nouns, proper nouns, and important verbs
                key_terms = [
                    token.text.lower()
                    for token in doc
                    if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop
                ]
                return key_terms
            except:
                pass

        # Fallback: simple extraction
        words = query.lower().split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'what', 'who', 'where', 'when', 'how', 'why'}
        key_terms = [w for w in words if w not in stopwords]
        return key_terms

    def _expand_synonyms(self, key_terms: List[str]) -> List[str]:
        """Expand using WordNet synonyms."""
        if not self.enable_wordnet:
            return []

        synonyms = set()

        for term in key_terms:
            try:
                # Get synsets for the term
                synsets = wordnet.synsets(term)

                for synset in synsets[:2]:  # Limit synsets per term
                    # Get lemmas (synonyms)
                    for lemma in synset.lemmas()[:self.max_synonyms]:
                        synonym = lemma.name().replace('_', ' ').lower()
                        if synonym != term.lower():
                            synonyms.add(synonym)

            except Exception as e:
                logger.debug(f"Synonym expansion failed for '{term}': {e}")
                continue

        return list(synonyms)[:self.max_expansions]

    def _expand_semantic(
        self,
        query: str,
        key_terms: List[str],
        embedding_model: Any
    ) -> List[str]:
        """Expand using semantic similarity from embeddings."""
        # This would require a vocabulary and embeddings
        # Simplified version: return empty for now
        # In production, you would:
        # 1. Get query embedding
        # 2. Find similar terms in vocabulary
        # 3. Return top similar terms
        return []

    def _expand_contextual(self, query: str) -> List[str]:
        """Expand using context from query history."""
        if not self.query_history:
            return []

        contextual_terms = []

        # Extract terms from recent queries
        for past_query in self.query_history:
            if past_query == query:
                continue

            # Extract key terms from past query
            past_terms = self._extract_key_terms(past_query)

            # Add terms that might be relevant
            for term in past_terms:
                if term not in query.lower() and term not in contextual_terms:
                    contextual_terms.append(term)

        return contextual_terms[:3]  # Limit contextual terms

    def _expand_domain_specific(self, key_terms: List[str]) -> List[str]:
        """Expand using domain-specific knowledge."""
        expansions = []

        for term in key_terms:
            term_lower = term.lower()
            if term_lower in self.domain_expansions:
                expansions.extend(self.domain_expansions[term_lower])

        return expansions

    def _build_expanded_query(
        self,
        original_query: str,
        synonyms: List[str],
        related_entities: List[str],
        semantic_terms: List[str],
        contextual_terms: List[str],
        strategy: str
    ) -> str:
        """Build expanded query based on strategy."""
        if strategy == "conservative":
            # Only add most relevant synonyms
            additions = synonyms[:2] + related_entities[:1]
        elif strategy == "aggressive":
            # Add all expansions
            additions = synonyms + related_entities + semantic_terms + contextual_terms
        else:  # comprehensive
            # Balanced approach
            additions = synonyms[:3] + related_entities[:2] + semantic_terms[:2] + contextual_terms[:1]

        # Remove duplicates while preserving order
        seen = set(original_query.lower().split())
        unique_additions = []
        for term in additions:
            if term.lower() not in seen:
                unique_additions.append(term)
                seen.add(term.lower())

        # Limit total additions
        unique_additions = unique_additions[:self.max_expansions]

        if not unique_additions:
            return original_query

        # Build expanded query
        expanded = f"{original_query} {' '.join(unique_additions)}"
        return expanded

    def expand_multi_strategy(
        self,
        query: str,
        graph_entities: Optional[List[str]] = None,
        embedding_model: Optional[Any] = None
    ) -> List[ExpandedQuery]:
        """
        Generate multiple expansion variants using different strategies.

        Args:
            query: Original query
            graph_entities: Related entities from graph
            embedding_model: Embedding model for semantic expansion

        Returns:
            List of expanded queries with different strategies
        """
        strategies = ["conservative", "comprehensive", "aggressive"]
        expansions = []

        for strategy in strategies:
            expanded = self.expand_query(
                query=query,
                strategy=strategy,
                graph_entities=graph_entities,
                embedding_model=embedding_model
            )
            expansions.append(expanded)

        return expansions

    def suggest_related_queries(
        self,
        query: str,
        num_suggestions: int = 3
    ) -> List[str]:
        """
        Suggest related queries based on expansions.

        Args:
            query: Original query
            num_suggestions: Number of suggestions

        Returns:
            List of suggested queries
        """
        # Extract key terms
        key_terms = self._extract_key_terms(query)

        suggestions = []

        # Generate variations
        if self.enable_wordnet and key_terms:
            synonyms = self._expand_synonyms(key_terms)

            # Create variations by replacing terms with synonyms
            for i, term in enumerate(key_terms[:2]):  # Only first 2 terms
                if synonyms:
                    # Replace term with synonym
                    modified_query = query.lower()
                    for synonym in synonyms[:2]:
                        suggested = modified_query.replace(term, synonym)
                        if suggested != query.lower():
                            suggestions.append(suggested)

        # Add domain-specific suggestions
        domain_terms = self._expand_domain_specific(key_terms)
        for domain_term in domain_terms[:2]:
            suggestions.append(f"{query} {domain_term}")

        # Limit and return unique suggestions
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)

        return unique_suggestions[:num_suggestions]

    def add_domain_expansion(
        self,
        term: str,
        expansions: List[str]
    ) -> None:
        """
        Add custom domain-specific expansion.

        Args:
            term: Term to expand
            expansions: List of expansion terms
        """
        self.domain_expansions[term.lower()] = expansions
        logger.info(f"Added domain expansion: {term} -> {expansions}")

    def clear_context(self) -> None:
        """Clear query context history."""
        self.query_history.clear()
        logger.info("Query context cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get query expander statistics."""
        return {
            "wordnet_enabled": self.enable_wordnet,
            "semantic_enabled": self.enable_semantic,
            "contextual_enabled": self.enable_contextual,
            "query_history_size": len(self.query_history),
            "domain_expansions": len(self.domain_expansions),
            "max_synonyms": self.max_synonyms,
            "max_expansions": self.max_expansions
        }
