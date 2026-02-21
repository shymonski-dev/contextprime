"""
Enhanced Query Routing System for Contextprime.

Implements intelligent query routing with:
- Query type classification
- Complexity analysis
- Strategy selection
- Performance-based learning
- Fallback mechanisms
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import re
import json
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not available, query analysis will be limited")


class QueryType(Enum):
    """Types of queries for routing."""
    FACTUAL = "factual"  # Simple fact retrieval (what, when, where, who)
    DEFINITION = "definition"  # Definition or explanation
    RELATIONSHIP = "relationship"  # Relationship or connection queries
    COMPARISON = "comparison"  # Comparison between entities
    ANALYTICAL = "analytical"  # Analysis or reasoning
    MULTI_HOP = "multi_hop"  # Requires multiple reasoning steps
    PROCEDURAL = "procedural"  # How-to or process queries
    TEMPORAL = "temporal"  # Time-based queries
    AGGREGATION = "aggregation"  # Statistical or summary queries


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"  # Single fact lookup
    MODERATE = "moderate"  # Requires some reasoning
    COMPLEX = "complex"  # Multi-hop or deep analysis


class RetrievalStrategy(Enum):
    """Retrieval strategies."""
    VECTOR_ONLY = "vector_only"  # Pure semantic search
    GRAPH_ONLY = "graph_only"  # Pure graph traversal
    HYBRID = "hybrid"  # Combination of vector and graph
    HIERARCHICAL = "hierarchical"  # Start broad, then narrow
    MULTI_STAGE = "multi_stage"  # Multiple retrieval rounds


@dataclass
class QueryAnalysis:
    """Complete analysis of a query."""
    query_text: str
    query_type: QueryType
    complexity: QueryComplexity
    recommended_strategy: RetrievalStrategy
    confidence: float  # 0-1 confidence in classification
    key_entities: List[str] = field(default_factory=list)
    temporal_indicators: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingPerformance:
    """Track routing performance for learning."""
    strategy: RetrievalStrategy
    query_type: QueryType
    success_count: int = 0
    failure_count: int = 0
    avg_confidence: float = 0.0
    avg_results: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total


class QueryRouter:
    """
    Intelligent query routing system with learning capabilities.

    Features:
    - Multi-dimensional query classification
    - Complexity analysis
    - Strategy recommendation
    - Performance tracking
    - Adaptive routing based on history
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        enable_learning: bool = True,
        history_size: int = 1000,
        performance_file: Optional[Path] = None
    ):
        """
        Initialize query router.

        Args:
            spacy_model: spaCy model for NLP analysis
            enable_learning: Enable adaptive learning from performance
            history_size: Number of queries to keep in history
            performance_file: File to persist performance metrics
        """
        self.enable_learning = enable_learning
        self.history_size = history_size
        self.performance_file = performance_file

        # Initialize spaCy
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")

        # Query patterns for classification
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(what|who|where|when|which)\b.*\bis\b',
                r'\bwhat\s+is\b',
                r'\bwho\s+(is|was|are|were)\b',
                r'\bwhere\s+(is|was|are|were)\b',
                r'\bwhen\s+(did|was|is)\b',
            ],
            QueryType.DEFINITION: [
                r'\bdefin(e|ition)\b',
                r'\bwhat\s+(does|is)\s+\w+\s+mean\b',
                r'\bexplain\s+(what|the)\b',
                r'\bmean(ing)?\s+of\b',
            ],
            QueryType.RELATIONSHIP: [
                r'\b(how|what)\s+(is|are)\s+.+\s+(related|connected|linked)\b',
                r'\brelationship\s+between\b',
                r'\bconnection\s+between\b',
                r'\b(causes?|effects?|impacts?|influences?)\b',
                r'\bbetween\s+\w+\s+and\s+\w+\b',
            ],
            QueryType.COMPARISON: [
                r'\b(compare|comparison|difference|vs|versus)\b',
                r'\bdifferent\s+from\b',
                r'\bsimilar(ity)?\s+(to|between)\b',
                r'\bbetter\s+than\b',
                r'\bworse\s+than\b',
            ],
            QueryType.ANALYTICAL: [
                r'\b(why|how)\s+(did|does|do|is|are)\b',
                r'\banalyze|analysis\b',
                r'\bevaluate|assess\b',
                r'\breason(s|ing)?\s+(for|behind)\b',
            ],
            QueryType.MULTI_HOP: [
                r'\b(first|then|after|before|finally)\b',
                r'\bmultiple\s+(steps?|stages?)\b',
                r'\btrace\s+the\b',
                r'\bpath\s+(from|between)\b',
            ],
            QueryType.PROCEDURAL: [
                r'\bhow\s+to\b',
                r'\b(steps?|process|procedure|method)\s+(to|for)\b',
                r'\b(can|could)\s+i\s+\w+\b',
                r'\binstructions?\s+(for|to)\b',
            ],
            QueryType.TEMPORAL: [
                r'\b(recent|latest|current|now|today)\b',
                r'\b(before|after|during|since)\s+\d+\b',
                r'\b(history|historical|timeline)\b',
                r'\bover\s+time\b',
            ],
            QueryType.AGGREGATION: [
                r'\b(how\s+many|count|total|sum|average)\b',
                r'\b(all|every|each)\s+.+\s+(that|which)\b',
                r'\blist\s+(all|of)\b',
                r'\bsummarize|summary|overview\b',
            ],
        }

        # Complexity indicators
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: [
                r'\bwhat\s+is\b',
                r'\bwho\s+is\b',
                r'^\w{1,5}\s+\w+\?$',  # Very short queries
            ],
            QueryComplexity.MODERATE: [
                r'\b(how|why)\b',
                r'\bexplain\b',
                r'\bcompare\b',
            ],
            QueryComplexity.COMPLEX: [
                r'\band\s+.+\s+and\b',  # Multiple conjunctions
                r'\bmultiple\b',
                r'\ball\s+.+\s+that\b',
                r'\bstep\s+by\s+step\b',
            ],
        }

        # Strategy mappings
        self.strategy_map = {
            QueryType.FACTUAL: RetrievalStrategy.VECTOR_ONLY,
            QueryType.DEFINITION: RetrievalStrategy.VECTOR_ONLY,
            QueryType.RELATIONSHIP: RetrievalStrategy.GRAPH_ONLY,
            QueryType.COMPARISON: RetrievalStrategy.HYBRID,
            QueryType.ANALYTICAL: RetrievalStrategy.HYBRID,
            QueryType.MULTI_HOP: RetrievalStrategy.MULTI_STAGE,
            QueryType.PROCEDURAL: RetrievalStrategy.HIERARCHICAL,
            QueryType.TEMPORAL: RetrievalStrategy.HYBRID,
            QueryType.AGGREGATION: RetrievalStrategy.HYBRID,
        }

        # Performance tracking
        self.performance: Dict[Tuple[RetrievalStrategy, QueryType], RoutingPerformance] = {}
        self.query_history: deque = deque(maxlen=history_size)

        # Load performance history if available
        if self.performance_file and self.performance_file.exists():
            self._load_performance()

        logger.info("Query router initialized")

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query to determine type, complexity, and routing strategy.

        Args:
            query: Query text

        Returns:
            Complete query analysis
        """
        query_lower = query.lower().strip()

        # Classify query type
        query_type, type_confidence = self._classify_query_type(query_lower)

        # Determine complexity
        complexity = self._determine_complexity(query_lower, query_type)

        # Extract key entities
        key_entities = self._extract_entities(query)

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Detect temporal indicators
        temporal_indicators = self._extract_temporal_indicators(query_lower)

        # Recommend strategy
        recommended_strategy = self._recommend_strategy(
            query_type, complexity, len(key_entities)
        )

        analysis = QueryAnalysis(
            query_text=query,
            query_type=query_type,
            complexity=complexity,
            recommended_strategy=recommended_strategy,
            confidence=type_confidence,
            key_entities=key_entities,
            temporal_indicators=temporal_indicators,
            keywords=keywords,
            metadata={
                "query_length": len(query),
                "word_count": len(query.split()),
                "has_question_mark": query.strip().endswith("?"),
            }
        )

        logger.info(
            f"Query analyzed - Type: {query_type.value}, "
            f"Complexity: {complexity.value}, "
            f"Strategy: {recommended_strategy.value}"
        )

        return analysis

    def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[RetrievalStrategy, QueryAnalysis]:
        """
        Route query to appropriate retrieval strategy.

        Args:
            query: Query text
            context: Optional context for routing decisions

        Returns:
            Tuple of (strategy, analysis)
        """
        # Analyze query
        analysis = self.analyze_query(query)

        # Apply learning-based adjustments if enabled
        if self.enable_learning:
            strategy = self._apply_learning_adjustment(analysis)
        else:
            strategy = analysis.recommended_strategy

        # Store in history
        self.query_history.append({
            "query": query,
            "analysis": analysis,
            "strategy": strategy,
            "context": context
        })

        return strategy, analysis

    def record_performance(
        self,
        query: str,
        strategy: RetrievalStrategy,
        query_type: QueryType,
        success: bool,
        confidence: float = 0.0,
        num_results: int = 0
    ) -> None:
        """
        Record performance for learning.

        Args:
            query: Query text
            strategy: Strategy that was used
            query_type: Type of query
            success: Whether retrieval was successful
            confidence: Confidence in results
            num_results: Number of results returned
        """
        if not self.enable_learning:
            return

        key = (strategy, query_type)

        if key not in self.performance:
            self.performance[key] = RoutingPerformance(
                strategy=strategy,
                query_type=query_type
            )

        perf = self.performance[key]

        if success:
            perf.success_count += 1
        else:
            perf.failure_count += 1

        # Update running averages
        total = perf.success_count + perf.failure_count
        perf.avg_confidence = (
            perf.avg_confidence * (total - 1) + confidence
        ) / total
        perf.avg_results = (
            perf.avg_results * (total - 1) + num_results
        ) / total

        # Periodically save performance
        if total % 10 == 0 and self.performance_file:
            self._save_performance()

    def _classify_query_type(self, query: str) -> Tuple[QueryType, float]:
        """Classify query type with confidence."""
        scores = defaultdict(int)

        # Check patterns for each type
        for qtype, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    scores[qtype] += 1

        if not scores:
            # Default to factual for unclassified queries
            return QueryType.FACTUAL, 0.5

        # Get type with highest score
        max_type = max(scores.items(), key=lambda x: x[1])
        total_matches = sum(scores.values())

        # Confidence based on match strength
        confidence = min(1.0, max_type[1] / max(2, total_matches))

        return max_type[0], confidence

    def _determine_complexity(
        self,
        query: str,
        query_type: QueryType
    ) -> QueryComplexity:
        """Determine query complexity."""
        # Count indicators for each complexity level
        complexity_scores = defaultdict(int)

        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    complexity_scores[complexity] += 1

        # Additional heuristics
        word_count = len(query.split())
        question_words = len(re.findall(r'\b(what|who|where|when|why|how)\b', query))

        # Simple queries: short, single question word
        if word_count <= 6 and question_words <= 1:
            complexity_scores[QueryComplexity.SIMPLE] += 2

        # Complex queries: long, multiple question words, or inherently complex type
        if word_count > 15 or question_words > 2:
            complexity_scores[QueryComplexity.COMPLEX] += 2

        if query_type in [QueryType.MULTI_HOP, QueryType.ANALYTICAL]:
            complexity_scores[QueryComplexity.COMPLEX] += 1

        # Return complexity with highest score, default to moderate
        if complexity_scores:
            return max(complexity_scores.items(), key=lambda x: x[1])[0]
        else:
            return QueryComplexity.MODERATE

    def _recommend_strategy(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        entity_count: int
    ) -> RetrievalStrategy:
        """Recommend retrieval strategy based on query characteristics."""
        # Base strategy from query type
        base_strategy = self.strategy_map.get(query_type, RetrievalStrategy.HYBRID)

        # Adjust based on complexity
        if complexity == QueryComplexity.COMPLEX:
            # Complex queries benefit from multi-stage or hybrid
            if base_strategy == RetrievalStrategy.VECTOR_ONLY:
                return RetrievalStrategy.HYBRID
            elif base_strategy == RetrievalStrategy.GRAPH_ONLY:
                return RetrievalStrategy.HYBRID

        # Adjust based on entity count
        if entity_count > 2:
            # Multiple entities suggest relationship queries
            if base_strategy == RetrievalStrategy.VECTOR_ONLY:
                return RetrievalStrategy.HYBRID

        return base_strategy

    def _apply_learning_adjustment(
        self,
        analysis: QueryAnalysis
    ) -> RetrievalStrategy:
        """Apply performance-based learning to adjust strategy."""
        recommended = analysis.recommended_strategy
        query_type = analysis.query_type

        # Check performance of recommended strategy for this query type
        key = (recommended, query_type)
        if key not in self.performance:
            return recommended

        perf = self.performance[key]

        # If success rate is low, try alternative strategy
        if perf.success_rate < 0.4 and (perf.success_count + perf.failure_count) > 5:
            logger.info(
                f"Low success rate ({perf.success_rate:.2f}) for "
                f"{recommended.value} on {query_type.value}, "
                f"trying alternative"
            )

            # Find best alternative strategy for this query type
            alternatives = [
                (strat, qtype) for strat, qtype in self.performance.keys()
                if qtype == query_type and strat != recommended
            ]

            if alternatives:
                best_alt = max(
                    alternatives,
                    key=lambda k: self.performance[k].success_rate
                )
                return best_alt[0]

        return recommended

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        if not self.nlp:
            return []

        try:
            doc = self.nlp(query)
            entities = [ent.text for ent in doc.ents]
            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Simple extraction: nouns and key question words
        if self.nlp:
            try:
                doc = self.nlp(query)
                keywords = [
                    token.text for token in doc
                    if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop
                ]
                return keywords[:10]  # Limit to top 10
            except:
                pass

        # Fallback: simple word extraction
        words = re.findall(r'\b\w+\b', query.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'where', 'when'}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return keywords[:10]

    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query."""
        temporal_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(recent|latest|current|now|today|yesterday|tomorrow)\b',
            r'\b(before|after|during|since)\b',
            r'\b(past|future|historical)\b',
        ]

        indicators = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query)
            indicators.extend(matches)

        return indicators

    def _save_performance(self) -> None:
        """Save performance metrics to file."""
        if not self.performance_file:
            return

        try:
            data = {
                f"{strat.value}_{qtype.value}": {
                    "success_count": perf.success_count,
                    "failure_count": perf.failure_count,
                    "avg_confidence": perf.avg_confidence,
                    "avg_results": perf.avg_results,
                }
                for (strat, qtype), perf in self.performance.items()
            }

            with open(self.performance_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Performance saved to {self.performance_file}")

        except Exception as e:
            logger.error(f"Failed to save performance: {e}")

    def _load_performance(self) -> None:
        """Load performance metrics from file."""
        if not self.performance_file or not self.performance_file.exists():
            return

        try:
            with open(self.performance_file, 'r') as f:
                data = json.load(f)

            for key, metrics in data.items():
                parts = key.rsplit('_', 1)
                if len(parts) != 2:
                    continue

                strat_str, qtype_str = parts
                try:
                    strat = RetrievalStrategy(strat_str)
                    qtype = QueryType(qtype_str)

                    perf = RoutingPerformance(
                        strategy=strat,
                        query_type=qtype,
                        success_count=metrics.get("success_count", 0),
                        failure_count=metrics.get("failure_count", 0),
                        avg_confidence=metrics.get("avg_confidence", 0.0),
                        avg_results=metrics.get("avg_results", 0),
                    )

                    self.performance[(strat, qtype)] = perf

                except ValueError:
                    continue

            logger.info(f"Performance loaded from {self.performance_file}")

        except Exception as e:
            logger.error(f"Failed to load performance: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if not self.performance:
            return {"total_tracked": 0}

        total_queries = sum(
            p.success_count + p.failure_count
            for p in self.performance.values()
        )

        avg_success_rate = np.mean([
            p.success_rate for p in self.performance.values()
        ])

        by_strategy = defaultdict(lambda: {"success": 0, "failure": 0})
        for (strat, _), perf in self.performance.items():
            by_strategy[strat.value]["success"] += perf.success_count
            by_strategy[strat.value]["failure"] += perf.failure_count

        return {
            "total_tracked": total_queries,
            "avg_success_rate": float(avg_success_rate),
            "by_strategy": dict(by_strategy),
            "unique_combinations": len(self.performance),
            "history_size": len(self.query_history),
        }
