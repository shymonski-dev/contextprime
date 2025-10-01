"""
Community Summarizer for generating LLM-based summaries of detected communities.

Generates multi-level summaries including:
- Brief overviews
- Detailed descriptions
- Key entities and themes
- Representative examples
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import time

import networkx as nx
from openai import OpenAI
from loguru import logger

from .community_detector import CommunityResult
from .graph_analyzer import GraphAnalyzer, CommunityMetrics


@dataclass
class CommunitySummary:
    """Summary of a community."""
    community_id: int
    title: str
    brief_summary: str
    detailed_summary: str
    key_entities: List[Tuple[str, float]]  # (entity, importance_score)
    themes: List[str]
    topics: List[str]
    relationships: List[Dict[str, Any]]
    representative_nodes: List[str]
    size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalSummary:
    """Global summary across all communities."""
    num_communities: int
    main_themes: List[str]
    community_summaries: List[CommunitySummary]
    cross_community_relationships: List[Dict[str, Any]]
    overall_structure: str
    key_insights: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunitySummarizer:
    """
    Generates summaries for detected communities using LLMs.

    Supports:
    - Brief and detailed summaries
    - Theme extraction
    - Representative selection
    - Multi-level summaries
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 500,
        temperature: float = 0.3
    ):
        """
        Initialize community summarizer.

        Args:
            api_key: OpenAI API key
            model: Model to use for summarization
            max_tokens: Maximum tokens per summary
            temperature: Sampling temperature
        """
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.graph_analyzer = GraphAnalyzer()

    def summarize_community(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        community_id: int,
        include_detailed: bool = True
    ) -> CommunitySummary:
        """
        Generate summary for a single community.

        Args:
            graph: NetworkX graph
            community_result: Community detection result
            community_id: Community to summarize
            include_detailed: Whether to generate detailed summary

        Returns:
            CommunitySummary object
        """
        logger.info(f"Summarizing community {community_id}")
        start_time = time.time()

        if community_id not in community_result.communities:
            raise ValueError(f"Community {community_id} not found")

        members = community_result.communities[community_id]

        # Get community subgraph
        subgraph = graph.subgraph(members)

        # Extract key entities (top nodes by PageRank)
        pagerank = nx.pagerank(subgraph)
        key_entities = sorted(
            pagerank.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # Extract relationships within community
        relationships = self._extract_relationships(subgraph)

        # Generate title and summaries
        title = self._generate_title(members, key_entities, relationships)
        brief_summary = self._generate_brief_summary(
            members, key_entities, relationships
        )

        detailed_summary = ""
        if include_detailed:
            detailed_summary = self._generate_detailed_summary(
                members, key_entities, relationships, subgraph
            )

        # Extract themes and topics
        themes = self._extract_themes(members, relationships)
        topics = self._extract_topics(members, key_entities)

        # Select representative nodes
        representative_nodes = [entity for entity, score in key_entities[:5]]

        execution_time = time.time() - start_time

        return CommunitySummary(
            community_id=community_id,
            title=title,
            brief_summary=brief_summary,
            detailed_summary=detailed_summary,
            key_entities=key_entities,
            themes=themes,
            topics=topics,
            relationships=relationships,
            representative_nodes=representative_nodes,
            size=len(members),
            metadata={
                "execution_time": execution_time,
                "num_relationships": len(relationships)
            }
        )

    def summarize_all_communities(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        include_detailed: bool = False
    ) -> Dict[int, CommunitySummary]:
        """
        Generate summaries for all communities.

        Args:
            graph: NetworkX graph
            community_result: Community detection result
            include_detailed: Whether to generate detailed summaries

        Returns:
            Dictionary mapping community IDs to summaries
        """
        logger.info(f"Summarizing {community_result.num_communities} communities")

        summaries = {}
        for comm_id in community_result.communities.keys():
            try:
                summary = self.summarize_community(
                    graph,
                    community_result,
                    comm_id,
                    include_detailed=include_detailed
                )
                summaries[comm_id] = summary
            except Exception as e:
                logger.error(f"Failed to summarize community {comm_id}: {e}")

        return summaries

    def generate_global_summary(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary]
    ) -> GlobalSummary:
        """
        Generate a global summary across all communities.

        Args:
            graph: NetworkX graph
            community_result: Community detection result
            community_summaries: Individual community summaries

        Returns:
            GlobalSummary object
        """
        logger.info("Generating global summary")

        # Extract main themes across all communities
        all_themes = []
        for summary in community_summaries.values():
            all_themes.extend(summary.themes)

        # Count theme frequency
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        main_themes = sorted(
            theme_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        main_themes = [theme for theme, count in main_themes]

        # Identify cross-community relationships
        cross_community_rels = self._identify_cross_community_relationships(
            graph, community_result
        )

        # Generate overall structure description
        overall_structure = self._describe_overall_structure(
            community_result, community_summaries, cross_community_rels
        )

        # Extract key insights
        key_insights = self._extract_key_insights(
            graph, community_result, community_summaries, cross_community_rels
        )

        return GlobalSummary(
            num_communities=community_result.num_communities,
            main_themes=main_themes,
            community_summaries=list(community_summaries.values()),
            cross_community_relationships=cross_community_rels,
            overall_structure=overall_structure,
            key_insights=key_insights,
            metadata={
                "total_nodes": graph.number_of_nodes(),
                "total_edges": graph.number_of_edges(),
                "modularity": community_result.modularity
            }
        )

    def _generate_title(
        self,
        members: Set[str],
        key_entities: List[Tuple[str, float]],
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Generate a descriptive title for the community."""
        if not key_entities:
            return f"Community ({len(members)} entities)"

        # Use top entities to create title
        top_entities = [entity for entity, score in key_entities[:3]]

        if len(top_entities) == 1:
            return f"{top_entities[0]} Community"
        elif len(top_entities) == 2:
            return f"{top_entities[0]} and {top_entities[1]} Community"
        else:
            return f"{top_entities[0]}, {top_entities[1]}, and related entities"

    def _generate_brief_summary(
        self,
        members: Set[str],
        key_entities: List[Tuple[str, float]],
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Generate a brief summary of the community."""
        size = len(members)
        num_rels = len(relationships)

        if not key_entities:
            return f"A community of {size} entities with {num_rels} relationships."

        top_entities = [entity for entity, score in key_entities[:5]]
        entities_str = ", ".join(top_entities[:3])

        if len(top_entities) > 3:
            entities_str += f", and {len(top_entities) - 3} others"

        return (
            f"A community of {size} entities centered around {entities_str}. "
            f"Contains {num_rels} key relationships."
        )

    def _generate_detailed_summary(
        self,
        members: Set[str],
        key_entities: List[Tuple[str, float]],
        relationships: List[Dict[str, Any]],
        subgraph: nx.Graph
    ) -> str:
        """Generate a detailed summary using LLM if available."""
        if self.client is None:
            return self._generate_rule_based_detailed_summary(
                members, key_entities, relationships, subgraph
            )

        # Prepare context for LLM
        context = self._prepare_llm_context(
            members, key_entities, relationships, subgraph
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at analyzing knowledge graphs and entity communities. "
                            "Generate a detailed, insightful summary of the given community."
                        )
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._generate_rule_based_detailed_summary(
                members, key_entities, relationships, subgraph
            )

    def _generate_rule_based_detailed_summary(
        self,
        members: Set[str],
        key_entities: List[Tuple[str, float]],
        relationships: List[Dict[str, Any]],
        subgraph: nx.Graph
    ) -> str:
        """Generate detailed summary using rule-based approach."""
        size = len(members)
        num_edges = subgraph.number_of_edges()
        density = nx.density(subgraph)

        summary_parts = []

        # Basic statistics
        summary_parts.append(
            f"This community contains {size} entities with {num_edges} relationships "
            f"(density: {density:.3f})."
        )

        # Key entities
        if key_entities:
            top_5 = [entity for entity, score in key_entities[:5]]
            summary_parts.append(
                f"The most central entities are: {', '.join(top_5)}."
            )

        # Relationship patterns
        if relationships:
            rel_types = {}
            for rel in relationships:
                rel_type = rel.get("type", "RELATED_TO")
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

            top_rel_types = sorted(
                rel_types.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            rel_desc = ", ".join([f"{rtype} ({count})" for rtype, count in top_rel_types])
            summary_parts.append(f"Primary relationship types: {rel_desc}.")

        return " ".join(summary_parts)

    def _prepare_llm_context(
        self,
        members: Set[str],
        key_entities: List[Tuple[str, float]],
        relationships: List[Dict[str, Any]],
        subgraph: nx.Graph
    ) -> str:
        """Prepare context for LLM summarization."""
        context_parts = []

        # Community size
        context_parts.append(f"Community size: {len(members)} entities")

        # Key entities
        top_entities = [f"{entity} (importance: {score:.3f})"
                       for entity, score in key_entities[:10]]
        context_parts.append(f"Key entities:\n" + "\n".join([f"- {e}" for e in top_entities]))

        # Sample relationships
        sample_rels = relationships[:20]
        rel_strs = []
        for rel in sample_rels:
            source = rel.get("source", "?")
            target = rel.get("target", "?")
            rel_type = rel.get("type", "RELATED_TO")
            rel_strs.append(f"- {source} --[{rel_type}]--> {target}")

        if rel_strs:
            context_parts.append(f"Sample relationships:\n" + "\n".join(rel_strs))

        # Graph metrics
        density = nx.density(subgraph)
        avg_degree = sum(dict(subgraph.degree()).values()) / len(members) if members else 0
        context_parts.append(
            f"Graph metrics: density={density:.3f}, avg_degree={avg_degree:.2f}"
        )

        return "\n\n".join(context_parts)

    def _extract_relationships(self, subgraph: nx.Graph) -> List[Dict[str, Any]]:
        """Extract relationships from subgraph."""
        relationships = []

        for source, target, data in subgraph.edges(data=True):
            rel = {
                "source": source,
                "target": target,
                "type": data.get("type", "RELATED_TO")
            }
            # Add any additional edge attributes
            for key, value in data.items():
                if key not in ["type"]:
                    rel[key] = value

            relationships.append(rel)

        return relationships

    def _extract_themes(
        self,
        members: Set[str],
        relationships: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract themes from community members and relationships."""
        themes = set()

        # Extract from relationship types
        for rel in relationships:
            rel_type = rel.get("type", "")
            if rel_type:
                # Convert to readable theme
                theme = rel_type.replace("_", " ").lower()
                themes.add(theme)

        return sorted(list(themes))[:10]

    def _extract_topics(
        self,
        members: Set[str],
        key_entities: List[Tuple[str, float]]
    ) -> List[str]:
        """Extract topics from key entities."""
        # Simple approach: use entity names as topics
        topics = [entity for entity, score in key_entities[:10]]
        return topics

    def _identify_cross_community_relationships(
        self,
        graph: nx.Graph,
        community_result: CommunityResult
    ) -> List[Dict[str, Any]]:
        """Identify relationships that cross community boundaries."""
        cross_rels = []

        for source, target, data in graph.edges(data=True):
            source_comm = community_result.node_to_community.get(source)
            target_comm = community_result.node_to_community.get(target)

            if source_comm is not None and target_comm is not None and source_comm != target_comm:
                cross_rels.append({
                    "source": source,
                    "target": target,
                    "source_community": source_comm,
                    "target_community": target_comm,
                    "type": data.get("type", "RELATED_TO")
                })

        return cross_rels

    def _describe_overall_structure(
        self,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary],
        cross_community_rels: List[Dict[str, Any]]
    ) -> str:
        """Describe the overall structure of the graph."""
        num_communities = community_result.num_communities
        num_cross_rels = len(cross_community_rels)

        # Community sizes
        sizes = [summary.size for summary in community_summaries.values()]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0

        description = (
            f"The graph is organized into {num_communities} distinct communities "
            f"with an average size of {avg_size:.1f} entities. "
            f"The largest community contains {max_size} entities. "
            f"There are {num_cross_rels} relationships connecting different communities, "
            f"indicating {'strong' if num_cross_rels > 100 else 'moderate'} inter-community connections."
        )

        return description

    def _extract_key_insights(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary],
        cross_community_rels: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key insights from the community structure."""
        insights = []

        # Modularity insight
        if community_result.modularity > 0.5:
            insights.append(
                f"High modularity ({community_result.modularity:.3f}) indicates well-defined community structure."
            )
        elif community_result.modularity < 0.3:
            insights.append(
                f"Low modularity ({community_result.modularity:.3f}) suggests overlapping or fuzzy community boundaries."
            )

        # Size distribution
        sizes = [summary.size for summary in community_summaries.values()]
        if sizes:
            size_variance = np.var(sizes)
            if size_variance > np.mean(sizes):
                insights.append(
                    "Community sizes vary significantly, indicating a hierarchical or hub-based structure."
                )

        # Cross-community connections
        if cross_community_rels:
            # Find most connected communities
            comm_connections = {}
            for rel in cross_community_rels:
                pair = tuple(sorted([rel["source_community"], rel["target_community"]]))
                comm_connections[pair] = comm_connections.get(pair, 0) + 1

            if comm_connections:
                top_pair = max(comm_connections.items(), key=lambda x: x[1])
                insights.append(
                    f"Communities {top_pair[0][0]} and {top_pair[0][1]} are most strongly connected "
                    f"with {top_pair[1]} inter-community relationships."
                )

        return insights
