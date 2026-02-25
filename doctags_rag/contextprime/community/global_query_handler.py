"""
Global Query Handler for answering queries using community-level insights.

Handles queries requiring global understanding such as:
- "What are the main themes?"
- "How are topics related?"
- "What communities exist?"
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import time

import networkx as nx
from openai import OpenAI
from loguru import logger

from .community_detector import CommunityResult
from .community_summarizer import CommunitySummary, GlobalSummary
from .graph_analyzer import GraphAnalyzer


@dataclass
class QueryResponse:
    """Response to a global query."""
    query: str
    answer: str
    evidence: List[Dict[str, Any]]
    confidence: float
    relevant_communities: List[int]
    metadata: Dict[str, Any] = None


class GlobalQueryHandler:
    """
    Handles queries requiring global understanding of the knowledge graph.

    Uses community structure and summaries to provide high-level insights.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize global query handler.

        Args:
            api_key: OpenAI API key
            model: Model to use for query answering
        """
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.graph_analyzer = GraphAnalyzer()

    def answer_query(
        self,
        query: str,
        graph: nx.Graph,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary],
        global_summary: Optional[GlobalSummary] = None
    ) -> QueryResponse:
        """
        Answer a global query using community insights.

        Args:
            query: User query
            graph: NetworkX graph
            community_result: Community detection result
            community_summaries: Community summaries
            global_summary: Global summary (optional)

        Returns:
            QueryResponse object
        """
        logger.info(f"Answering global query: {query}")
        start_time = time.time()

        # Classify query type
        query_type = self._classify_query(query)

        # Route to appropriate handler
        if query_type == "theme":
            response = self._handle_theme_query(
                query, community_summaries, global_summary
            )
        elif query_type == "community":
            response = self._handle_community_query(
                query, community_result, community_summaries
            )
        elif query_type == "relationship":
            response = self._handle_relationship_query(
                query, graph, community_result, community_summaries
            )
        elif query_type == "structure":
            response = self._handle_structure_query(
                query, graph, community_result, global_summary
            )
        else:
            response = self._handle_general_query(
                query, graph, community_result, community_summaries, global_summary
            )

        response.metadata = {
            "execution_time": time.time() - start_time,
            "query_type": query_type
        }

        return response

    def _classify_query(self, query: str) -> str:
        """Classify query into types."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["theme", "topic", "about", "main"]):
            return "theme"
        elif any(word in query_lower for word in ["community", "communities", "group", "cluster"]):
            return "community"
        elif any(word in query_lower for word in ["related", "connection", "relationship", "link"]):
            return "relationship"
        elif any(word in query_lower for word in ["structure", "organized", "overview", "summary"]):
            return "structure"
        else:
            return "general"

    def _handle_theme_query(
        self,
        query: str,
        community_summaries: Dict[int, CommunitySummary],
        global_summary: Optional[GlobalSummary]
    ) -> QueryResponse:
        """Handle queries about themes and topics."""
        # Collect all themes
        all_themes = []
        theme_communities = {}

        for comm_id, summary in community_summaries.items():
            for theme in summary.themes:
                all_themes.append(theme)
                if theme not in theme_communities:
                    theme_communities[theme] = []
                theme_communities[theme].append(comm_id)

        # Count theme frequencies
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Get top themes
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Build answer
        if top_themes:
            answer_parts = ["The main themes are:"]
            for theme, count in top_themes[:5]:
                communities = theme_communities[theme]
                answer_parts.append(
                    f"- {theme} (appears in {count} communities: {communities[:3]})"
                )
            answer = "\n".join(answer_parts)
        else:
            answer = "No clear themes were identified in the data."

        # Build evidence
        evidence = [
            {"theme": theme, "frequency": count, "communities": theme_communities[theme]}
            for theme, count in top_themes
        ]

        return QueryResponse(
            query=query,
            answer=answer,
            evidence=evidence,
            confidence=0.8 if top_themes else 0.3,
            relevant_communities=list(community_summaries.keys())
        )

    def _handle_community_query(
        self,
        query: str,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary]
    ) -> QueryResponse:
        """Handle queries about communities."""
        num_communities = community_result.num_communities

        # Get community sizes
        community_info = []
        for comm_id, summary in community_summaries.items():
            community_info.append({
                "id": comm_id,
                "size": summary.size,
                "title": summary.title,
                "summary": summary.brief_summary
            })

        # Sort by size
        community_info.sort(key=lambda x: x["size"], reverse=True)

        # Build answer
        answer_parts = [f"There are {num_communities} main communities:"]

        for info in community_info[:5]:
            answer_parts.append(
                f"\n{info['id']}. {info['title']} ({info['size']} entities)\n"
                f"   {info['summary']}"
            )

        if len(community_info) > 5:
            answer_parts.append(f"\n...and {len(community_info) - 5} more communities.")

        answer = "".join(answer_parts)

        return QueryResponse(
            query=query,
            answer=answer,
            evidence=community_info,
            confidence=0.9,
            relevant_communities=list(community_summaries.keys())
        )

    def _handle_relationship_query(
        self,
        query: str,
        graph: nx.Graph,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary]
    ) -> QueryResponse:
        """Handle queries about relationships."""
        # Analyze cross-community relationships
        cross_comm_edges = 0
        comm_connections = {}

        for source, target in graph.edges():
            source_comm = community_result.node_to_community.get(source)
            target_comm = community_result.node_to_community.get(target)

            if source_comm is not None and target_comm is not None:
                if source_comm != target_comm:
                    cross_comm_edges += 1
                    pair = tuple(sorted([source_comm, target_comm]))
                    comm_connections[pair] = comm_connections.get(pair, 0) + 1

        # Find most connected community pairs
        top_connections = sorted(
            comm_connections.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Build answer
        answer_parts = [
            f"The graph has {graph.number_of_edges()} total relationships, "
            f"with {cross_comm_edges} crossing community boundaries."
        ]

        if top_connections:
            answer_parts.append("\n\nMost connected communities:")
            for (comm1, comm2), count in top_connections:
                title1 = community_summaries.get(comm1, {}).title if comm1 in community_summaries else f"Community {comm1}"
                title2 = community_summaries.get(comm2, {}).title if comm2 in community_summaries else f"Community {comm2}"
                answer_parts.append(f"- {title1} <-> {title2}: {count} connections")

        answer = "".join(answer_parts)

        evidence = [
            {"communities": pair, "connection_count": count}
            for pair, count in top_connections
        ]

        return QueryResponse(
            query=query,
            answer=answer,
            evidence=evidence,
            confidence=0.85,
            relevant_communities=[c for pair in top_connections for c in pair]
        )

    def _handle_structure_query(
        self,
        query: str,
        graph: nx.Graph,
        community_result: CommunityResult,
        global_summary: Optional[GlobalSummary]
    ) -> QueryResponse:
        """Handle queries about overall structure."""
        # Compute graph metrics
        metrics = self.graph_analyzer.analyze_graph(graph, compute_diameter=False)

        answer_parts = [
            f"The knowledge graph contains {metrics.num_nodes} entities "
            f"and {metrics.num_edges} relationships.",
            f"\nIt is organized into {community_result.num_communities} communities "
            f"with a modularity score of {community_result.modularity:.3f}.",
            f"\nThe graph has {metrics.num_connected_components} connected components "
            f"with an average degree of {metrics.avg_degree:.2f}."
        ]

        if global_summary and global_summary.overall_structure:
            answer_parts.append(f"\n\n{global_summary.overall_structure}")

        answer = "".join(answer_parts)

        evidence = [{
            "num_nodes": metrics.num_nodes,
            "num_edges": metrics.num_edges,
            "num_communities": community_result.num_communities,
            "modularity": community_result.modularity,
            "avg_degree": metrics.avg_degree
        }]

        return QueryResponse(
            query=query,
            answer=answer,
            evidence=evidence,
            confidence=0.95,
            relevant_communities=list(range(community_result.num_communities))
        )

    def _handle_general_query(
        self,
        query: str,
        graph: nx.Graph,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary],
        global_summary: Optional[GlobalSummary]
    ) -> QueryResponse:
        """Handle general queries using LLM if available."""
        if self.client is None:
            return self._fallback_general_query(
                query, graph, community_result, community_summaries, global_summary
            )

        # Prepare context from community summaries
        context = self._prepare_query_context(
            graph, community_result, community_summaries, global_summary
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a knowledgeable assistant analyzing a knowledge graph. "
                            "Answer queries using the provided community-level information."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuery: {query}"
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )

            answer = response.choices[0].message.content.strip()

            return QueryResponse(
                query=query,
                answer=answer,
                evidence=[{"source": "llm", "context": context}],
                confidence=0.7,
                relevant_communities=list(community_summaries.keys())
            )

        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return self._fallback_general_query(
                query, graph, community_result, community_summaries, global_summary
            )

    def _fallback_general_query(
        self,
        query: str,
        graph: nx.Graph,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary],
        global_summary: Optional[GlobalSummary]
    ) -> QueryResponse:
        """Fallback for general queries without LLM."""
        answer = (
            f"The knowledge graph contains {graph.number_of_nodes()} entities "
            f"organized into {community_result.num_communities} communities. "
            f"For more specific information, please ask about themes, communities, "
            f"relationships, or structure."
        )

        return QueryResponse(
            query=query,
            answer=answer,
            evidence=[],
            confidence=0.5,
            relevant_communities=[]
        )

    def _prepare_query_context(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        community_summaries: Dict[int, CommunitySummary],
        global_summary: Optional[GlobalSummary]
    ) -> str:
        """Prepare context for LLM query."""
        context_parts = []

        # Graph overview
        context_parts.append(
            f"Graph: {graph.number_of_nodes()} entities, "
            f"{graph.number_of_edges()} relationships, "
            f"{community_result.num_communities} communities"
        )

        # Community summaries (top 5)
        context_parts.append("\nCommunities:")
        sorted_summaries = sorted(
            community_summaries.values(),
            key=lambda x: x.size,
            reverse=True
        )[:5]

        for summary in sorted_summaries:
            context_parts.append(
                f"- {summary.title} ({summary.size} entities): {summary.brief_summary}"
            )

        # Global summary if available
        if global_summary:
            if global_summary.main_themes:
                context_parts.append(
                    f"\nMain themes: {', '.join(global_summary.main_themes[:5])}"
                )

        return "\n".join(context_parts)
