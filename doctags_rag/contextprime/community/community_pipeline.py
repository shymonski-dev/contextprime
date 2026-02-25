"""
Community Pipeline for orchestrating the complete community detection workflow.

Coordinates:
1. Graph loading
2. Community detection
3. Quality assessment
4. Summary generation
5. Storage
6. Query preparation
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

import networkx as nx
from loguru import logger

from .community_detector import CommunityDetector, CommunityAlgorithm, CommunityResult
from .graph_analyzer import GraphAnalyzer, GraphMetrics, CommunityMetrics
from .community_summarizer import CommunitySummarizer, CommunitySummary, GlobalSummary
from .document_clusterer import DocumentClusterer, ClusteringMethod
from .cross_document_analyzer import CrossDocumentAnalyzer
from .global_query_handler import GlobalQueryHandler
from .community_storage import CommunityStorage
from ..knowledge_graph.neo4j_manager import Neo4jManager


@dataclass
class PipelineResult:
    """Results from community detection pipeline."""
    community_result: CommunityResult
    graph_metrics: GraphMetrics
    community_metrics: Dict[int, CommunityMetrics]
    community_summaries: Dict[int, CommunitySummary]
    global_summary: GlobalSummary
    version: str
    execution_time: float
    metadata: Dict[str, Any]


class CommunityPipeline:
    """
    Orchestrates the complete community detection and analysis workflow.

    Provides end-to-end pipeline from graph loading to queryable community structure.
    """

    def __init__(
        self,
        neo4j_manager: Optional[Neo4jManager] = None,
        algorithm: CommunityAlgorithm = CommunityAlgorithm.AUTO,
        api_key: Optional[str] = None,
        enable_storage: bool = True,
        enable_summaries: bool = True
    ):
        """
        Initialize community pipeline.

        Args:
            neo4j_manager: Neo4j manager instance
            algorithm: Community detection algorithm
            api_key: OpenAI API key for summarization
            enable_storage: Whether to store results in Neo4j
            enable_summaries: Whether to generate summaries
        """
        self.neo4j = neo4j_manager or Neo4jManager()
        self.algorithm = algorithm
        self.enable_storage = enable_storage
        self.enable_summaries = enable_summaries

        # Initialize components
        self.detector = CommunityDetector(algorithm=algorithm)
        self.graph_analyzer = GraphAnalyzer()
        self.summarizer = CommunitySummarizer(api_key=api_key) if enable_summaries else None
        self.doc_clusterer = DocumentClusterer()
        self.cross_doc_analyzer = CrossDocumentAnalyzer()
        self.query_handler = GlobalQueryHandler(api_key=api_key)
        self.storage = CommunityStorage(neo4j_manager=self.neo4j) if enable_storage else None

    def run(
        self,
        graph: Optional[nx.Graph] = None,
        load_from_neo4j: bool = True,
        store_results: bool = True,
        version: Optional[str] = None
    ) -> PipelineResult:
        """
        Run the complete community detection pipeline.

        Args:
            graph: NetworkX graph (will load from Neo4j if None)
            load_from_neo4j: Whether to load graph from Neo4j
            store_results: Whether to store results
            version: Version identifier for storage

        Returns:
            PipelineResult with all analysis results
        """
        logger.info("Starting community detection pipeline")
        start_time = time.time()

        # Step 1: Load or build graph
        if graph is None and load_from_neo4j:
            graph = self._load_graph_from_neo4j()
        elif graph is None:
            raise ValueError("Graph must be provided or load_from_neo4j must be True")

        logger.info(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Step 2: Detect communities
        logger.info("Step 1/6: Detecting communities...")
        community_result = self.detector.detect_communities(graph)

        # Step 3: Analyze graph and community quality
        logger.info("Step 2/6: Analyzing graph metrics...")
        graph_metrics = self.graph_analyzer.analyze_graph(graph)

        logger.info("Step 3/6: Analyzing community quality...")
        community_metrics = self.graph_analyzer.analyze_communities(graph, community_result)

        # Step 4: Generate summaries
        community_summaries = {}
        global_summary = None

        if self.enable_summaries and self.summarizer:
            logger.info("Step 4/6: Generating community summaries...")
            community_summaries = self.summarizer.summarize_all_communities(
                graph,
                community_result,
                include_detailed=False
            )

            logger.info("Step 5/6: Generating global summary...")
            global_summary = self.summarizer.generate_global_summary(
                graph,
                community_result,
                community_summaries
            )
        else:
            logger.info("Step 4-5/6: Skipping summaries (disabled)")

        # Step 5: Store results
        if self.enable_storage and store_results and self.storage:
            logger.info("Step 6/6: Storing results...")
            version = self.storage.store_communities(
                community_result,
                community_summaries,
                version
            )

            if global_summary:
                self.storage.store_global_summary(global_summary, version)
        else:
            logger.info("Step 6/6: Skipping storage")
            if version is None:
                from datetime import datetime
                version = datetime.now().isoformat()

        execution_time = time.time() - start_time

        result = PipelineResult(
            community_result=community_result,
            graph_metrics=graph_metrics,
            community_metrics=community_metrics,
            community_summaries=community_summaries,
            global_summary=global_summary,
            version=version,
            execution_time=execution_time,
            metadata={
                "algorithm": community_result.algorithm,
                "num_communities": community_result.num_communities,
                "modularity": community_result.modularity
            }
        )

        logger.info(
            f"Pipeline completed in {execution_time:.2f}s: "
            f"{community_result.num_communities} communities, "
            f"modularity={community_result.modularity:.3f}"
        )

        return result

    def run_incremental_update(
        self,
        new_graph: nx.Graph,
        previous_version: str
    ) -> PipelineResult:
        """
        Run incremental community detection on updated graph.

        Args:
            new_graph: Updated graph
            previous_version: Previous version identifier

        Returns:
            PipelineResult with updated communities
        """
        logger.info(f"Running incremental update from version {previous_version}")

        # Load previous communities
        if self.storage:
            previous_data = self.storage.load_communities(previous_version)
        else:
            logger.warning("Storage disabled, running full detection")
            return self.run(graph=new_graph, load_from_neo4j=False)

        # Run full detection on new graph
        # TODO: Implement smarter incremental update based on graph diff
        result = self.run(graph=new_graph, load_from_neo4j=False)

        return result

    def run_document_clustering(
        self,
        doc_embeddings: Dict[str, Any],
        doc_entities: Dict[str, Any],
        method: ClusteringMethod = ClusteringMethod.HYBRID
    ) -> Any:
        """
        Run document clustering as part of pipeline.

        Args:
            doc_embeddings: Document embeddings
            doc_entities: Document entities
            method: Clustering method

        Returns:
            Clustering result
        """
        logger.info(f"Running document clustering with method: {method.value}")

        if method == ClusteringMethod.ENTITY_BASED:
            return self.doc_clusterer.cluster_by_entities(doc_entities)
        elif method == ClusteringMethod.SEMANTIC:
            return self.doc_clusterer.cluster_by_semantics(doc_embeddings)
        elif method == ClusteringMethod.HYBRID:
            return self.doc_clusterer.hybrid_cluster(doc_embeddings, doc_entities)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

    def answer_global_query(
        self,
        query: str,
        graph: Optional[nx.Graph] = None,
        version: Optional[str] = None
    ) -> Any:
        """
        Answer a global query using community structure.

        Args:
            query: User query
            graph: Graph to query (loads from Neo4j if None)
            version: Community version to use (latest if None)

        Returns:
            Query response
        """
        # Load graph if needed
        if graph is None:
            graph = self._load_graph_from_neo4j()

        # Load community data
        if self.storage and version:
            community_data = self.storage.load_communities(version)
            # Reconstruct community result from stored data
            # For now, run fresh detection
            community_result = self.detector.detect_communities(graph)
            community_summaries = {}
            global_summary = None
        else:
            # Run fresh detection
            pipeline_result = self.run(graph=graph, load_from_neo4j=False, store_results=False)
            community_result = pipeline_result.community_result
            community_summaries = pipeline_result.community_summaries
            global_summary = pipeline_result.global_summary

        # Answer query
        response = self.query_handler.answer_query(
            query,
            graph,
            community_result,
            community_summaries,
            global_summary
        )

        return response

    def compare_algorithms(
        self,
        graph: nx.Graph,
        algorithms: Optional[List[CommunityAlgorithm]] = None
    ) -> Dict[str, CommunityResult]:
        """
        Compare different community detection algorithms.

        Args:
            graph: Graph to analyze
            algorithms: List of algorithms to compare

        Returns:
            Dictionary mapping algorithm names to results
        """
        logger.info("Comparing community detection algorithms")

        results = self.detector.compare_algorithms(graph, algorithms)

        # Analyze each result
        for algo_name, result in results.items():
            metrics = self.graph_analyzer.analyze_communities(graph, result)
            logger.info(
                f"{algo_name}: {result.num_communities} communities, "
                f"modularity={result.modularity:.3f}"
            )

        return results

    def _load_graph_from_neo4j(self) -> nx.Graph:
        """Load graph from Neo4j."""
        logger.info("Loading graph from Neo4j...")

        # Query to get all entities and relationships
        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        RETURN e1.name as source, e2.name as target,
               type(r) as rel_type, properties(r) as rel_props
        """

        results = self.neo4j.execute_query(query)

        # Build NetworkX graph
        graph = nx.DiGraph()

        for record in results:
            source = record["source"]
            target = record["target"]
            rel_type = record["rel_type"]
            rel_props = record.get("rel_props", {})

            # Add nodes if not exist
            if not graph.has_node(source):
                graph.add_node(source, entity=source)
            if not graph.has_node(target):
                graph.add_node(target, entity=target)

            # Add edge
            graph.add_edge(source, target, type=rel_type, **rel_props)

        logger.info(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        return graph

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline configuration."""
        return {
            "algorithm": self.algorithm.value,
            "storage_enabled": self.enable_storage,
            "summaries_enabled": self.enable_summaries,
            "components": {
                "detector": self.detector is not None,
                "graph_analyzer": self.graph_analyzer is not None,
                "summarizer": self.summarizer is not None,
                "doc_clusterer": self.doc_clusterer is not None,
                "cross_doc_analyzer": self.cross_doc_analyzer is not None,
                "query_handler": self.query_handler is not None,
                "storage": self.storage is not None
            }
        }

    def visualize_communities(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        output_path: str,
        layout: str = "spring"
    ) -> None:
        """
        Visualize communities (requires visualizer).

        Args:
            graph: Graph to visualize
            community_result: Community detection result
            output_path: Output file path
            layout: Layout algorithm
        """
        try:
            from .community_visualizer import CommunityVisualizer
            visualizer = CommunityVisualizer()
            visualizer.visualize_communities(
                graph,
                community_result,
                output_path,
                layout=layout
            )
        except ImportError as e:
            logger.error(f"Visualization failed: {e}")
