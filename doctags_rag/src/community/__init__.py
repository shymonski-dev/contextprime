"""
Community Detection Module for Contextprime.

This module provides comprehensive community detection capabilities for entity and document
clustering, enabling global insights similar to Microsoft's GraphRAG approach.

Components:
- CommunityDetector: Multiple community detection algorithms (Louvain, Leiden, etc.)
- GraphAnalyzer: Graph metrics and analysis
- CommunitySummarizer: LLM-based community summarization
- DocumentClusterer: Document clustering based on entities and semantics
- CrossDocumentAnalyzer: Cross-document relationship analysis
- GlobalQueryHandler: Global query answering using community insights
- CommunityStorage: Persist community detection results
- CommunityPipeline: Orchestrate complete community detection workflow
- CommunityVisualizer: Visualize community structures
"""

from .community_detector import CommunityDetector, CommunityAlgorithm
from .graph_analyzer import GraphAnalyzer, GraphMetrics, CommunityMetrics
from .community_summarizer import CommunitySummarizer, CommunitySummary
from .document_clusterer import DocumentClusterer, ClusteringMethod
from .cross_document_analyzer import CrossDocumentAnalyzer
from .global_query_handler import GlobalQueryHandler
from .community_storage import CommunityStorage
from .community_pipeline import CommunityPipeline
from .community_visualizer import CommunityVisualizer

__all__ = [
    "CommunityDetector",
    "CommunityAlgorithm",
    "GraphAnalyzer",
    "GraphMetrics",
    "CommunityMetrics",
    "CommunitySummarizer",
    "CommunitySummary",
    "DocumentClusterer",
    "ClusteringMethod",
    "CrossDocumentAnalyzer",
    "GlobalQueryHandler",
    "CommunityStorage",
    "CommunityPipeline",
    "CommunityVisualizer",
]
