"""
RAPTOR Pipeline - Orchestrates the complete hierarchical summarization system.

Coordinates:
- Document chunking
- Embedding generation
- Clustering
- Summary generation
- Tree construction
- Storage
- Retrieval
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from loguru import logger
from tqdm import tqdm

from ..processing.chunker import Chunk, StructurePreservingChunker
from ..processing.doctags_processor import DocTagsDocument
from .cluster_manager import ClusterManager, ClusteringMethod
from .summary_generator import SummaryGenerator, SummaryLevel
from .tree_builder import TreeBuilder, TreeNode, TreeStats
from .tree_storage import TreeStorage
from .hierarchical_retriever import HierarchicalRetriever, RetrievalStrategy


@dataclass
class PipelineConfig:
    """Configuration for RAPTOR pipeline."""
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Clustering
    clustering_method: ClusteringMethod = ClusteringMethod.AUTO
    min_cluster_size: int = 3
    max_cluster_size: int = 50

    # Tree building
    max_tree_depth: int = 5
    target_branching_factor: int = 5

    # Summary generation
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.1

    # Retrieval
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    top_k: int = 10

    # Processing
    batch_size: int = 32
    show_progress: bool = True


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    tree_id: str
    root: TreeNode
    all_nodes: Dict[str, TreeNode]
    stats: TreeStats
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'tree_id': self.tree_id,
            'root_node_id': self.root.node_id,
            'num_nodes': len(self.all_nodes),
            'stats': {
                'total_nodes': self.stats.total_nodes,
                'leaf_nodes': self.stats.leaf_nodes,
                'max_depth': self.stats.max_depth,
                'avg_branching': self.stats.avg_branching_factor
            },
            'metadata': self.metadata,
            'processing_time': self.processing_time
        }


class RAPTORPipeline:
    """
    Complete RAPTOR pipeline for hierarchical document understanding.

    Orchestrates all components from document processing to retrieval.
    """

    def __init__(
        self,
        config: PipelineConfig,
        embeddings_model: Any,
        neo4j_manager: Any,
        qdrant_manager: Any,
        api_key: Optional[str] = None
    ):
        """
        Initialize RAPTOR pipeline.

        Args:
            config: Pipeline configuration
            embeddings_model: Model for generating embeddings
            neo4j_manager: Neo4j manager instance
            qdrant_manager: Qdrant manager instance
            api_key: API key for LLM
        """
        self.config = config
        self.embeddings_model = embeddings_model

        # Initialize components
        self.chunker = StructurePreservingChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

        self.cluster_manager = ClusterManager(
            method=config.clustering_method,
            min_cluster_size=config.min_cluster_size,
            max_cluster_size=config.max_cluster_size
        )

        self.summary_generator = SummaryGenerator(
            provider=config.llm_provider,
            model=config.llm_model,
            api_key=api_key,
            temperature=config.llm_temperature
        )

        self.tree_builder = TreeBuilder(
            cluster_manager=self.cluster_manager,
            summary_generator=self.summary_generator,
            max_depth=config.max_tree_depth,
            target_branching_factor=config.target_branching_factor
        )

        self.tree_storage = TreeStorage(
            neo4j_manager=neo4j_manager,
            qdrant_manager=qdrant_manager
        )

        self.hierarchical_retriever = HierarchicalRetriever(
            tree_storage=self.tree_storage,
            strategy=config.retrieval_strategy,
            top_k=config.top_k
        )

        logger.info("RAPTORPipeline initialized")

    def process_document(
        self,
        doc: DocTagsDocument,
        doc_id: Optional[str] = None,
        save_tree: bool = True
    ) -> PipelineResult:
        """
        Process document through complete pipeline.

        Args:
            doc: DocTags document
            doc_id: Document identifier
            save_tree: Whether to save tree to storage

        Returns:
            Pipeline result
        """
        start_time = datetime.now()

        if doc_id is None:
            doc_id = doc.doc_id

        logger.info(f"Processing document {doc_id} through RAPTOR pipeline")

        # Stage 1: Chunk document
        logger.info("Stage 1: Chunking document...")
        chunks = self.chunker.chunk_document(doc)
        logger.info(f"Created {len(chunks)} chunks")

        if not chunks:
            logger.error("No chunks created from document")
            raise ValueError("No chunks created from document")

        # Stage 2: Generate embeddings
        logger.info("Stage 2: Generating embeddings...")
        embeddings = self._generate_embeddings(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        # Stage 3: Build tree
        logger.info("Stage 3: Building hierarchical tree...")
        root, all_nodes = self.tree_builder.build_tree(
            chunks=[c.to_dict() for c in chunks],
            embeddings=embeddings,
            doc_id=doc_id
        )
        logger.info(f"Built tree with {len(all_nodes)} nodes, depth {root.level}")

        # Compute statistics
        stats = self.tree_builder.compute_tree_stats(root, all_nodes)
        logger.info(f"Tree stats: {stats}")

        # Stage 4: Save tree
        tree_id = f"{doc_id}_raptor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if save_tree:
            logger.info("Stage 4: Saving tree to storage...")
            success = self.tree_storage.save_tree(
                root=root,
                all_nodes=all_nodes,
                tree_id=tree_id,
                metadata={
                    'doc_id': doc_id,
                    'doc_title': doc.title,
                    'num_chunks': len(chunks),
                    'created_at': datetime.now().isoformat()
                }
            )

            if success:
                logger.info(f"Tree saved with ID: {tree_id}")
            else:
                logger.warning("Failed to save tree")

        # Compute processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        result = PipelineResult(
            tree_id=tree_id,
            root=root,
            all_nodes=all_nodes,
            stats=stats,
            metadata={
                'doc_id': doc_id,
                'doc_title': doc.title,
                'num_chunks': len(chunks)
            },
            processing_time=processing_time
        )

        logger.info(f"Pipeline completed in {processing_time:.2f}s")

        return result

    def process_documents_batch(
        self,
        documents: List[DocTagsDocument],
        save_trees: bool = True
    ) -> List[PipelineResult]:
        """
        Process multiple documents in batch.

        Args:
            documents: List of documents
            save_trees: Whether to save trees

        Returns:
            List of pipeline results
        """
        logger.info(f"Processing batch of {len(documents)} documents")

        results = []

        iterator = tqdm(documents) if self.config.show_progress else documents

        for doc in iterator:
            try:
                result = self.process_document(
                    doc=doc,
                    doc_id=doc.doc_id,
                    save_tree=save_trees
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process document {doc.doc_id}: {e}")
                continue

        logger.info(f"Batch processing complete: {len(results)}/{len(documents)} successful")

        return results

    def query(
        self,
        query_text: str,
        tree_id: str,
        strategy: Optional[RetrievalStrategy] = None,
        top_k: Optional[int] = None,
        return_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query a tree with hierarchical retrieval.

        Args:
            query_text: Query text
            tree_id: Tree identifier
            strategy: Retrieval strategy
            top_k: Number of results
            return_context: Whether to return full context

        Returns:
            List of results with content and scores
        """
        logger.info(f"Querying tree {tree_id}: '{query_text}'")

        # Generate query embedding
        query_embedding = self._generate_query_embedding(query_text)

        # Retrieve results
        if top_k:
            original_top_k = self.hierarchical_retriever.top_k
            self.hierarchical_retriever.top_k = top_k

        results = self.hierarchical_retriever.retrieve(
            query_embedding=query_embedding,
            tree_id=tree_id,
            query_text=query_text,
            strategy=strategy
        )

        if top_k:
            self.hierarchical_retriever.top_k = original_top_k

        # Format results
        formatted_results = []

        for result in results:
            formatted = {
                'node_id': result.node.node_id,
                'content': result.node.content,
                'score': result.score,
                'level': result.level,
                'is_leaf': result.node.is_leaf,
                'metadata': result.node.metadata
            }

            # Add context if requested
            if return_context and result.context:
                formatted['context'] = result.context

                # Build full contextual passage
                tree_data = self.tree_storage.load_tree(tree_id)
                if tree_data:
                    _, all_nodes = tree_data
                    formatted['full_context'] = self.hierarchical_retriever.get_contextual_passage(
                        result,
                        all_nodes
                    )

            formatted_results.append(formatted)

        logger.info(f"Retrieved {len(formatted_results)} results")

        return formatted_results

    def incremental_update(
        self,
        tree_id: str,
        new_chunks: List[Chunk]
    ) -> bool:
        """
        Incrementally update tree with new chunks.

        Args:
            tree_id: Tree identifier
            new_chunks: New chunks to add

        Returns:
            Success status
        """
        logger.info(f"Incrementally updating tree {tree_id} with {len(new_chunks)} chunks")

        # Load existing tree
        tree_data = self.tree_storage.load_tree(tree_id)
        if not tree_data:
            logger.error(f"Tree {tree_id} not found")
            return False

        root, all_nodes = tree_data

        # Generate embeddings for new chunks
        new_embeddings = self._generate_embeddings(new_chunks)

        # Get all leaf nodes
        leaf_nodes = [n for n in all_nodes.values() if n.is_leaf]

        # Combine with new chunks
        all_chunks = [
            {
                'content': n.content,
                'chunk_id': n.node_id,
                'metadata': n.metadata
            }
            for n in leaf_nodes
        ] + [c.to_dict() for c in new_chunks]

        all_embeddings = np.vstack([
            np.array([n.embedding for n in leaf_nodes]),
            new_embeddings
        ])

        # Rebuild tree
        doc_id = root.metadata.get('doc_id', 'doc')
        new_root, new_all_nodes = self.tree_builder.build_tree(
            chunks=all_chunks,
            embeddings=all_embeddings,
            doc_id=doc_id
        )

        # Save updated tree
        success = self.tree_storage.save_tree(
            root=new_root,
            all_nodes=new_all_nodes,
            tree_id=tree_id,
            metadata={
                'updated_at': datetime.now().isoformat(),
                'num_new_chunks': len(new_chunks)
            }
        )

        logger.info(f"Tree update {'successful' if success else 'failed'}")

        return success

    def _generate_embeddings(
        self,
        chunks: List[Chunk]
    ) -> np.ndarray:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunks

        Returns:
            Embedding matrix
        """
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings in batches
        embeddings = []
        batch_size = self.config.batch_size

        iterator = range(0, len(texts), batch_size)
        if self.config.show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            try:
                batch_embeddings = self.embeddings_model.encode(
                    batch_texts,
                    show_progress_bar=False
                )
                embeddings.append(batch_embeddings)

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i}: {e}")
                # Create zero embeddings as fallback
                batch_embeddings = np.zeros((len(batch_texts), 768))
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def _generate_query_embedding(self, query_text: str) -> np.ndarray:
        """
        Generate embedding for query.

        Args:
            query_text: Query text

        Returns:
            Query embedding
        """
        try:
            embedding = self.embeddings_model.encode(
                [query_text],
                show_progress_bar=False
            )[0]
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(768)

    def get_tree_summary(self, tree_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information about a tree.

        Args:
            tree_id: Tree identifier

        Returns:
            Tree summary or None
        """
        tree_data = self.tree_storage.load_tree(tree_id)
        if not tree_data:
            return None

        root, all_nodes = tree_data

        stats = self.tree_builder.compute_tree_stats(root, all_nodes)

        return {
            'tree_id': tree_id,
            'root_content': root.content[:200],
            'stats': {
                'total_nodes': stats.total_nodes,
                'leaf_nodes': stats.leaf_nodes,
                'internal_nodes': stats.internal_nodes,
                'max_depth': stats.max_depth,
                'avg_branching_factor': stats.avg_branching_factor,
                'nodes_per_level': stats.nodes_per_level
            },
            'metadata': root.metadata
        }

    def compare_retrieval_strategies(
        self,
        query_text: str,
        tree_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compare different retrieval strategies for a query.

        Args:
            query_text: Query text
            tree_id: Tree identifier

        Returns:
            Results for each strategy
        """
        logger.info(f"Comparing retrieval strategies for query: '{query_text}'")

        strategies = [
            RetrievalStrategy.TOP_DOWN,
            RetrievalStrategy.BOTTOM_UP,
            RetrievalStrategy.MID_LEVEL,
            RetrievalStrategy.FULL_TREE
        ]

        comparison = {}

        for strategy in strategies:
            results = self.query(
                query_text=query_text,
                tree_id=tree_id,
                strategy=strategy,
                return_context=False
            )

            comparison[strategy.value] = results

            logger.info(
                f"{strategy.value}: {len(results)} results, "
                f"avg_score={np.mean([r['score'] for r in results]):.3f}"
            )

        return comparison

    def validate_tree(self, tree_id: str) -> Dict[str, Any]:
        """
        Validate tree structure and integrity.

        Args:
            tree_id: Tree identifier

        Returns:
            Validation report
        """
        logger.info(f"Validating tree {tree_id}")

        tree_data = self.tree_storage.load_tree(tree_id)
        if not tree_data:
            return {'valid': False, 'error': 'Tree not found'}

        root, all_nodes = tree_data

        issues = []

        # Check for orphaned nodes
        for node in all_nodes.values():
            if node.parent_id and node.parent_id not in all_nodes:
                issues.append(f"Node {node.node_id} has invalid parent {node.parent_id}")

            # Check children exist
            for child_id in node.children_ids:
                if child_id not in all_nodes:
                    issues.append(f"Node {node.node_id} has invalid child {child_id}")

        # Check root is reachable from all nodes
        for node in all_nodes.values():
            path = self.tree_builder.get_node_path_to_root(node.node_id, all_nodes)
            if not path or path[-1].node_id != root.node_id:
                issues.append(f"Node {node.node_id} cannot reach root")

        # Check embeddings
        nodes_without_embeddings = sum(
            1 for n in all_nodes.values() if n.embedding is None
        )

        return {
            'valid': len(issues) == 0,
            'num_nodes': len(all_nodes),
            'num_issues': len(issues),
            'issues': issues[:10],  # Limit to first 10
            'nodes_without_embeddings': nodes_without_embeddings
        }
