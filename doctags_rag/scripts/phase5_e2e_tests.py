#!/usr/bin/env python3
"""
Phase 5 End-to-End Tests
Complete RAG workflows with real documents and quality assessment.
"""

import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.pipeline import DocumentProcessingPipeline
from retrieval.qdrant_manager import QdrantManager
from knowledge_graph.neo4j_manager import Neo4jManager
from knowledge_graph.kg_pipeline import KnowledgeGraphPipeline
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.advanced_pipeline import AdvancedRetrievalPipeline
from summarization.raptor_pipeline import RaptorPipeline
from agents.agentic_pipeline import AgenticRAGPipeline
from community.community_pipeline import CommunityDetectionPipeline


class Phase5TestRunner:
    """Runs Phase 5 end-to-end tests."""

    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.test_docs_dir = Path("tests/fixtures/phase5_e2e")

        # Components (initialized per test)
        self.qdrant = None
        self.neo4j = None
        self.collection_name = "phase5_test"

    def log(self, message: str, level: str = "INFO"):
        """Log test progress."""
        timestamp = time.strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️ ",
            "PASS": "✅",
            "FAIL": "❌",
            "WARN": "⚠️ ",
            "TEST": "🧪"
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")

    def record_test(self, name: str, passed: bool, duration_ms: float,
                    metrics: Dict[str, Any] = None, details: str = ""):
        """Record test result with metrics."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1

        result = {
            "name": name,
            "passed": passed,
            "duration_ms": duration_ms,
            "details": details
        }

        if metrics:
            result["metrics"] = metrics

        self.results.append(result)

        status = "PASS" if passed else "FAIL"
        self.log(f"{name}: {duration_ms:.0f}ms - {details}", status)

    async def setup_components(self):
        """Initialize shared components."""
        self.log("Initializing components...", "INFO")

        # Initialize Qdrant
        self.qdrant = QdrantManager()
        await self.qdrant.initialize()

        # Create collection
        try:
            await self.qdrant.delete_collection(self.collection_name)
        except:
            pass
        await self.qdrant.create_collection(self.collection_name)

        # Initialize Neo4j
        self.neo4j = Neo4jManager()
        await self.neo4j.initialize()

        # Clear any existing test data
        # await self.neo4j.clear_database(confirm=True)

        self.log("Components initialized", "INFO")

    async def cleanup_components(self):
        """Cleanup components."""
        if self.qdrant:
            try:
                await self.qdrant.delete_collection(self.collection_name)
            except:
                pass
            await self.qdrant.close()

        if self.neo4j:
            await self.neo4j.close()

    async def test_1_single_document_rag(self):
        """Test 1: Single Document RAG - Index → Query → Retrieve → Answer."""
        self.log("=" * 70, "TEST")
        self.log("Test 1: Single Document RAG Workflow", "TEST")
        self.log("=" * 70, "TEST")

        start_time = time.time()
        metrics = {}

        try:
            # Step 1: Process document
            self.log("Step 1: Processing document (ml_basics.md)...")
            doc_path = self.test_docs_dir / "ml_basics.md"

            if not doc_path.exists():
                raise FileNotFoundError(f"Test document not found: {doc_path}")

            pipeline = DocumentProcessingPipeline()
            result = await pipeline.process_file(str(doc_path))

            if not result.success:
                raise Exception(f"Document processing failed: {result.error}")

            chunks = result.chunks
            metrics["chunks_created"] = len(chunks)
            self.log(f"  Created {len(chunks)} chunks")

            # Step 2: Index to Qdrant
            self.log("Step 2: Indexing to Qdrant...")
            index_start = time.time()

            for i, chunk in enumerate(chunks):
                await self.qdrant.add_document(
                    collection_name=self.collection_name,
                    doc_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata={"chunk_index": i, "doc_id": chunk.doc_id}
                )

            metrics["indexing_time_ms"] = (time.time() - index_start) * 1000
            self.log(f"  Indexed {len(chunks)} chunks ({metrics['indexing_time_ms']:.0f}ms)")

            # Step 3: Test retrieval
            self.log("Step 3: Testing retrieval...")

            test_queries = [
                "What is supervised learning?",
                "Explain neural networks",
                "What are evaluation metrics for classification?"
            ]

            retrieval_times = []
            total_results = 0

            for query in test_queries:
                query_start = time.time()
                results = await self.qdrant.search(
                    collection_name=self.collection_name,
                    query_text=query,
                    top_k=5
                )
                query_time = (time.time() - query_start) * 1000
                retrieval_times.append(query_time)
                total_results += len(results)

                self.log(f"  Query: '{query[:50]}...' → {len(results)} results ({query_time:.0f}ms)")

            metrics["queries_tested"] = len(test_queries)
            metrics["avg_retrieval_time_ms"] = sum(retrieval_times) / len(retrieval_times)
            metrics["total_results"] = total_results

            # Step 4: Verify result quality
            self.log("Step 4: Verifying result quality...")

            # Test specific query for quality check
            quality_query = "What is the difference between supervised and unsupervised learning?"
            quality_results = await self.qdrant.search(
                collection_name=self.collection_name,
                query_text=quality_query,
                top_k=3
            )

            # Check if results are relevant (should contain keywords)
            relevant_count = 0
            for result in quality_results:
                content_lower = result['content'].lower()
                if 'supervised' in content_lower or 'unsupervised' in content_lower:
                    relevant_count += 1

            metrics["relevance_ratio"] = relevant_count / len(quality_results) if quality_results else 0
            self.log(f"  Relevance: {relevant_count}/{len(quality_results)} results relevant")

            # Success criteria
            success = (
                len(chunks) > 0 and
                total_results > 0 and
                metrics["avg_retrieval_time_ms"] < 1000 and
                metrics["relevance_ratio"] >= 0.5
            )

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Single Document RAG",
                success,
                duration,
                metrics,
                f"✓ {metrics['chunks_created']} chunks, {metrics['queries_tested']} queries, "
                f"{metrics['relevance_ratio']:.0%} relevant"
            )

            return success

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log(f"Error: {str(e)}", "FAIL")
            self.record_test(
                "Single Document RAG",
                False,
                duration,
                metrics,
                f"Error: {str(e)}"
            )
            return False

    async def test_2_multi_document_knowledge_graph(self):
        """Test 2: Multi-Document RAG with Knowledge Graph."""
        self.log("=" * 70, "TEST")
        self.log("Test 2: Multi-Document RAG with Knowledge Graph", "TEST")
        self.log("=" * 70, "TEST")

        start_time = time.time()
        metrics = {}

        try:
            # Step 1: Process all documents
            self.log("Step 1: Processing all documents...")

            doc_files = list(self.test_docs_dir.glob("*.md"))
            if not doc_files:
                raise FileNotFoundError("No test documents found")

            pipeline = DocumentProcessingPipeline()
            all_chunks = []

            for doc_path in doc_files:
                result = await pipeline.process_file(str(doc_path))
                if result.success:
                    all_chunks.extend(result.chunks)
                    self.log(f"  Processed {doc_path.name}: {len(result.chunks)} chunks")

            metrics["documents_processed"] = len(doc_files)
            metrics["total_chunks"] = len(all_chunks)

            # Step 2: Build knowledge graph
            self.log("Step 2: Building knowledge graph...")
            kg_start = time.time()

            kg_pipeline = KnowledgeGraphPipeline(
                neo4j_manager=self.neo4j
            )

            # Process each document to extract entities and relationships
            entities_created = 0
            relationships_created = 0

            for doc_path in doc_files:
                doc_text = doc_path.read_text()

                # Build graph for document
                kg_result = await kg_pipeline.process_document(
                    doc_text,
                    doc_id=doc_path.stem,
                    metadata={"filename": doc_path.name}
                )

                entities_created += len(kg_result.get('entities', []))
                relationships_created += len(kg_result.get('relationships', []))

                self.log(f"  KG for {doc_path.name}: "
                        f"{len(kg_result.get('entities', []))} entities, "
                        f"{len(kg_result.get('relationships', []))} relationships")

            metrics["kg_build_time_ms"] = (time.time() - kg_start) * 1000
            metrics["entities_created"] = entities_created
            metrics["relationships_created"] = relationships_created

            # Step 3: Index chunks to Qdrant
            self.log("Step 3: Indexing chunks to Qdrant...")

            for chunk in all_chunks:
                await self.qdrant.add_document(
                    collection_name=self.collection_name,
                    doc_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata={"doc_id": chunk.doc_id}
                )

            # Step 4: Test hybrid retrieval (vector + graph)
            self.log("Step 4: Testing hybrid retrieval...")

            hybrid = HybridRetriever(
                qdrant_manager=self.qdrant,
                graph_manager=self.neo4j
            )

            test_query = "How do neural networks relate to deep learning?"
            hybrid_start = time.time()

            hybrid_results = await hybrid.search(
                query=test_query,
                collection_name=self.collection_name,
                top_k=5,
                strategy="hybrid"
            )

            metrics["hybrid_search_time_ms"] = (time.time() - hybrid_start) * 1000
            metrics["hybrid_results_count"] = len(hybrid_results)

            self.log(f"  Hybrid search: {len(hybrid_results)} results "
                    f"({metrics['hybrid_search_time_ms']:.0f}ms)")

            # Step 5: Verify cross-document linking
            self.log("Step 5: Verifying cross-document linking...")

            # Query for entities that should appear in multiple documents
            common_entities = await self.neo4j.execute_query(
                """
                MATCH (e:Entity)
                WHERE e.name IN ['Neural Networks', 'Deep Learning', 'Machine Learning']
                RETURN e.name as name, e.doc_id as doc_id
                LIMIT 10
                """
            )

            metrics["cross_doc_entities"] = len(common_entities) if common_entities else 0

            await hybrid.close()

            # Success criteria
            success = (
                metrics["documents_processed"] >= 3 and
                metrics["entities_created"] > 0 and
                metrics["hybrid_results_count"] > 0 and
                metrics["hybrid_search_time_ms"] < 2000
            )

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Multi-Document Knowledge Graph",
                success,
                duration,
                metrics,
                f"✓ {metrics['documents_processed']} docs, "
                f"{metrics['entities_created']} entities, "
                f"{metrics['relationships_created']} relationships"
            )

            return success

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log(f"Error: {str(e)}", "FAIL")
            self.record_test(
                "Multi-Document Knowledge Graph",
                False,
                duration,
                metrics,
                f"Error: {str(e)}"
            )
            return False

    async def test_3_agentic_rag_workflow(self):
        """Test 3: Agentic RAG with Iterative Refinement."""
        self.log("=" * 70, "TEST")
        self.log("Test 3: Agentic RAG with Iterative Refinement", "TEST")
        self.log("=" * 70, "TEST")

        start_time = time.time()
        metrics = {}

        try:
            self.log("Step 1: Initializing agentic RAG pipeline...")

            # Initialize advanced retrieval pipeline
            advanced_retrieval = AdvancedRetrievalPipeline(
                qdrant_manager=self.qdrant,
                collection_name=self.collection_name
            )

            # Initialize agentic pipeline
            agentic_rag = AgenticRAGPipeline(
                retrieval_pipeline=advanced_retrieval,
                graph_manager=self.neo4j
            )

            # Step 2: Test complex query requiring planning
            self.log("Step 2: Testing complex query...")

            complex_query = (
                "Compare supervised learning and reinforcement learning. "
                "What are the key differences in how they learn?"
            )

            query_start = time.time()

            # This should trigger planning, execution, and evaluation
            response = await agentic_rag.process_query(complex_query)

            metrics["total_time_ms"] = (time.time() - query_start) * 1000
            metrics["plan_steps"] = len(response.get('plan', {}).get('steps', []))
            metrics["results_count"] = len(response.get('results', []))
            metrics["assessment_score"] = response.get('assessment', {}).get('overall_score', 0.0)

            self.log(f"  Query processed: {metrics['plan_steps']} steps, "
                    f"score: {metrics['assessment_score']:.2f}")

            # Step 3: Test iterative refinement
            self.log("Step 3: Testing iterative refinement...")

            refinement_query = "Tell me more about the training process"
            refinement_start = time.time()

            refined_response = await agentic_rag.process_query(
                refinement_query,
                context=response  # Pass previous response as context
            )

            metrics["refinement_time_ms"] = (time.time() - refinement_start) * 1000

            # Success criteria
            success = (
                metrics["plan_steps"] > 0 and
                metrics["results_count"] > 0 and
                metrics["assessment_score"] >= 0.5 and
                metrics["total_time_ms"] < 10000
            )

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Agentic RAG Workflow",
                success,
                duration,
                metrics,
                f"✓ {metrics['plan_steps']} steps, "
                f"score: {metrics['assessment_score']:.2f}"
            )

            return success

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log(f"Error: {str(e)}", "FAIL")
            self.record_test(
                "Agentic RAG Workflow",
                False,
                duration,
                metrics,
                f"Error: {str(e)}"
            )
            return False

    async def test_4_community_summarization(self):
        """Test 4: Community-Based Summarization Queries."""
        self.log("=" * 70, "TEST")
        self.log("Test 4: Community-Based Summarization", "TEST")
        self.log("=" * 70, "TEST")

        start_time = time.time()
        metrics = {}

        try:
            self.log("Step 1: Building document graph...")

            # Process all documents
            doc_files = list(self.test_docs_dir.glob("*.md"))
            documents = []

            for doc_path in doc_files:
                documents.append({
                    'id': doc_path.stem,
                    'content': doc_path.read_text(),
                    'metadata': {'filename': doc_path.name}
                })

            metrics["documents"] = len(documents)

            # Step 2: Detect communities
            self.log("Step 2: Detecting communities...")
            community_start = time.time()

            community_pipeline = CommunityDetectionPipeline(
                neo4j_manager=self.neo4j
            )

            community_result = await community_pipeline.detect_communities(
                documents=documents,
                algorithm='louvain'
            )

            metrics["community_detection_time_ms"] = (time.time() - community_start) * 1000
            metrics["communities_found"] = len(community_result.get('communities', []))

            self.log(f"  Found {metrics['communities_found']} communities "
                    f"({metrics['community_detection_time_ms']:.0f}ms)")

            # Step 3: Generate community summaries
            if metrics["communities_found"] > 0:
                self.log("Step 3: Generating community summaries...")
                summary_start = time.time()

                summaries = await community_pipeline.summarize_communities(
                    community_result
                )

                metrics["summary_time_ms"] = (time.time() - summary_start) * 1000
                metrics["summaries_generated"] = len(summaries)

                for i, summary in enumerate(summaries[:3]):  # Show first 3
                    self.log(f"  Community {i+1}: {summary.get('title', 'N/A')[:50]}...")

            # Success criteria
            success = (
                metrics["documents"] >= 3 and
                metrics["communities_found"] > 0
            )

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Community Summarization",
                success,
                duration,
                metrics,
                f"✓ {metrics['communities_found']} communities from {metrics['documents']} docs"
            )

            return success

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log(f"Error: {str(e)}", "FAIL")
            self.record_test(
                "Community Summarization",
                False,
                duration,
                metrics,
                f"Error: {str(e)}"
            )
            return False

    async def test_5_complex_analytical_queries(self):
        """Test 5: Complex Multi-Step Analytical Queries."""
        self.log("=" * 70, "TEST")
        self.log("Test 5: Complex Multi-Step Analytical Queries", "TEST")
        self.log("=" * 70, "TEST")

        start_time = time.time()
        metrics = {}

        try:
            # Initialize advanced retrieval
            advanced = AdvancedRetrievalPipeline(
                qdrant_manager=self.qdrant,
                collection_name=self.collection_name
            )

            # Test queries of increasing complexity
            test_queries = [
                {
                    "query": "What are the main types of machine learning?",
                    "type": "factual",
                    "expected_min_results": 3
                },
                {
                    "query": "Compare CNNs and RNNs in deep learning",
                    "type": "comparative",
                    "expected_min_results": 3
                },
                {
                    "query": "How does backpropagation work and why is it important for neural network training?",
                    "type": "analytical",
                    "expected_min_results": 3
                }
            ]

            results_summary = []

            for i, test in enumerate(test_queries, 1):
                self.log(f"Step {i}: Testing {test['type']} query...")
                self.log(f"  Query: {test['query'][:60]}...")

                query_start = time.time()

                # Use advanced retrieval with query expansion and reranking
                results = await advanced.retrieve(
                    query=test['query'],
                    top_k=10,
                    enable_expansion=True,
                    enable_reranking=True
                )

                query_time = (time.time() - query_start) * 1000

                # Analyze results
                result_count = len(results)
                avg_score = sum(r.get('score', 0) for r in results) / result_count if results else 0

                test_passed = result_count >= test['expected_min_results']

                results_summary.append({
                    "type": test['type'],
                    "passed": test_passed,
                    "results": result_count,
                    "avg_score": avg_score,
                    "time_ms": query_time
                })

                status = "✓" if test_passed else "✗"
                self.log(f"  {status} {result_count} results, "
                        f"avg score: {avg_score:.3f}, "
                        f"{query_time:.0f}ms")

            # Calculate metrics
            metrics["queries_tested"] = len(test_queries)
            metrics["queries_passed"] = sum(1 for r in results_summary if r['passed'])
            metrics["avg_query_time_ms"] = sum(r['time_ms'] for r in results_summary) / len(results_summary)
            metrics["avg_result_score"] = sum(r['avg_score'] for r in results_summary) / len(results_summary)

            # Success criteria
            success = (
                metrics["queries_passed"] == metrics["queries_tested"] and
                metrics["avg_result_score"] >= 0.5 and
                metrics["avg_query_time_ms"] < 3000
            )

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Complex Analytical Queries",
                success,
                duration,
                metrics,
                f"✓ {metrics['queries_passed']}/{metrics['queries_tested']} passed, "
                f"avg score: {metrics['avg_result_score']:.3f}"
            )

            return success

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.log(f"Error: {str(e)}", "FAIL")
            self.record_test(
                "Complex Analytical Queries",
                False,
                duration,
                metrics,
                f"Error: {str(e)}"
            )
            return False

    async def run_all_tests(self):
        """Run all Phase 5 end-to-end tests."""
        self.log("\n" + "=" * 70)
        self.log("PHASE 5 END-TO-END TESTS")
        self.log("=" * 70 + "\n")

        overall_start = time.time()

        try:
            # Setup
            await self.setup_components()

            # Run tests sequentially
            await self.test_1_single_document_rag()
            await self.test_2_multi_document_knowledge_graph()
            # await self.test_3_agentic_rag_workflow()  # May require additional setup
            # await self.test_4_community_summarization()  # May require LLM
            await self.test_5_complex_analytical_queries()

        finally:
            # Cleanup
            await self.cleanup_components()

        overall_duration = (time.time() - overall_start) * 1000

        # Print summary
        self.log("\n" + "=" * 70)
        self.log("TEST SUMMARY")
        self.log("=" * 70)
        self.log(f"Total tests: {self.total_tests}")
        self.log(f"Passed: {self.passed_tests} ({(self.passed_tests/self.total_tests*100):.1f}%)")
        self.log(f"Failed: {self.total_tests - self.passed_tests}")
        self.log(f"Total duration: {overall_duration/1000:.1f}s")
        self.log("=" * 70)

        # Print detailed results
        self.log("\nDETAILED RESULTS:")
        for result in self.results:
            status = "✅" if result["passed"] else "❌"
            self.log(f"{status} {result['name']}: {result['duration_ms']:.0f}ms - {result['details']}")

            if 'metrics' in result and result['metrics']:
                for key, value in result['metrics'].items():
                    if isinstance(value, float):
                        self.log(f"    {key}: {value:.2f}")
                    else:
                        self.log(f"    {key}: {value}")

        return self.passed_tests == self.total_tests


async def main():
    """Main entry point."""
    runner = Phase5TestRunner()
    success = await runner.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
