#!/usr/bin/env python3
"""
Phase 4 Integration Tests
Tests cross-component workflows and measures end-to-end latency.
"""

import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from processing.pipeline import DocumentProcessingPipeline
from retrieval.qdrant_manager import QdrantManager
from knowledge_graph.neo4j_manager import Neo4jManager
from retrieval.hybrid_retriever import HybridRetriever
from summarization.raptor_pipeline import RaptorPipeline
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from agents.evaluation_agent import EvaluationAgent


class IntegrationTestRunner:
    """Runs Phase 4 integration tests."""

    def __init__(self):
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0

    def log(self, message: str, level: str = "INFO"):
        """Log test progress."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def record_test(self, name: str, passed: bool, duration_ms: float, details: str = ""):
        """Record test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1

        self.results.append({
            "name": name,
            "passed": passed,
            "duration_ms": duration_ms,
            "details": details
        })

        status = "✓ PASS" if passed else "✗ FAIL"
        self.log(f"{status} - {name} ({duration_ms:.0f}ms) {details}",
                 "PASS" if passed else "FAIL")

    async def test_document_to_retrieval_workflow(self):
        """Test 1: Document → DocTags → Index → Retrieve."""
        self.log("=" * 60)
        self.log("Test 1: Document → DocTags → Index → Retrieve Workflow")
        self.log("=" * 60)

        start_time = time.time()

        try:
            # Step 1: Process a sample document
            self.log("Step 1: Processing sample document...")
            processor = DocumentProcessingPipeline()

            # Create a test document
            test_doc = Path("tests/fixtures/sample.txt")
            if not test_doc.exists():
                test_content = """
                Machine learning is a subset of artificial intelligence.
                It focuses on algorithms that can learn from data.
                Deep learning uses neural networks with multiple layers.
                """
                test_doc.parent.mkdir(parents=True, exist_ok=True)
                test_doc.write_text(test_content)

            # Process document
            chunks = await processor.process_file(str(test_doc))

            if not chunks or len(chunks) == 0:
                raise ValueError("No chunks generated from document")

            self.log(f"  Generated {len(chunks)} chunks")

            # Step 2: Index to Qdrant
            self.log("Step 2: Indexing to Qdrant...")
            qdrant = QdrantManager()
            await qdrant.initialize()

            collection_name = "integration_test"
            await qdrant.create_collection(collection_name)

            # Add chunks to Qdrant
            for i, chunk in enumerate(chunks[:5]):  # Limit to 5 for speed
                await qdrant.add_document(
                    collection_name=collection_name,
                    doc_id=f"test_doc_{i}",
                    content=chunk.content,
                    metadata={"chunk_id": i}
                )

            self.log(f"  Indexed {min(len(chunks), 5)} chunks")

            # Step 3: Retrieve
            self.log("Step 3: Retrieving documents...")
            results = await qdrant.search(
                collection_name=collection_name,
                query_text="What is machine learning?",
                top_k=3
            )

            if not results or len(results) == 0:
                raise ValueError("No retrieval results")

            self.log(f"  Retrieved {len(results)} results")

            # Cleanup
            await qdrant.delete_collection(collection_name)
            await qdrant.close()

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Document → Index → Retrieve",
                True,
                duration,
                f"({len(chunks)} chunks, {len(results)} results)"
            )
            return True

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Document → Index → Retrieve",
                False,
                duration,
                f"Error: {str(e)}"
            )
            return False

    async def test_multi_document_knowledge_graph(self):
        """Test 2: Multi-document knowledge graph construction."""
        self.log("=" * 60)
        self.log("Test 2: Multi-document Knowledge Graph Construction")
        self.log("=" * 60)

        start_time = time.time()

        try:
            self.log("Step 1: Connecting to Neo4j...")
            neo4j = Neo4jManager()
            await neo4j.initialize()

            # Step 2: Create entities from multiple documents
            self.log("Step 2: Creating entities...")
            entities_created = 0

            # Document 1 entities
            entity1 = await neo4j.create_entity(
                name="Machine Learning",
                entity_type="concept",
                properties={"source": "doc1", "description": "ML concept"}
            )
            entities_created += 1

            entity2 = await neo4j.create_entity(
                name="Neural Networks",
                entity_type="concept",
                properties={"source": "doc1", "description": "NN concept"}
            )
            entities_created += 1

            # Document 2 entities (cross-document)
            entity3 = await neo4j.create_entity(
                name="Deep Learning",
                entity_type="concept",
                properties={"source": "doc2", "description": "DL concept"}
            )
            entities_created += 1

            self.log(f"  Created {entities_created} entities")

            # Step 3: Create relationships
            self.log("Step 3: Creating relationships...")
            relationships_created = 0

            rel1 = await neo4j.create_relationship(
                source_id=entity1,
                target_id=entity2,
                relationship_type="uses"
            )
            relationships_created += 1

            rel2 = await neo4j.create_relationship(
                source_id=entity2,
                target_id=entity3,
                relationship_type="enables"
            )
            relationships_created += 1

            self.log(f"  Created {relationships_created} relationships")

            # Step 4: Query the graph
            self.log("Step 4: Querying knowledge graph...")
            neighbors = await neo4j.get_neighbors(entity1)

            if not neighbors or len(neighbors) == 0:
                raise ValueError("No neighbors found for entity")

            self.log(f"  Found {len(neighbors)} connected entities")

            # Cleanup
            await neo4j.delete_entity(entity1)
            await neo4j.delete_entity(entity2)
            await neo4j.delete_entity(entity3)
            await neo4j.close()

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Multi-document Knowledge Graph",
                True,
                duration,
                f"({entities_created} entities, {relationships_created} rels)"
            )
            return True

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Multi-document Knowledge Graph",
                False,
                duration,
                f"Error: {str(e)}"
            )
            return False

    async def test_hybrid_retrieval_e2e(self):
        """Test 3: Hybrid retrieval (vector + graph) end-to-end."""
        self.log("=" * 60)
        self.log("Test 3: Hybrid Retrieval (Vector + Graph) End-to-End")
        self.log("=" * 60)

        start_time = time.time()

        try:
            self.log("Step 1: Initializing hybrid retrieval...")

            # Initialize components
            qdrant = QdrantManager()
            neo4j = Neo4jManager()

            await qdrant.initialize()
            await neo4j.initialize()

            hybrid = HybridRetriever(
                qdrant_manager=qdrant,
                graph_manager=neo4j
            )

            # Step 2: Setup test data
            self.log("Step 2: Setting up test data...")
            collection_name = "hybrid_test"
            await qdrant.create_collection(collection_name)

            # Add vector data
            await qdrant.add_document(
                collection_name=collection_name,
                doc_id="ml_doc",
                content="Machine learning is a powerful technique for data analysis.",
                metadata={"topic": "ML"}
            )

            # Add graph data
            entity_id = await neo4j.create_entity(
                name="Machine Learning",
                entity_type="concept",
                properties={"description": "ML techniques"}
            )

            self.log("  Test data ready")

            # Step 3: Execute hybrid search
            self.log("Step 3: Executing hybrid search...")
            results = await hybrid.search(
                query="What is machine learning?",
                collection_name=collection_name,
                top_k=5
            )

            if not results or len(results) == 0:
                raise ValueError("No hybrid retrieval results")

            self.log(f"  Retrieved {len(results)} hybrid results")

            # Cleanup
            await qdrant.delete_collection(collection_name)
            await neo4j.delete_entity(entity_id)
            await qdrant.close()
            await neo4j.close()

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Hybrid Retrieval E2E",
                True,
                duration,
                f"({len(results)} results)"
            )
            return True

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Hybrid Retrieval E2E",
                False,
                duration,
                f"Error: {str(e)}"
            )
            return False

    async def test_raptor_with_retrieval(self):
        """Test 4: RAPTOR summarization with retrieval."""
        self.log("=" * 60)
        self.log("Test 4: RAPTOR Summarization with Retrieval")
        self.log("=" * 60)

        start_time = time.time()

        try:
            self.log("Step 1: Creating sample documents...")

            # Create test documents
            test_docs = [
                "Machine learning algorithms learn patterns from data.",
                "Neural networks are inspired by biological neurons.",
                "Deep learning uses multiple layers of neural networks.",
                "Supervised learning requires labeled training data.",
                "Unsupervised learning finds patterns without labels."
            ]

            # Step 2: Initialize RAPTOR
            self.log("Step 2: Initializing RAPTOR pipeline...")
            raptor = RaptorPipeline()

            # Step 3: Build hierarchical summary
            self.log("Step 3: Building hierarchical summary tree...")
            tree = await raptor.build_tree(test_docs)

            if not tree or len(tree.get_all_nodes()) == 0:
                raise ValueError("No RAPTOR tree nodes generated")

            self.log(f"  Built tree with {len(tree.get_all_nodes())} nodes")

            # Step 4: Query the tree
            self.log("Step 4: Querying RAPTOR tree...")
            results = await raptor.query(
                tree=tree,
                query="What is deep learning?",
                top_k=3
            )

            if not results or len(results) == 0:
                raise ValueError("No RAPTOR retrieval results")

            self.log(f"  Retrieved {len(results)} results from tree")

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "RAPTOR with Retrieval",
                True,
                duration,
                f"({len(tree.get_all_nodes())} nodes, {len(results)} results)"
            )
            return True

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.record_test(
                "RAPTOR with Retrieval",
                False,
                duration,
                f"Error: {str(e)}"
            )
            return False

    async def test_agentic_pipeline_full(self):
        """Test 5: Full agentic pipeline with all components."""
        self.log("=" * 60)
        self.log("Test 5: Full Agentic Pipeline (All Components)")
        self.log("=" * 60)

        start_time = time.time()

        try:
            # Step 1: Initialize agents
            self.log("Step 1: Initializing agent system...")
            planner = PlanningAgent()
            executor = ExecutionAgent()
            evaluator = EvaluationAgent()

            # Step 2: Plan query
            self.log("Step 2: Planning query execution...")
            query = "What are the key differences between supervised and unsupervised learning?"

            plan = await planner.create_plan(query)

            if not plan or len(plan.steps) == 0:
                raise ValueError("No execution plan generated")

            self.log(f"  Generated plan with {len(plan.steps)} steps")

            # Step 3: Execute plan
            self.log("Step 3: Executing plan...")
            results = await executor.execute_plan(plan.steps)

            if not results or len(results) == 0:
                raise ValueError("No execution results")

            self.log(f"  Executed {len(results)} steps")

            # Step 4: Evaluate results
            self.log("Step 4: Evaluating results...")
            assessment = await evaluator.assess_results(
                query=query,
                plan=plan.__dict__,
                results=[r.__dict__ for r in results]
            )

            if not assessment:
                raise ValueError("No assessment generated")

            overall_score = assessment.get("overall_score", 0.0)
            self.log(f"  Assessment score: {overall_score:.2f}")

            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Agentic Pipeline Full",
                True,
                duration,
                f"({len(plan.steps)} steps, score: {overall_score:.2f})"
            )
            return True

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.record_test(
                "Agentic Pipeline Full",
                False,
                duration,
                f"Error: {str(e)}"
            )
            return False

    async def run_all_tests(self):
        """Run all Phase 4 integration tests."""
        self.log("\n" + "=" * 60)
        self.log("PHASE 4 INTEGRATION TESTS")
        self.log("=" * 60 + "\n")

        overall_start = time.time()

        # Run each test
        await self.test_document_to_retrieval_workflow()
        await self.test_multi_document_knowledge_graph()
        await self.test_hybrid_retrieval_e2e()
        await self.test_raptor_with_retrieval()
        await self.test_agentic_pipeline_full()

        overall_duration = (time.time() - overall_start) * 1000

        # Print summary
        self.log("\n" + "=" * 60)
        self.log("TEST SUMMARY")
        self.log("=" * 60)
        self.log(f"Total tests: {self.total_tests}")
        self.log(f"Passed: {self.passed_tests}")
        self.log(f"Failed: {self.total_tests - self.passed_tests}")
        self.log(f"Pass rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        self.log(f"Total duration: {overall_duration:.0f}ms")
        self.log("=" * 60)

        # Print individual results
        self.log("\nDETAILED RESULTS:")
        for result in self.results:
            status = "✓" if result["passed"] else "✗"
            self.log(f"{status} {result['name']}: {result['duration_ms']:.0f}ms - {result['details']}")

        return self.passed_tests == self.total_tests


async def main():
    """Main entry point."""
    runner = IntegrationTestRunner()
    success = await runner.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
