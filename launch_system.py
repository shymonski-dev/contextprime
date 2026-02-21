#!/usr/bin/env python3
"""
Contextprime - Comprehensive Launch and Test Script

This script:
1. Verifies all dependencies
2. Checks database connectivity
3. Runs system tests
4. Launches interactive demo
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def check_python_version() -> bool:
    """Check if Python version is 3.8+."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def check_dependencies() -> Tuple[List[str], List[str]]:
    """Check which dependencies are installed."""
    print_header("Checking Dependencies")

    required_packages = [
        'numpy', 'pandas', 'loguru', 'pydantic', 'pydantic_settings',
        'neo4j', 'qdrant_client', 'openai', 'anthropic',
        'spacy', 'networkx', 'scikit-learn', 'sentence_transformers',
        'rapidfuzz', 'nltk', 'diskcache', 'umap', 'hdbscan',
        'python_louvain', 'leidenalg', 'igraph', 'pyvis', 'matplotlib',
        'fastapi', 'uvicorn', 'pytest', 'paddleocr'
    ]

    installed = []
    missing = []

    for package in required_packages:
        # Handle package name variations
        import_name = package.replace('-', '_').replace('python_', '')
        try:
            importlib.import_module(import_name)
            installed.append(package)
            print_success(f"{package}")
        except ImportError:
            missing.append(package)
            print_warning(f"{package} - MISSING")

    print(f"\n{Colors.BOLD}Installed: {len(installed)}/{len(required_packages)}{Colors.END}")

    return installed, missing

def check_database_connectivity() -> Dict[str, bool]:
    """Check if databases are accessible."""
    print_header("Checking Database Connectivity")

    status = {
        'neo4j': False,
        'qdrant': False
    }

    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
        driver.verify_connectivity()
        driver.close()
        status['neo4j'] = True
        print_success("Neo4j: Connected (bolt://localhost:7687)")
    except Exception as e:
        print_warning(f"Neo4j: Not available - {str(e)[:50]}")
        print_info("  You can start Neo4j with: docker-compose up -d neo4j")

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        status['qdrant'] = True
        print_success("Qdrant: Connected (http://localhost:6333)")
    except Exception as e:
        print_warning(f"Qdrant: Not available - {str(e)[:50]}")
        print_info("  You can start Qdrant with: docker-compose up -d qdrant")

    return status

def check_module_imports() -> Dict[str, bool]:
    """Check if all custom modules can be imported."""
    print_header("Checking Module Imports")

    modules = [
        ('src.core.config', 'Configuration'),
        ('src.retrieval.hybrid_retriever', 'Hybrid Retriever'),
        ('src.retrieval.advanced_pipeline', 'Advanced Pipeline'),
        ('src.processing.document_parser', 'Document Parser'),
        ('src.processing.chunker', 'Chunker'),
        ('src.knowledge_graph.entity_extractor', 'Entity Extractor'),
        ('src.knowledge_graph.graph_builder', 'Graph Builder'),
        ('src.summarization.raptor_pipeline', 'RAPTOR Pipeline'),
        ('src.community.community_detector', 'Community Detector'),
        ('src.agents.agentic_pipeline', 'Agentic Pipeline'),
    ]

    status = {}

    # Add doctags_rag to path
    sys.path.insert(0, str(Path(__file__).parent / 'doctags_rag'))

    for module_name, display_name in modules:
        try:
            importlib.import_module(module_name)
            status[module_name] = True
            print_success(f"{display_name}")
        except Exception as e:
            status[module_name] = False
            print_error(f"{display_name}: {str(e)[:60]}")

    success_count = sum(status.values())
    print(f"\n{Colors.BOLD}Modules: {success_count}/{len(modules)} imported successfully{Colors.END}")

    return status

def run_basic_tests() -> bool:
    """Run basic functionality tests."""
    print_header("Running Basic Tests")

    try:
        # Test 1: Configuration loading
        print_info("Test 1: Configuration loading...")
        from src.core.config import get_settings
        settings = get_settings()
        print_success("Configuration loaded")

        # Test 2: Document processing
        print_info("Test 2: Document processing...")
        from src.processing.document_parser import DocumentParser
        parser = DocumentParser()
        print_success("Document parser initialized")

        # Test 3: Hybrid retriever (with fallback)
        print_info("Test 3: Hybrid retriever initialization...")
        from src.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()  # Should use lazy initialization
        print_success("Hybrid retriever initialized (with lazy loading)")

        # Test 4: Query processing
        print_info("Test 4: Query type detection...")
        query_type = retriever.detect_query_type("What is machine learning?")
        print_success(f"Query type detected: {query_type}")

        return True

    except Exception as e:
        print_error(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_system_info():
    """Display comprehensive system information."""
    print_header("Contextprime Information")

    # Count implementation files
    src_path = Path(__file__).parent / 'doctags_rag' / 'src'

    if src_path.exists():
        py_files = list(src_path.rglob('*.py'))
        total_lines = 0

        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    total_lines += len(f.readlines())
            except:
                pass

        print(f"📁 Python files: {len(py_files)}")
        print(f"📝 Lines of code: {total_lines:,}")

    # Show component status
    print(f"\n{Colors.BOLD}Components:{Colors.END}")
    print("  ✓ Dual Indexing (Neo4j + Qdrant)")
    print("  ✓ Document Processing (DocTags)")
    print("  ✓ Knowledge Graph Construction")
    print("  ✓ Advanced Retrieval Features")
    print("  ✓ RAPTOR Summarization")
    print("  ✓ Community Detection")
    print("  ✓ Agentic Feedback Loop")

def run_interactive_demo():
    """Run interactive demo mode."""
    print_header("Interactive Demo Mode")

    print("Available demos:")
    print("1. Document Processing Demo")
    print("2. Retrieval System Demo")
    print("3. Knowledge Graph Demo")
    print("4. RAPTOR Summarization Demo")
    print("5. Community Detection Demo")
    print("6. Agentic Pipeline Demo")
    print("7. Full System Integration Demo")
    print("0. Exit")

    try:
        choice = input(f"\n{Colors.BOLD}Select demo (0-7): {Colors.END}").strip()

        demos = {
            '1': 'scripts/demo_processing.py',
            '2': 'scripts/demo_advanced_retrieval.py',
            '3': 'scripts/build_sample_kg.py',
            '4': 'scripts/demo_raptor.py',
            '5': 'scripts/demo_community.py',
            '6': 'scripts/demo_agentic.py',
        }

        if choice in demos:
            script_path = Path(__file__).parent / 'doctags_rag' / demos[choice]
            if script_path.exists():
                print(f"\n{Colors.BLUE}Launching {demos[choice]}...{Colors.END}\n")
                subprocess.run([sys.executable, str(script_path)])
            else:
                print_error(f"Demo script not found: {script_path}")
        elif choice == '7':
            print_info("Full system integration demo coming soon!")
        elif choice == '0':
            print_info("Exiting...")
        else:
            print_warning("Invalid choice")

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo interrupted{Colors.END}")

def main():
    """Main launch function."""
    print(f"""
{Colors.BOLD}{Colors.BLUE}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                           Contextprime v1.0                                ║
║                                                                           ║
║     Ultimate RAG combining IBM structure preservation with Microsoft     ║
║        cross-document intelligence for advanced agentic reasoning        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.END}
    """)

    # Change to doctags_rag directory
    project_dir = Path(__file__).parent / 'doctags_rag'
    if project_dir.exists():
        import os
        os.chdir(project_dir)
        print_info(f"Working directory: {project_dir}")

    # Run checks
    if not check_python_version():
        sys.exit(1)

    installed, missing = check_dependencies()
    db_status = check_database_connectivity()
    module_status = check_module_imports()

    # Show summary
    print_header("System Status Summary")

    if missing:
        print_warning(f"{len(missing)} missing dependencies")
        print_info("Install with: pip install -r requirements.txt")
    else:
        print_success("All dependencies installed")

    if not any(db_status.values()):
        print_warning("No databases available")
        print_info("System will work in limited mode")
        print_info("Start databases with: docker-compose up -d")
    else:
        print_success("Databases connected")

    if all(module_status.values()):
        print_success("All modules imported successfully")
    else:
        print_warning("Some modules failed to import")

    # Show system info
    show_system_info()

    # Run basic tests
    if input(f"\n{Colors.BOLD}Run basic tests? (y/N): {Colors.END}").lower() == 'y':
        run_basic_tests()

    # Launch demo
    if input(f"\n{Colors.BOLD}Launch interactive demo? (y/N): {Colors.END}").lower() == 'y':
        run_interactive_demo()

    print(f"\n{Colors.BOLD}{Colors.GREEN}Launch complete!{Colors.END}")
    print(f"\n{Colors.BLUE}For more information:{Colors.END}")
    print("  • Documentation: docs/")
    print("  • Examples: scripts/")
    print("  • Tests: pytest tests/")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Launch interrupted by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Launch failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
