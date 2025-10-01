#!/usr/bin/env python3
"""
Verification script for the Dual Indexing Infrastructure.
Checks that all files are present and modules can be imported.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

def check_file(filepath: Path, description: str) -> bool:
    """Check if a file exists."""
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        print(f"{GREEN}✓{NC} {description}: {filepath.name} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"{RED}✗{NC} {description}: {filepath.name} NOT FOUND")
        return False

def check_imports() -> bool:
    """Check if modules can be imported."""
    print("\nChecking module imports...")

    base_path = Path(__file__).parent / "doctags_rag"
    sys.path.insert(0, str(base_path))

    imports = [
        ("src.core.config", "Config module"),
        ("src.knowledge_graph.neo4j_manager", "Neo4j Manager"),
        ("src.retrieval.qdrant_manager", "Qdrant Manager"),
        ("src.retrieval.hybrid_retriever", "Hybrid Retriever"),
    ]

    all_success = True
    for module_name, description in imports:
        try:
            __import__(module_name)
            print(f"{GREEN}✓{NC} {description} can be imported")
        except ImportError as e:
            print(f"{RED}✗{NC} {description} import failed: {e}")
            all_success = False
        except Exception as e:
            print(f"{YELLOW}⚠{NC} {description} import warning: {e}")

    return all_success

def main():
    """Main verification function."""
    print("=" * 60)
    print("Dual Indexing Infrastructure - Installation Verification")
    print("=" * 60)
    print()

    base_path = Path(__file__).parent
    doctags_path = base_path / "doctags_rag"

    # Files to check
    files_to_check = [
        # Core modules
        (doctags_path / "src/knowledge_graph/neo4j_manager.py", "Neo4j Manager"),
        (doctags_path / "src/retrieval/qdrant_manager.py", "Qdrant Manager"),
        (doctags_path / "src/retrieval/hybrid_retriever.py", "Hybrid Retriever"),

        # Init files
        (doctags_path / "src/knowledge_graph/__init__.py", "Knowledge Graph Init"),
        (doctags_path / "src/retrieval/__init__.py", "Retrieval Init"),
        (doctags_path / "tests/__init__.py", "Tests Init"),

        # Scripts
        (doctags_path / "scripts/setup_databases.py", "Setup Script"),
        (doctags_path / "scripts/example_usage.py", "Example Script"),

        # Tests
        (doctags_path / "tests/test_indexing.py", "Test Suite"),

        # Infrastructure
        (base_path / "docker-compose.yml", "Docker Compose"),
        (base_path / "monitoring/prometheus.yml", "Prometheus Config"),
        (base_path / "quickstart.sh", "QuickStart Script"),

        # Documentation
        (base_path / "DUAL_INDEXING_SETUP.md", "Setup Guide"),
        (base_path / "IMPLEMENTATION_SUMMARY.md", "Implementation Summary"),
    ]

    print("Checking files...")
    print("-" * 60)

    results = [check_file(filepath, desc) for filepath, desc in files_to_check]

    # Check imports
    import_success = check_imports()

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    total_files = len(results)
    files_found = sum(results)

    print(f"Files checked: {total_files}")
    print(f"Files found: {files_found}")
    print(f"Files missing: {total_files - files_found}")

    if files_found == total_files:
        print(f"\n{GREEN}✓ All files are present!{NC}")

        if import_success:
            print(f"{GREEN}✓ All modules can be imported!{NC}")
            print("\n{GREEN}Installation fully verified!{NC}")
        else:
            print(f"\n{YELLOW}⚠ Dependencies not installed yet{NC}")
            print("This is expected if you haven't run the quickstart script.")

        print("\nNext steps:")
        print("  1. Run: ./quickstart.sh")
        print("  2. Or manually:")
        print("     a. Start databases: docker-compose up -d neo4j qdrant")
        print("     b. Install deps: cd doctags_rag && pip install -r requirements.txt")
        print("     c. Setup databases: python scripts/setup_databases.py")
        return 0
    else:
        print(f"\n{RED}✗ Installation verification failed{NC}")
        print("\nSome files are missing.")
        print("Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
