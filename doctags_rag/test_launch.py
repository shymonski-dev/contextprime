#!/usr/bin/env python3
"""
Minimal launch test - checks what can actually run
"""

import sys
from pathlib import Path

print("=" * 80)
print("Contextprime - Launch Test")
print("=" * 80)
print()

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Test 1: Check Python
print("✓ Python:", sys.version.split()[0])

# Test 2: Check dependencies
print("\nChecking dependencies:")
deps_status = {}

critical_deps = [
    ('loguru', 'Logging'),
    ('neo4j', 'Graph Database'),
    ('qdrant_client', 'Vector Database'),
    ('openai', 'LLM'),
    ('spacy', 'NLP'),
]

for module, name in critical_deps:
    try:
        __import__(module)
        print(f"  ✓ {name:20} ({module})")
        deps_status[module] = True
    except ImportError:
        print(f"  ✗ {name:20} ({module}) - MISSING")
        deps_status[module] = False

# Test 3: Try importing modules
print("\nChecking module imports:")

if deps_status.get('loguru'):
    try:
        from src.core.config import get_settings
        print("  ✓ Configuration module")
    except Exception as e:
        print(f"  ✗ Configuration module: {e}")
else:
    print("  ⊘ Configuration module (needs loguru)")

# Test 4: Database connectivity
print("\nChecking databases:")

if deps_status.get('neo4j'):
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        driver.verify_connectivity()
        driver.close()
        print("  ✓ Neo4j connected")
    except Exception as e:
        print(f"  ✗ Neo4j: {str(e)[:60]}")
else:
    print("  ⊘ Neo4j (neo4j package not installed)")

if deps_status.get('qdrant_client'):
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        print("  ✓ Qdrant connected")
    except Exception as e:
        print(f"  ✗ Qdrant: {str(e)[:60]}")
else:
    print("  ⊘ Qdrant (qdrant-client package not installed)")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

installed = sum(deps_status.values())
total = len(deps_status)

if installed == 0:
    print("\n⚠ No dependencies installed")
    print("\nTo install minimum dependencies:")
    print("  pip install loguru neo4j qdrant-client")
    print("\nTo install all dependencies:")
    print("  pip install -r requirements.txt")
elif installed < total:
    print(f"\n⚠ Partial installation: {installed}/{total} dependencies")
    print("\nTo install all dependencies:")
    print("  pip install -r requirements.txt")
else:
    print(f"\n✓ All critical dependencies installed ({installed}/{total})")
    print("\nNext steps:")
    print("  1. Start databases: docker-compose up -d (from parent dir)")
    print("  2. Run tests: pytest tests/ -v")
    print("  3. Try demos: python scripts/demo_processing.py")

print()
