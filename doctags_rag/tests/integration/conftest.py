"""
Fixtures for integration tests.

Services required:
- Qdrant at localhost:6333 (no auth)
- Neo4j at bolt://localhost:7687 (neo4j / see .env NEO4J_PASSWORD)
- OpenAI/OpenRouter API key (from .env OPENAI_API_KEY)

All tests in this directory are marked @pytest.mark.integration and are
skipped automatically when the services are not reachable.
"""

import os
import uuid
import pytest
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env so API keys + NEO4J_PASSWORD are available ──────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ── Host-side overrides — containers are exposed on localhost ──────────────────
# Force these BEFORE any call to get_settings() so the singleton gets the right values.
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"

# Reset the cached singleton so the next get_settings() call uses our overrides.
try:
    from src.core.config import reset_settings
    reset_settings()
except Exception:
    pass


# ── Reachability guards ────────────────────────────────────────────────────────

def _qdrant_reachable() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(host="localhost", port=6333, check_compatibility=False).get_collections()
        return True
    except Exception:
        return False


def _neo4j_reachable() -> bool:
    try:
        from neo4j import GraphDatabase
        password = os.environ.get("NEO4J_PASSWORD", "replace_with_strong_neo4j_password")
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))
        with driver.session() as s:
            s.run("RETURN 1").single()
        driver.close()
        return True
    except Exception:
        return False


_SERVICES_UP = _qdrant_reachable() and _neo4j_reachable()

requires_services = pytest.mark.skipif(
    not _SERVICES_UP,
    reason="Qdrant or Neo4j not reachable — skipping integration test",
)


# ── Qdrant test-collection lifecycle ──────────────────────────────────────────

@pytest.fixture(scope="session")
def test_collection_name():
    """A unique Qdrant collection name for this test session."""
    return f"test_web_e2e_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session", autouse=False)
def cleanup_test_collection(test_collection_name):
    """Delete the test Qdrant collection after the session."""
    yield
    if not _qdrant_reachable():
        return
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
        cols = [c.name for c in client.get_collections().collections]
        if test_collection_name in cols:
            client.delete_collection(test_collection_name)
    except Exception:
        pass


# ── pytest-httpserver fixture (re-exported for convenience) ───────────────────

@pytest.fixture()
def test_page_url(httpserver):
    """Serve the static test_page.html and return its URL."""
    html_path = Path(__file__).parent / "fixtures" / "test_page.html"
    html_content = html_path.read_bytes()
    httpserver.expect_request("/test_page.html").respond_with_data(
        html_content, content_type="text/html; charset=utf-8"
    )
    return httpserver.url_for("/test_page.html")
