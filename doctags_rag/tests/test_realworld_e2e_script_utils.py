import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pdf_realworld_e2e.py"
MODULE_NAME = "run_pdf_realworld_e2e"

spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)


def test_term_coverage_scoring():
    coverage = module._term_coverage(
        "engine oil capacity and service interval details",
        ["engine oil", "capacity", "battery"],
    )
    assert coverage == 2 / 3


def test_search_payload_defaults():
    payload = module._build_search_payload("test query")
    assert payload["query"] == "test query"
    assert payload["graph_policy"] == "community"
    assert payload["strategy"] == "hybrid"
    assert payload["top_k"] == 8


def test_resolve_auth_token_prefers_explicit_token(monkeypatch):
    monkeypatch.setenv("SECURITY__JWT_SECRET", "x" * 40)
    token = module._resolve_auth_token("explicit-token", "runner")
    assert token == "explicit-token"


def test_resolve_auth_token_builds_signed_token_from_secret(monkeypatch):
    monkeypatch.setenv("SECURITY__JWT_SECRET", "y" * 40)
    token = module._resolve_auth_token("", "runner")
    assert token.count(".") == 2


def test_resolve_auth_token_falls_back_to_access_token(monkeypatch):
    monkeypatch.delenv("SECURITY__JWT_SECRET", raising=False)
    monkeypatch.delenv("SECURITY_JWT_SECRET", raising=False)
    monkeypatch.setenv("SECURITY__ACCESS_TOKEN", "fallback-token")
    token = module._resolve_auth_token("", "runner")
    assert token == "fallback-token"


def test_build_fallback_queries_returns_unique_candidates():
    spec = module.QuerySpec(
        query="What troubleshooting or diagnostic guidance is present?",
        expected_terms=["troubleshooting", "diagnostic", "fault", "issue", "check"],
        answer_terms=["troubleshooting", "diagnostic", "guidance"],
    )
    queries = module._build_fallback_queries(spec)
    assert len(queries) >= 3
    assert len(queries) == len(set(item.lower() for item in queries))
    assert queries[-1] == "document overview and key points"
