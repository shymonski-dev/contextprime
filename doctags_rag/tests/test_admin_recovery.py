from pathlib import Path

from fastapi.testclient import TestClient
from neo4j.exceptions import AuthError

from src.api.main import app
from src.api.routers import admin as admin_router
from src.core.config import get_settings


def _set_auth_requirement(enabled: bool):
    settings = get_settings()
    original = settings.security.require_access_token
    settings.security.require_access_token = enabled
    return settings, original


def test_neo4j_connectivity_reports_missing_password():
    settings, original_require_token = _set_auth_requirement(False)
    original_password = settings.neo4j.password
    settings.neo4j.password = ""

    try:
        with TestClient(app) as client:
            response = client.get("/api/admin/neo4j/connectivity")
        assert response.status_code == 200
        payload = response.json()
        assert payload["connected"] is False
        assert payload["configured_password_present"] is False
    finally:
        settings.neo4j.password = original_password
        settings.security.require_access_token = original_require_token


def test_neo4j_password_recovery_saves_env(monkeypatch, tmp_path: Path):
    settings, original_require_token = _set_auth_requirement(False)
    original_password = settings.neo4j.password
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")

    monkeypatch.setattr(admin_router, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(admin_router, "_verify_neo4j_credentials", lambda uri, username, password: None)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/admin/neo4j/recover-password",
                json={"password": "new-password", "persist_to_env": True},
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["persisted"] is True
        assert "NEO4J_PASSWORD=new-password" in env_path.read_text(encoding="utf-8")
        assert settings.neo4j.password == "new-password"
    finally:
        settings.neo4j.password = original_password
        settings.security.require_access_token = original_require_token


def test_neo4j_password_recovery_rejects_invalid_password(monkeypatch):
    settings, original_require_token = _set_auth_requirement(False)
    original_password = settings.neo4j.password

    def _raise_auth_error(uri, username, password):
        raise AuthError("bad credentials")

    monkeypatch.setattr(admin_router, "_verify_neo4j_credentials", _raise_auth_error)

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/admin/neo4j/recover-password",
                json={"password": "wrong-password", "persist_to_env": False},
            )
        assert response.status_code == 401
    finally:
        settings.neo4j.password = original_password
        settings.security.require_access_token = original_require_token
