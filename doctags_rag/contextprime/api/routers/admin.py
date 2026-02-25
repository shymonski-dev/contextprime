"""Administrative API routes for local recovery tasks."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

from ..models import (
    Neo4jConnectivityResponse,
    Neo4jPasswordRecoveryRequest,
    Neo4jPasswordRecoveryResponse,
)
from contextprime.core.config import get_settings

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin_identity(request: Request) -> None:
    """Reject the request with 403 unless the caller holds an admin or owner role."""
    identity = getattr(request.state, "auth_identity", None)
    if identity is None:
        raise HTTPException(status_code=403, detail="Forbidden")
    roles = set(identity.get("roles", []))
    if not roles.intersection({"admin", "owner"}):
        raise HTTPException(status_code=403, detail="Forbidden")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE_PATH = PROJECT_ROOT / ".env"


@router.get("/neo4j/connectivity", response_model=Neo4jConnectivityResponse)
async def neo4j_connectivity(_: None = Depends(require_admin_identity)) -> Neo4jConnectivityResponse:
    """Check whether configured Neo4j credentials are valid."""
    settings = get_settings()
    password = (settings.neo4j.password or "").strip()
    if not password:
        return Neo4jConnectivityResponse(
            connected=False,
            configured_password_present=False,
            message="Neo4j password is not configured.",
        )

    try:
        await run_in_threadpool(
            _verify_neo4j_credentials,
            settings.neo4j.uri,
            settings.neo4j.username,
            password,
        )
        return Neo4jConnectivityResponse(
            connected=True,
            configured_password_present=True,
            message="Configured Neo4j password is valid.",
        )
    except AuthError:
        return Neo4jConnectivityResponse(
            connected=False,
            configured_password_present=True,
            message="Configured Neo4j password is invalid.",
        )
    except (ServiceUnavailable, Neo4jError):
        return Neo4jConnectivityResponse(
            connected=False,
            configured_password_present=True,
            message="Unable to reach Neo4j service.",
        )
    except Exception as exc:  # pragma: no cover - defensive
        error_id = uuid4().hex[:12]
        logger.exception("Neo4j connectivity check failed (error_id=%s)", error_id)
        return Neo4jConnectivityResponse(
            connected=False,
            configured_password_present=True,
            message=f"Connectivity check failed (reference: {error_id})",
        )


@router.post("/neo4j/recover-password", response_model=Neo4jPasswordRecoveryResponse)
async def recover_neo4j_password(
    body: Neo4jPasswordRecoveryRequest,
    request: Request,
    _: None = Depends(require_admin_identity),
) -> Neo4jPasswordRecoveryResponse:
    """Verify Neo4j password and optionally write it to local .env."""
    candidate = body.password.strip()
    if not candidate:
        raise HTTPException(status_code=400, detail="Neo4j password is required.")

    settings = get_settings()
    try:
        await run_in_threadpool(
            _verify_neo4j_credentials,
            settings.neo4j.uri,
            settings.neo4j.username,
            candidate,
        )
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="Neo4j authentication failed.") from exc
    except (ServiceUnavailable, Neo4jError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify password because Neo4j service is unavailable.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        error_id = uuid4().hex[:12]
        logger.exception("Neo4j password verification failed (error_id=%s)", error_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error (reference: {error_id})",
        ) from exc

    settings.neo4j.password = candidate
    persisted = False
    if body.persist_to_env:
        await run_in_threadpool(_upsert_env_password, ENV_FILE_PATH, candidate)
        persisted = True

    message = "Neo4j password verified."
    if persisted:
        message += " Password saved to local .env."

    return Neo4jPasswordRecoveryResponse(
        success=True,
        persisted=persisted,
        message=message,
    )


def _verify_neo4j_credentials(uri: str, username: str, password: str) -> None:
    """Verify Neo4j credentials against the configured server."""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
    finally:
        driver.close()


def _upsert_env_password(env_file_path: Path, password: str) -> None:
    """Write NEO4J_PASSWORD in a local .env file while preserving other lines."""
    env_file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if env_file_path.exists():
        lines = env_file_path.read_text(encoding="utf-8").splitlines()

    updated = False
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("NEO4J_PASSWORD="):
            lines[idx] = f"NEO4J_PASSWORD={password}"
            updated = True
            break

    if not updated:
        lines.append(f"NEO4J_PASSWORD={password}")

    payload = "\n".join(lines).rstrip() + "\n"
    tmp_path = env_file_path.with_suffix(env_file_path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(env_file_path)
