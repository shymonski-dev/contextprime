"""Security and request limiting middleware for protected routes."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
from time import time
from typing import Optional, Set, Tuple
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.core.config import get_settings
from .services.request_limit_store import SharedSlidingWindowRateLimiter


@dataclass(frozen=True)
class AuthIdentity:
    """Authenticated caller identity."""

    subject: str
    scopes: Set[str]
    roles: Set[str]
    mode: str


class AccessControlAndRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply token validation and request limiting to protected routes."""

    _JWT_DIGESTS = {
        "HS256": "sha256",
        "HS384": "sha384",
        "HS512": "sha512",
    }

    def __init__(self, app) -> None:  # type: ignore[override]
        super().__init__(app)
        settings = get_settings()
        self._require_access_token = bool(settings.security.require_access_token)
        self._access_token = (settings.security.access_token or "").strip()
        self._auth_mode = (settings.security.auth_mode or "jwt").strip().lower()
        self._exempt_paths = {
            self._normalize_path(path) for path in settings.security.exempt_paths if path
        }
        self._token_header = (settings.security.token_header or "Authorization").strip().lower()
        self._trust_proxy_headers = bool(settings.api.trust_proxy_headers)
        self._jwt_secret = (settings.security.jwt_secret or "").strip()
        self._jwt_algorithm = (settings.security.jwt_algorithm or "HS256").strip().upper()
        self._jwt_issuer = (settings.security.jwt_issuer or "").strip() or None
        self._jwt_audience = (settings.security.jwt_audience or "").strip() or None
        self._jwt_subject_claim = (settings.security.jwt_subject_claim or "sub").strip() or "sub"
        self._jwt_roles_claim = (settings.security.jwt_roles_claim or "roles").strip() or "roles"
        self._jwt_scopes_claim = (settings.security.jwt_scopes_claim or "scopes").strip() or "scopes"
        self._jwt_enforce_permissions = bool(settings.security.jwt_enforce_permissions)
        self._jwt_required_read_scopes = self._normalise_permission_values(
            settings.security.jwt_required_read_scopes
        )
        self._jwt_required_write_scopes = self._normalise_permission_values(
            settings.security.jwt_required_write_scopes
        )
        self._jwt_admin_roles = self._normalise_permission_values(settings.security.jwt_admin_roles)
        self._rate_limiter = (
            SharedSlidingWindowRateLimiter(
                max_requests=settings.api.rate_limit,
                window_seconds=settings.api.rate_limit_window_seconds,
                redis_url=settings.api.rate_limit_redis_url,
                sqlite_path=settings.api.rate_limit_store_path,
            )
            if settings.api.rate_limit > 0
            else None
        )

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        path = self._normalize_path(request.url.path)
        if not path.startswith("/api"):
            return await call_next(request)

        if path in self._exempt_paths:
            return await call_next(request)

        auth_subject = self._request_subject(request)
        identity: Optional[AuthIdentity] = None

        if self._require_access_token:
            provided = self._extract_token(request)
            if not provided:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

            identity, error_response = self._authenticate_token(provided)
            if error_response is not None:
                return error_response
            if identity is None:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

            if self._auth_mode == "jwt" and self._jwt_enforce_permissions and identity.mode == "jwt":
                if self._is_permission_denied(path=path, method=request.method, identity=identity):
                    return JSONResponse(status_code=403, content={"detail": "Forbidden"})

            request.state.auth_identity = {
                "subject": identity.subject,
                "roles": sorted(identity.roles),
                "scopes": sorted(identity.scopes),
                "mode": identity.mode,
            }
            auth_subject = f"user:{identity.subject}"

        if self._rate_limiter and request.method.upper() != "OPTIONS":
            decision = self._rate_limiter.check(auth_subject)
            if not decision.allowed:
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )
                response.headers["Retry-After"] = str(decision.retry_after_seconds)
                return response

        return await call_next(request)

    def _authenticate_token(self, token: str) -> Tuple[Optional[AuthIdentity], Optional[JSONResponse]]:
        """Authenticate a request token with token or JWT mode."""
        mode = self._auth_mode
        if mode == "token":
            if not self._access_token:
                return None, JSONResponse(
                    status_code=503,
                    content={"detail": "Access token is required but not configured"},
                )
            if not secrets.compare_digest(token, self._access_token):
                return None, JSONResponse(status_code=401, content={"detail": "Unauthorized"})
            return AuthIdentity(
                subject=f"token:{self._token_fingerprint(token)}",
                scopes=set(),
                roles={"token_admin"},
                mode="token",
            ), None

        if mode != "jwt":
            return None, JSONResponse(
                status_code=503,
                content={"detail": "Unsupported authentication mode"},
            )

        if self._jwt_algorithm not in self._JWT_DIGESTS:
            return None, JSONResponse(
                status_code=503,
                content={"detail": "JWT algorithm is not supported"},
            )

        if not self._jwt_secret:
            return None, JSONResponse(
                status_code=503,
                content={"detail": "JWT secret is required but not configured"},
            )

        jwt_identity = self._validate_jwt(token)
        if jwt_identity is not None:
            return jwt_identity, None

        return None, JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    def _validate_jwt(self, token: str) -> Optional[AuthIdentity]:
        """Validate and decode a JWT signed with shared secret HMAC."""
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_segment, payload_segment, signature_segment = parts
        signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")

        try:
            header_bytes = self._decode_base64url(header_segment)
            payload_bytes = self._decode_base64url(payload_segment)
            signature = self._decode_base64url(signature_segment)
            header = json.loads(header_bytes.decode("utf-8"))
            claims = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            return None

        if not isinstance(header, dict) or not isinstance(claims, dict):
            return None

        algorithm = str(header.get("alg", "")).upper()
        if algorithm != self._jwt_algorithm:
            return None

        digest_name = self._JWT_DIGESTS.get(algorithm)
        if not digest_name or not self._jwt_secret:
            return None

        expected_signature = hmac.new(
            key=self._jwt_secret.encode("utf-8"),
            msg=signing_input,
            digestmod=digest_name,
        ).digest()
        if not secrets.compare_digest(signature, expected_signature):
            return None

        now = int(time())
        skew_seconds = 30

        exp_ts = self._coerce_unix_timestamp(claims.get("exp"))
        if exp_ts is not None and now >= exp_ts:
            return None

        nbf_ts = self._coerce_unix_timestamp(claims.get("nbf"))
        if nbf_ts is not None and now + skew_seconds < nbf_ts:
            return None

        iat_ts = self._coerce_unix_timestamp(claims.get("iat"))
        if iat_ts is not None and now + skew_seconds < iat_ts:
            return None

        if self._jwt_issuer:
            issuer = str(claims.get("iss", "")).strip()
            if issuer != self._jwt_issuer:
                return None

        if self._jwt_audience:
            audience = claims.get("aud")
            if not self._audience_matches(audience, self._jwt_audience):
                return None

        subject_value = claims.get(self._jwt_subject_claim)
        subject = str(subject_value or "").strip()
        if not subject:
            return None

        roles = self._normalise_permission_values(claims.get(self._jwt_roles_claim))
        scopes = self._normalise_permission_values(claims.get(self._jwt_scopes_claim))
        return AuthIdentity(subject=subject, scopes=scopes, roles=roles, mode="jwt")

    def _is_permission_denied(self, path: str, method: str, identity: AuthIdentity) -> bool:
        if self._is_admin(identity.roles):
            return False

        permission_type = self._resolve_permission_type(path=path, method=method)
        if permission_type == "read":
            required_scopes = self._jwt_required_read_scopes
        else:
            required_scopes = self._jwt_required_write_scopes

        if not required_scopes:
            return False
        if identity.scopes.intersection(required_scopes):
            return False
        return True

    def _resolve_permission_type(self, path: str, method: str) -> str:
        normalized_path = self._normalize_path(path)
        upper_method = (method or "").upper()

        if upper_method in {"GET", "HEAD", "OPTIONS"}:
            return "read"

        if upper_method == "POST":
            if normalized_path.startswith("/api/search"):
                return "read"
            if normalized_path.startswith("/api/agentic"):
                return "read"
            if normalized_path.startswith("/api/feedback/trends"):
                return "read"

        return "write"

    def _is_admin(self, roles: Set[str]) -> bool:
        return bool(self._jwt_admin_roles.intersection(roles))

    def _normalise_permission_values(self, raw_values) -> Set[str]:
        values: Set[str] = set()

        if raw_values is None:
            return values

        if isinstance(raw_values, str):
            parts = raw_values.replace(",", " ").split()
            for part in parts:
                normalized = str(part).strip().lower()
                if normalized:
                    values.add(normalized)
            return values

        if isinstance(raw_values, (list, tuple, set)):
            for item in raw_values:
                if isinstance(item, str):
                    normalized = item.strip().lower()
                else:
                    normalized = str(item).strip().lower()
                if normalized:
                    values.add(normalized)
            return values

        normalized = str(raw_values).strip().lower()
        if normalized:
            values.add(normalized)
        return values

    def _decode_base64url(self, value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)

    def _coerce_unix_timestamp(self, value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _audience_matches(self, audience_claim, expected_audience: str) -> bool:
        expected = expected_audience.strip()
        if not expected:
            return True

        if isinstance(audience_claim, str):
            return audience_claim.strip() == expected

        if isinstance(audience_claim, (list, tuple, set)):
            for candidate in audience_claim:
                if str(candidate).strip() == expected:
                    return True
            return False

        return False

    def _extract_token(self, request: Request) -> Optional[str]:
        token_candidates = []

        configured_header = request.headers.get(self._token_header)
        if configured_header:
            token_candidates.append(configured_header)

        auth_value = request.headers.get("authorization")
        if auth_value:
            token_candidates.append(auth_value)

        x_token = request.headers.get("x-access-token")
        if x_token:
            token_candidates.append(x_token)

        for value in token_candidates:
            candidate = value.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("bearer "):
                bearer_token = candidate[7:].strip()
                if bearer_token:
                    return bearer_token
                continue
            return candidate
        return None

    def _request_subject(self, request: Request) -> str:
        client_ip = None
        if self._trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
        if not client_ip and request.client:
            client_ip = request.client.host
        if not client_ip:
            client_ip = "unknown"
        return f"ip:{client_ip}"

    def _token_fingerprint(self, token: str) -> str:
        return sha256(token.encode("utf-8")).hexdigest()[:16]

    def _normalize_path(self, path: str) -> str:
        normalized = (path or "").strip()
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        if len(normalized) > 1 and normalized.endswith("/"):
            return normalized.rstrip("/")
        return normalized
