"""
Configuration management for Contextprime system.
Handles loading and validation of configuration from YAML and environment variables.
"""

import os
import threading
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from loguru import logger


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="")
    database: str = Field(default="neo4j")
    max_connection_pool_size: int = Field(default=50)
    connection_timeout: int = Field(default=30)


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    api_key: Optional[str] = Field(default=None)
    collection_name: str = Field(default="doctags_vectors")
    vector_size: int = Field(default=1536)
    distance_metric: str = Field(default="cosine")


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""
    ocr_engine: str = Field(default="paddleocr")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    preserve_structure: bool = Field(default=True)
    max_file_size_mb: int = Field(default=100)
    supported_formats: list = Field(default_factory=lambda: ["pdf", "docx", "html", "txt", "png", "jpg"])


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=4000)
    api_key: Optional[str] = Field(default=None)


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""
    provider: str = Field(default="openai")
    model: str = Field(default="text-embedding-3-small")
    batch_size: int = Field(default=100)
    api_key: Optional[str] = Field(default=None)


class PathsConfig(BaseModel):
    """Filesystem paths used by the application."""

    models_dir: str = Field(default="models")


class APIConfig(BaseModel):
    """Public web server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: List[str] = Field(default_factory=list)
    rate_limit: int = Field(default=100)
    rate_limit_window_seconds: int = Field(default=60)
    rate_limit_redis_url: Optional[str] = Field(default=None)
    rate_limit_store_path: str = Field(default="data/storage/rate_limit.db")
    token_rate_limit: int = Field(default=0)
    token_rate_limit_window_seconds: int = Field(default=60)
    token_rate_limit_redis_url: Optional[str] = Field(default=None)
    token_rate_limit_store_path: str = Field(default="data/storage/token_rate_limit.db")
    token_unit_size: int = Field(default=64)
    trust_proxy_headers: bool = Field(default=False)


class SecurityConfig(BaseModel):
    """Access control settings for protected routes."""

    require_access_token: bool = Field(default=True)
    access_token: Optional[str] = Field(default=None)
    auth_mode: str = Field(default="jwt")
    token_header: str = Field(default="Authorization")
    jwt_secret: Optional[str] = Field(default=None)
    jwt_algorithm: str = Field(default="HS256")
    jwt_issuer: Optional[str] = Field(default=None)
    jwt_audience: Optional[str] = Field(default=None)
    jwt_subject_claim: str = Field(default="sub")
    jwt_roles_claim: str = Field(default="roles")
    jwt_scopes_claim: str = Field(default="scopes")
    jwt_enforce_permissions: bool = Field(default=True)
    jwt_require_expiry: bool = Field(default=True)
    jwt_required_read_scopes: List[str] = Field(default_factory=lambda: ["api:read"])
    jwt_required_write_scopes: List[str] = Field(default_factory=lambda: ["api:write"])
    jwt_admin_roles: List[str] = Field(default_factory=lambda: ["admin", "owner"])
    exempt_paths: List[str] = Field(
        default_factory=lambda: ["/api/health", "/api/readiness"]
    )


class StartupReadinessConfig(BaseModel):
    """Startup dependency readiness checks."""

    enabled: bool = Field(default=True)
    timeout_seconds: int = Field(default=60)
    check_interval_seconds: int = Field(default=2)
    required_services: List[str] = Field(default_factory=lambda: ["neo4j", "qdrant"])


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    model_config = ConfigDict(populate_by_name=True)
    hybrid_search: Dict[str, Any] = Field(default_factory=lambda: {
        "enable": True,
        "vector_weight": 0.7,
        "graph_weight": 0.3,
        "graph_vector_index": "chunk_embeddings",
        "graph_policy": {
            "mode": "standard",
            "local_seed_k": 8,
            "local_max_depth": 2,
            "local_neighbor_limit": 80,
            "global_scan_nodes": 1500,
            "global_max_terms": 8,
            "drift_local_weight": 0.65,
            "drift_global_weight": 0.35,
            "community_scan_nodes": 500,
            "community_max_terms": 8,
            "community_top_communities": 5,
            "community_members_per_community": 6,
            "community_vector_weight": 0.45,
            "community_summary_weight": 0.35,
            "community_member_weight": 0.20,
            "community_version": None,
        },
        "lexical": {
            "enable": True,
            "weight": 0.2,
            "max_scan_points": 1500,
            "scan_ratio": 0.02,
            "max_scan_cap": 20000,
            "page_size": 200,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
        },
        "corrective": {
            "enable": False,
            "min_results": 3,
            "min_average_confidence": 0.55,
            "top_k_multiplier": 2.0,
            "force_hybrid": True,
            "max_variants": 2,
            "max_initial_variants": 3,
        },
        "context_pruning": {
            "enable": False,
            "max_sentences_per_result": 4,
            "max_chars_per_result": 900,
            "min_sentence_tokens": 3,
            "context_selector": {
                "enable": False,
                "model_path": "models/context_selector.json",
                "min_score": 0.2,
                "min_results": 1,
            },
        },
        "cache": {
            "enable": True,
            "max_size": 128,
            "ttl_seconds": 600,
        },
        "request_budget": {
            "max_top_k": 12,
            "max_query_variants": 3,
            "max_corrective_variants": 2,
            "max_total_variant_searches": 5,
            "max_search_time_ms": 4500,
        },
    })
    max_results: int = Field(default=10)
    confidence_scoring: Dict[str, Any] = Field(default_factory=lambda: {
        "enable": True,
        "min_confidence": 0.1
    })
    rerank_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "enable": False,
        "model_name": "castorini/monot5-base-msmarco-10k",
        "top_n": 50,
    }, alias="rerank")


class LegalMetadataConfig(BaseModel):
    """Optional metadata for legal documents (version/amendment tracking)."""

    in_force_from: Optional[str] = Field(
        default=None,
        description="ISO-8601 date from which this version is in force (e.g. '2018-05-25')"
    )
    in_force_until: Optional[str] = Field(
        default=None,
        description="ISO-8601 date until which this version is in force, if superseded"
    )
    amended_by: Optional[List[str]] = Field(
        default=None,
        description="Doc IDs of instruments that amend this document"
    )
    supersedes: Optional[List[str]] = Field(
        default=None,
        description="Doc IDs of earlier documents that this document supersedes"
    )


class Settings(BaseSettings):
    """Main settings class that combines all configurations."""

    # Sub-configurations
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    startup_readiness: StartupReadinessConfig = Field(default_factory=StartupReadinessConfig)

    # System settings
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        extra = "allow"

    @classmethod
    def load_from_yaml(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from YAML file and environment variables."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        settings_dict = {}

        # Load from YAML if exists
        if config_path.exists():
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    settings_dict = yaml_config

        # Backward compatibility for legacy nested system settings.
        system_cfg = settings_dict.get("system") if isinstance(settings_dict, dict) else None
        if isinstance(system_cfg, dict):
            if "environment" in system_cfg and "environment" not in settings_dict:
                settings_dict["environment"] = system_cfg["environment"]
            if "log_level" in system_cfg and "log_level" not in settings_dict:
                settings_dict["log_level"] = system_cfg["log_level"]

        # Override with environment variables
        settings = cls(**settings_dict)

        # Override API keys from environment if present
        openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI__API_KEY")
        if openai_key:
            settings.llm.api_key = openai_key
            settings.embeddings.api_key = openai_key

        if os.getenv("ANTHROPIC_API_KEY"):
            if settings.llm.provider == "anthropic":
                settings.llm.api_key = os.getenv("ANTHROPIC_API_KEY")

        if os.getenv("NEO4J_PASSWORD"):
            settings.neo4j.password = os.getenv("NEO4J_PASSWORD")

        if os.getenv("QDRANT_API_KEY"):
            settings.qdrant.api_key = os.getenv("QDRANT_API_KEY")

        env_overrides = {
            "QDRANT_HOST": ("qdrant", "host"),
            "QDRANT__HOST": ("qdrant", "host"),
            "QDRANT_PORT": ("qdrant", "port"),
            "QDRANT__PORT": ("qdrant", "port"),
            "NEO4J_URI": ("neo4j", "uri"),
            "NEO4J__URI": ("neo4j", "uri"),
            "NEO4J_USERNAME": ("neo4j", "username"),
            "NEO4J__USERNAME": ("neo4j", "username"),
            "NEO4J_PASSWORD": ("neo4j", "password"),
            "NEO4J__PASSWORD": ("neo4j", "password"),
            "NEO4J_DATABASE": ("neo4j", "database"),
            "NEO4J__DATABASE": ("neo4j", "database"),
            "API__RATE_LIMIT": ("api", "rate_limit"),
            "API_RATE_LIMIT": ("api", "rate_limit"),
            "API__RATE_LIMIT_WINDOW_SECONDS": ("api", "rate_limit_window_seconds"),
            "API_RATE_LIMIT_WINDOW_SECONDS": ("api", "rate_limit_window_seconds"),
            "API__RATE_LIMIT_REDIS_URL": ("api", "rate_limit_redis_url"),
            "API_RATE_LIMIT_REDIS_URL": ("api", "rate_limit_redis_url"),
            "API__RATE_LIMIT_STORE_PATH": ("api", "rate_limit_store_path"),
            "API_RATE_LIMIT_STORE_PATH": ("api", "rate_limit_store_path"),
            "API__TOKEN_RATE_LIMIT": ("api", "token_rate_limit"),
            "API_TOKEN_RATE_LIMIT": ("api", "token_rate_limit"),
            "API__TOKEN_RATE_LIMIT_WINDOW_SECONDS": ("api", "token_rate_limit_window_seconds"),
            "API_TOKEN_RATE_LIMIT_WINDOW_SECONDS": ("api", "token_rate_limit_window_seconds"),
            "API__TOKEN_RATE_LIMIT_REDIS_URL": ("api", "token_rate_limit_redis_url"),
            "API_TOKEN_RATE_LIMIT_REDIS_URL": ("api", "token_rate_limit_redis_url"),
            "API__TOKEN_RATE_LIMIT_STORE_PATH": ("api", "token_rate_limit_store_path"),
            "API_TOKEN_RATE_LIMIT_STORE_PATH": ("api", "token_rate_limit_store_path"),
            "API__TOKEN_UNIT_SIZE": ("api", "token_unit_size"),
            "API_TOKEN_UNIT_SIZE": ("api", "token_unit_size"),
            "API__TRUST_PROXY_HEADERS": ("api", "trust_proxy_headers"),
            "API__CORS_ORIGINS": ("api", "cors_origins"),
            "SECURITY__REQUIRE_ACCESS_TOKEN": ("security", "require_access_token"),
            "SECURITY_REQUIRE_ACCESS_TOKEN": ("security", "require_access_token"),
            "SECURITY__ACCESS_TOKEN": ("security", "access_token"),
            "SECURITY_ACCESS_TOKEN": ("security", "access_token"),
            "SECURITY__AUTH_MODE": ("security", "auth_mode"),
            "SECURITY_AUTH_MODE": ("security", "auth_mode"),
            "SECURITY__TOKEN_HEADER": ("security", "token_header"),
            "SECURITY__JWT_SECRET": ("security", "jwt_secret"),
            "SECURITY_JWT_SECRET": ("security", "jwt_secret"),
            "SECURITY__JWT_ALGORITHM": ("security", "jwt_algorithm"),
            "SECURITY__JWT_ISSUER": ("security", "jwt_issuer"),
            "SECURITY__JWT_AUDIENCE": ("security", "jwt_audience"),
            "SECURITY__JWT_SUBJECT_CLAIM": ("security", "jwt_subject_claim"),
            "SECURITY__JWT_ROLES_CLAIM": ("security", "jwt_roles_claim"),
            "SECURITY__JWT_SCOPES_CLAIM": ("security", "jwt_scopes_claim"),
            "SECURITY__JWT_ENFORCE_PERMISSIONS": ("security", "jwt_enforce_permissions"),
            "SECURITY__JWT_REQUIRE_EXPIRY": ("security", "jwt_require_expiry"),
            "SECURITY__JWT_REQUIRED_READ_SCOPES": ("security", "jwt_required_read_scopes"),
            "SECURITY__JWT_REQUIRED_WRITE_SCOPES": ("security", "jwt_required_write_scopes"),
            "SECURITY__JWT_ADMIN_ROLES": ("security", "jwt_admin_roles"),
            "SECURITY__EXEMPT_PATHS": ("security", "exempt_paths"),
            "STARTUP_READINESS__ENABLED": ("startup_readiness", "enabled"),
            "STARTUP_READINESS__TIMEOUT_SECONDS": ("startup_readiness", "timeout_seconds"),
            "STARTUP_READINESS__CHECK_INTERVAL_SECONDS": ("startup_readiness", "check_interval_seconds"),
            "STARTUP_READINESS__REQUIRED_SERVICES": ("startup_readiness", "required_services"),
            "ENVIRONMENT": ("", "environment"),
            "SYSTEM__ENVIRONMENT": ("", "environment"),
            "LOG_LEVEL": ("", "log_level"),
            "SYSTEM__LOG_LEVEL": ("", "log_level"),
        }

        int_fields = {
            "port",
            "rate_limit",
            "rate_limit_window_seconds",
            "token_rate_limit",
            "token_rate_limit_window_seconds",
            "token_unit_size",
            "timeout_seconds",
            "check_interval_seconds",
        }
        bool_fields = {
            "require_access_token",
            "trust_proxy_headers",
            "enabled",
            "jwt_enforce_permissions",
            "jwt_require_expiry",
        }
        list_fields = {
            "cors_origins",
            "exempt_paths",
            "required_services",
            "jwt_required_read_scopes",
            "jwt_required_write_scopes",
            "jwt_admin_roles",
        }

        for env_name, (section, field) in env_overrides.items():
            value = os.getenv(env_name)
            if value is not None:
                target = settings if not section else getattr(settings, section)
                if field in int_fields:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"Invalid integer for {env_name}: {value}")
                        continue
                elif field in bool_fields:
                    lowered = str(value).strip().lower()
                    value = lowered in {"1", "true", "yes", "on"}
                elif field in list_fields:
                    value = [
                        item.strip()
                        for item in str(value).split(",")
                        if item and item.strip()
                    ]
                setattr(target, field, value)

        project_root = Path(__file__).resolve().parents[2]
        models_dir = Path(settings.paths.models_dir).expanduser()
        if not models_dir.is_absolute():
            models_dir = project_root / models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        settings.paths.models_dir = str(models_dir)

        # Setup logging
        logger.remove()
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=settings.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

        logger.info(f"Configuration loaded for environment: {settings.environment}")

        return settings

    def validate_runtime_security(self, strict: bool = False) -> List[str]:
        """Validate runtime security posture and optionally raise on hard failures."""
        issues: List[str] = []

        neo4j_password = (self.neo4j.password or "").strip()
        if not neo4j_password:
            issues.append("Neo4j password is missing")
        elif neo4j_password.lower() in {
            "password",
            "neo4j",
            "change_this_neo4j_password",
            "changeme",
        }:
            issues.append("Neo4j password uses a default value")

        if self.security.require_access_token:
            auth_mode = (self.security.auth_mode or "jwt").strip().lower()
            if auth_mode not in {"token", "jwt"}:
                issues.append("Auth mode must be token or jwt")
                auth_mode = "jwt"

            if auth_mode == "jwt":
                jwt_secret = (self.security.jwt_secret or "").strip()
                if not jwt_secret:
                    issues.append("JWT auth mode is enabled but SECURITY__JWT_SECRET is missing")
                elif len(jwt_secret) < 32:
                    issues.append("JWT secret is too short; use at least 32 characters")
                algorithm = (self.security.jwt_algorithm or "HS256").strip().upper()
                if algorithm not in {"HS256", "HS384", "HS512"}:
                    issues.append("JWT algorithm must be one of HS256, HS384, HS512")
                if self.security.jwt_enforce_permissions:
                    if not self.security.jwt_required_read_scopes:
                        issues.append("JWT read scopes are empty while permission enforcement is enabled")
                    if not self.security.jwt_required_write_scopes:
                        issues.append("JWT write scopes are empty while permission enforcement is enabled")
            else:
                token = (self.security.access_token or "").strip()
                if not token:
                    issues.append("Access token is required but not configured")
                elif len(token) < 24:
                    issues.append("Access token is too short; use at least 24 characters")

        cors_origins = [str(origin).strip() for origin in (self.api.cors_origins or []) if origin]
        if self.environment.lower() in {"docker", "production", "staging"} and "*" in cors_origins:
            issues.append("CORS origins cannot include wildcard in production-like environments")
        if int(getattr(self.api, "token_rate_limit", 0) or 0) <= 0:
            issues.append("Token rate limit is disabled; set API__TOKEN_RATE_LIMIT for cost control")
        if int(getattr(self.api, "token_unit_size", 0) or 0) <= 0:
            issues.append("Token unit size must be greater than zero")

        if strict and issues:
            raise ValueError("; ".join(issues))
        return issues

    def validate_connections(self) -> Dict[str, bool]:
        """Validate connections to external services."""
        results = {}

        # Validate Neo4j
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.neo4j.uri,
                auth=(self.neo4j.username, self.neo4j.password)
            )
            driver.verify_connectivity()
            driver.close()
            results["neo4j"] = True
            logger.success("Neo4j connection validated")
        except Exception as e:
            results["neo4j"] = False
            logger.warning(f"Neo4j connection failed: {e}")

        # Validate Qdrant
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                host=self.qdrant.host,
                port=self.qdrant.port,
                api_key=self.qdrant.api_key
            )
            client.get_collections()
            results["qdrant"] = True
            logger.success("Qdrant connection validated")
        except Exception as e:
            results["qdrant"] = False
            logger.warning(f"Qdrant connection failed: {e}")

        # Validate LLM API
        if self.llm.api_key:
            results["llm"] = True
            logger.success("LLM API key present")
        else:
            results["llm"] = False
            logger.warning("LLM API key not configured")

        return results


# Global settings instance
_settings: Optional[Settings] = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """Get or create the global settings instance (thread-safe)."""
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings.load_from_yaml()
    return _settings


def reset_settings() -> None:
    """Reset the cached settings singleton. Intended for testing only."""
    global _settings
    with _settings_lock:
        _settings = None
