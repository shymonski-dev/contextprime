"""
Configuration management for DocTags RAG system.
Handles loading and validation of configuration from YAML and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from loguru import logger


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""
    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="password")
    database: str = Field(default="doctags")
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


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    hybrid_search: Dict[str, Any] = Field(default_factory=lambda: {
        "enable": True,
        "vector_weight": 0.7,
        "graph_weight": 0.3,
        "graph_vector_index": "chunk_embeddings",
    })
    max_results: int = Field(default=10)
    rerank: bool = Field(default=True)
    confidence_scoring: Dict[str, Any] = Field(default_factory=lambda: {
        "enable": True,
        "min_confidence": 0.1
    })


class Settings(BaseSettings):
    """Main settings class that combines all configurations."""

    # Sub-configurations
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

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

        # Override with environment variables
        settings = cls(**settings_dict)

        # Override API keys from environment if present
        if os.getenv("OPENAI_API_KEY"):
            settings.llm.api_key = os.getenv("OPENAI_API_KEY")
            settings.embeddings.api_key = os.getenv("OPENAI_API_KEY")

        if os.getenv("ANTHROPIC_API_KEY"):
            if settings.llm.provider == "anthropic":
                settings.llm.api_key = os.getenv("ANTHROPIC_API_KEY")

        if os.getenv("NEO4J_PASSWORD"):
            settings.neo4j.password = os.getenv("NEO4J_PASSWORD")

        if os.getenv("QDRANT_API_KEY"):
            settings.qdrant.api_key = os.getenv("QDRANT_API_KEY")

        # Setup logging
        logger.remove()
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=settings.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

        logger.info(f"Configuration loaded for environment: {settings.environment}")

        return settings

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
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global settings
    if settings is None:
        settings = Settings.load_from_yaml()
    return settings
