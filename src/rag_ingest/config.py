"""Configuration settings for RAG Ingestion and Retrieval Service."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service
    SERVICE_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # SQLite
    SQLITE_PATH: str = "data/knowledge.db"
    STORAGE_DIR: str = "data/uploads"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "knowledge_base"

    # Embeddings (OpenAI compatible)
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_API_KEY: str = Field(
        default="mock-key",
        description="API key for embedding service",
    )
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIM: int = 1536
    EMBEDDING_BATCH_SIZE: int = 64

    # Chunking Defaults
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 100

    @property
    def sqlite_db_path(self) -> Path:
        """Resolved path to SQLite database."""
        return Path(self.SQLITE_PATH).resolve()

    @property
    def uploads_dir_path(self) -> Path:
        """Resolved path to uploaded files directory."""
        return Path(self.STORAGE_DIR).resolve()


settings = Settings()
