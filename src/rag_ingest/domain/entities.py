"""Domain entities and value objects for RAG pipeline."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SearchMode(StrEnum):
    """Supported search retrieval modes."""

    SEMANTIC = "semantic"
    FULL_TEXT = "full_text"
    HYBRID = "hybrid"


class ExtractedChunk(BaseModel):
    """Raw chunk extracted by document extractors."""

    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Processed text chunk with embeddings and tenant context."""

    chunk_id: str
    document_id: str
    team_id: str
    dataset_id: str
    chunk_index: int
    text: str
    embedding: list[float] | None = None
    file_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Standardized search hit returned to clients."""

    chunk_id: str
    content: str
    score: float
    document_id: str
    team_id: str
    file_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Team(BaseModel):
    """Tenant/Team account entity."""

    id: str
    name: str
    api_key: str
    default_chunk_size: int = 800
    default_chunk_overlap: int = 100
    created_at: str | None = None


class Dataset(BaseModel):
    """Knowledge base dataset container."""

    id: str
    team_id: str
    name: str
    description: str | None = None
    created_at: str | None = None


class Document(BaseModel):
    """Document record stored in metadata registry."""

    id: str
    team_id: str
    dataset_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    file_hash: str
    storage_path: str
    total_chunks: int = 0
    status: str = "PENDING"
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class IngestionJob(BaseModel):
    """Asynchronous ingestion job state."""

    id: str
    document_id: str | None = None
    team_id: str
    status: str = "RUNNING"
    progress_percentage: int = 0
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
