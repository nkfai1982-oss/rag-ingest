"""FastAPI dependency injection providers."""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from qdrant_client import QdrantClient

from rag_ingest.config import Settings, settings
from rag_ingest.domain.entities import Member, Team
from rag_ingest.domain.protocols import (
    BaseEmbedder,
    BaseMetadataRepository,
    BaseVectorStore,
)
from rag_ingest.pipeline.orchestrator import IngestionPipeline
from rag_ingest.services.embedder import EmbedderClient
from rag_ingest.services.search import SearchService
from rag_ingest.storage.qdrant.store import QdrantVectorStore
from rag_ingest.storage.sqlite.connection import SQLiteConnection
from rag_ingest.storage.sqlite.repository import SQLiteMetadataRepository

# Singletons / Process-level instances
_db_connection = SQLiteConnection(settings.SQLITE_PATH)
_repository = SQLiteMetadataRepository(_db_connection)

_qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)
_vector_store = QdrantVectorStore(_qdrant_client)

_embedder = EmbedderClient(
    base_url=settings.EMBEDDING_BASE_URL,
    api_key=settings.EMBEDDING_API_KEY,
    model=settings.EMBEDDING_MODEL,
    batch_size=settings.EMBEDDING_BATCH_SIZE,
)

_search_service = SearchService(
    vector_store=_vector_store,
    embedder=_embedder,
    collection_name=settings.QDRANT_COLLECTION,
)

_pipeline = IngestionPipeline(
    repository=_repository,
    vector_store=_vector_store,
    embedder=_embedder,
    storage_dir=settings.STORAGE_DIR,
    collection_name=settings.QDRANT_COLLECTION,
)


def get_settings() -> Settings:
    return settings


def get_repository() -> BaseMetadataRepository:
    return _repository


def get_vector_store() -> BaseVectorStore:
    return _vector_store


def get_embedder() -> BaseEmbedder:
    return _embedder


def get_search_service() -> SearchService:
    return _search_service


def get_pipeline() -> IngestionPipeline:
    return _pipeline


def get_current_member(
    authorization: Annotated[str | None, Header()] = None,
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)] = None,  # type: ignore[assignment]
) -> Member:
    """Authenticates team member using Bearer token from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme. Use 'Bearer <api_key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    member = repo.get_member_by_api_key(token)
    if not member:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid member API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return member


def get_current_team(
    member: Annotated[Member, Depends(get_current_member)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> Team:
    """Resolves the team belonging to the authenticated member."""
    team = repo.get_team_by_id(member.team_id)
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Team '{member.team_id}' not found for member '{member.email}'",
        )
    return team
