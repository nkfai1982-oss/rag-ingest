"""Document management endpoints for listing, downloading, and deleting documents."""

import contextlib
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from rag_ingest.api.dependencies import (
    get_current_team,
    get_repository,
    get_settings,
    get_vector_store,
)
from rag_ingest.api.schemas import DocumentResponse
from rag_ingest.config import Settings
from rag_ingest.domain.entities import Team
from rag_ingest.domain.protocols import BaseMetadataRepository, BaseVectorStore

router = APIRouter(prefix="/v1/documents", tags=["Documents"])


@router.get(
    "",
    response_model=list[DocumentResponse],
    summary="List indexed documents for authenticated team",
)
def list_documents(
    team: Annotated[Team, Depends(get_current_team)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
    dataset_id: Annotated[str | None, Query(description="Filter by dataset ID")] = None,
) -> list[DocumentResponse]:
    docs = repo.list_documents(team_id=team.id, dataset_id=dataset_id)
    return [
        DocumentResponse(
            id=d.id,
            team_id=d.team_id,
            dataset_id=d.dataset_id,
            file_name=d.file_name,
            file_type=d.file_type,
            file_size_bytes=d.file_size_bytes,
            file_hash=d.file_hash,
            total_chunks=d.total_chunks,
            status=d.status,
            error_message=d.error_message,
            created_at=d.created_at,
            updated_at=d.updated_at,
        )
        for d in docs
    ]


@router.get(
    "/{document_id}/download",
    summary="Download the raw binary document file",
)
def download_document(
    document_id: str,
    team: Annotated[Team, Depends(get_current_team)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> FileResponse:
    doc = repo.get_document_by_id(document_id, team_id=team.id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )

    file_path = Path(doc.storage_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored file was not found on disk",
        )

    return FileResponse(
        path=str(file_path),
        filename=doc.file_name,
        media_type="application/octet-stream",
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document and purge its vectors",
)
def delete_document(
    document_id: str,
    team: Annotated[Team, Depends(get_current_team)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
    vector_store: Annotated[BaseVectorStore, Depends(get_vector_store)],
    settings_obj: Annotated[Settings, Depends(get_settings)],
) -> None:
    doc = repo.get_document_by_id(document_id, team_id=team.id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found",
        )

    # 1. Purge points from Qdrant
    vector_store.delete_by_document_id(settings_obj.QDRANT_COLLECTION, document_id)

    # 2. Delete raw file from disk if present
    file_path = Path(doc.storage_path)
    if file_path.exists():
        with contextlib.suppress(OSError):
            file_path.unlink()

    # 3. Delete SQLite record
    repo.delete_document(document_id, team.id)
