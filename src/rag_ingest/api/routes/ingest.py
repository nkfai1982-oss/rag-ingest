"""Document ingestion and job tracking endpoints."""

import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from rag_ingest.api.dependencies import get_current_team, get_pipeline, get_repository
from rag_ingest.api.schemas import IngestResponse, JobResponse
from rag_ingest.domain.entities import IngestionJob, Team
from rag_ingest.domain.protocols import BaseMetadataRepository
from rag_ingest.pipeline.orchestrator import IngestionPipeline

router = APIRouter(prefix="/v1", tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Upload and ingest a document",
)
async def ingest_document(
    response: Response,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(...)],
    dataset_id: Annotated[str, Form(...)],
    async_mode: Annotated[bool, Form()] = False,
    overwrite: Annotated[bool, Form()] = False,
    chunk_size: Annotated[int | None, Form()] = None,
    chunk_overlap: Annotated[int | None, Form()] = None,
    team: Annotated[Team, Depends(get_current_team)] = None,  # type: ignore[assignment]
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)] = None,  # type: ignore[assignment]
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)] = None,  # type: ignore[assignment]
) -> IngestResponse:
    """Uploads and indexes a document with synchronous or asynchronous processing."""
    # Validate dataset exists
    dataset = repo.get_dataset(dataset_id, team.id)
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found for team '{team.id}'",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    file_name = file.filename or "uploaded_file"

    if async_mode:
        job_id = str(uuid.uuid4())
        job = IngestionJob(
            id=job_id,
            document_id=None,
            team_id=team.id,
            status="RUNNING",
            progress_percentage=0,
        )
        repo.create_job(job)

        def _run_bg() -> None:
            pipeline.run(
                file_content=file_bytes,
                file_name=file_name,
                team=team,
                dataset_id=dataset_id,
                overwrite=overwrite,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                job_id=job_id,
            )

        background_tasks.add_task(_run_bg)
        response.status_code = status.HTTP_202_ACCEPTED
        return IngestResponse(
            status="PROCESSING",
            document_id="pending",
            file_name=file_name,
            job_id=job_id,
        )

    # Synchronous execution
    run_status, document_id, total_chunks = pipeline.run(
        file_content=file_bytes,
        file_name=file_name,
        team=team,
        dataset_id=dataset_id,
        overwrite=overwrite,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    response.status_code = status.HTTP_200_OK
    return IngestResponse(
        status=run_status,
        document_id=document_id,
        file_name=file_name,
        total_chunks=total_chunks,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get status of an asynchronous ingestion job",
)
def get_job_status(
    job_id: str,
    team: Annotated[Team, Depends(get_current_team)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> JobResponse:
    job = repo.get_job(job_id, team.id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )
    return JobResponse(
        job_id=job.id,
        document_id=job.document_id,
        status=job.status,
        progress_percentage=job.progress_percentage,
        error_message=job.error_message,
    )
