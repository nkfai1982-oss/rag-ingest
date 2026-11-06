"""Team and dataset management endpoints."""

import secrets
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from rag_ingest.api.dependencies import get_current_team, get_repository
from rag_ingest.api.schemas import (
    DatasetCreateRequest,
    DatasetResponse,
    TeamCreateRequest,
    TeamResponse,
)
from rag_ingest.domain.entities import Dataset, Team
from rag_ingest.domain.protocols import BaseMetadataRepository

router = APIRouter(prefix="/v1", tags=["Tenants & Datasets"])


@router.post(
    "/teams",
    response_model=TeamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new tenant team and generate an API key",
)
def create_team(
    req: TeamCreateRequest,
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> TeamResponse:
    team_id = req.id or f"team_{secrets.token_hex(4)}"
    api_key = req.api_key or f"rag_{secrets.token_urlsafe(24)}"

    team = Team(
        id=team_id,
        name=req.name,
        api_key=api_key,
        default_chunk_size=req.default_chunk_size,
        default_chunk_overlap=req.default_chunk_overlap,
    )
    repo.create_team(team)
    return TeamResponse(
        id=team.id,
        name=team.name,
        api_key=team.api_key,
        default_chunk_size=team.default_chunk_size,
        default_chunk_overlap=team.default_chunk_overlap,
        created_at=team.created_at,
    )


@router.get(
    "/teams/me",
    response_model=TeamResponse,
    summary="Get authenticated team profile",
)
def get_my_team(
    team: Annotated[Team, Depends(get_current_team)],
) -> TeamResponse:
    return TeamResponse(
        id=team.id,
        name=team.name,
        api_key=team.api_key,
        default_chunk_size=team.default_chunk_size,
        default_chunk_overlap=team.default_chunk_overlap,
        created_at=team.created_at,
    )


@router.post(
    "/datasets",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a knowledge dataset for the authenticated team",
)
def create_dataset(
    req: DatasetCreateRequest,
    team: Annotated[Team, Depends(get_current_team)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> DatasetResponse:
    dataset_id = req.id or f"ds_{uuid.uuid4().hex[:8]}"

    existing = repo.get_dataset(dataset_id, team.id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset '{dataset_id}' already exists for team '{team.id}'",
        )

    ds = Dataset(
        id=dataset_id,
        team_id=team.id,
        name=req.name,
        description=req.description,
    )
    repo.create_dataset(ds)
    return DatasetResponse(
        id=ds.id,
        team_id=ds.team_id,
        name=ds.name,
        description=ds.description,
        created_at=ds.created_at,
    )


@router.get(
    "/datasets",
    response_model=list[DatasetResponse],
    summary="List all datasets for the authenticated team",
)
def list_datasets(
    team: Annotated[Team, Depends(get_current_team)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> list[DatasetResponse]:
    datasets = repo.list_datasets(team.id)
    return [
        DatasetResponse(
            id=ds.id,
            team_id=ds.team_id,
            name=ds.name,
            description=ds.description,
            created_at=ds.created_at,
        )
        for ds in datasets
    ]
