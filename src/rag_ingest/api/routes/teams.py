"""Team and dataset management endpoints."""

import secrets
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from rag_ingest.api.dependencies import get_current_member, get_current_team, get_repository
from rag_ingest.api.schemas import (
    DatasetCreateRequest,
    DatasetResponse,
    MemberRegisterResponse,
    MemberResponse,
    TeamCreateRequest,
    TeamResponse,
)
from rag_ingest.domain.entities import Dataset, Member, Team
from rag_ingest.domain.protocols import BaseMetadataRepository

router = APIRouter(prefix="/v1", tags=["Tenants & Datasets"])


@router.post(
    "/teams",
    response_model=TeamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new tenant team container",
)
def create_team(
    req: TeamCreateRequest,
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> TeamResponse:
    team_id = req.id or f"team_{secrets.token_hex(4)}"

    team = Team(
        id=team_id,
        name=req.name,
        default_chunk_size=req.default_chunk_size,
        default_chunk_overlap=req.default_chunk_overlap,
    )
    repo.create_team(team)
    return TeamResponse(
        id=team.id,
        name=team.name,
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
        default_chunk_size=team.default_chunk_size,
        default_chunk_overlap=team.default_chunk_overlap,
        created_at=team.created_at,
    )


@router.get(
    "/teams/{team_id}/members",
    response_model=list[MemberResponse],
    summary="List all members in the team",
)
def list_team_members(
    team_id: str,
    current_team: Annotated[Team, Depends(get_current_team)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> list[MemberResponse]:
    if current_team.id != team_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access members of another team",
        )
    members = repo.list_members_by_team(team_id)
    return [
        MemberResponse(
            email=m.email,
            team_id=m.team_id,
            created_at=m.created_at,
        )
        for m in members
    ]


@router.post(
    "/teams/{team_id}/members",
    response_model=MemberRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new member to the team",
)
def add_team_member(
    team_id: str,
    email: str,
    current_team: Annotated[Team, Depends(get_current_team)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> MemberRegisterResponse:
    if current_team.id != team_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot add members to another team",
        )
    existing = repo.get_member_by_email(email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Member with email '{email}' already exists",
        )

    api_key = f"rag_m_{secrets.token_urlsafe(24)}"
    member = Member(
        email=email,
        team_id=team_id,
        api_key=api_key,
    )
    repo.create_member(member)
    return MemberRegisterResponse(
        email=member.email,
        team_id=member.team_id,
        api_key=member.api_key,
        created_at=member.created_at,
    )


@router.delete(
    "/teams/{team_id}/members/{email}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a member from the team",
)
def remove_team_member(
    team_id: str,
    email: str,
    current_team: Annotated[Team, Depends(get_current_team)],
    current_member: Annotated[Member, Depends(get_current_member)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> None:
    if current_team.id != team_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot remove members from another team",
        )
    deleted = repo.delete_member(email, team_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Member with email '{email}' not found in team '{team_id}'",
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
