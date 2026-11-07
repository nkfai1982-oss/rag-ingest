"""Team member registration and profile endpoints."""

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from rag_ingest.api.dependencies import get_current_member, get_repository
from rag_ingest.api.schemas import (
    ApiKeyResponse,
    MemberRegisterRequest,
    MemberRegisterResponse,
    MemberResponse,
)
from rag_ingest.domain.entities import Member, Team
from rag_ingest.domain.protocols import BaseMetadataRepository

router = APIRouter(prefix="/v1/members", tags=["Team Members"])


@router.post(
    "/register",
    response_model=MemberRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new member and specify their team",
)
def register_member(
    req: MemberRegisterRequest,
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> MemberRegisterResponse:
    """Registers a member identified by email to a team, issuing a personal API key."""
    # Check duplicate email
    existing_member = repo.get_member_by_email(req.email)
    if existing_member:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Member with email '{req.email}' is already registered",
        )

    # Check or auto-create team
    team = repo.get_team_by_id(req.team_id)
    if not team:
        team_name = req.team_name or req.team_id
        team = Team(id=req.team_id, name=team_name)
        repo.create_team(team)

    api_key = f"rag_m_{secrets.token_urlsafe(24)}"
    member = Member(
        email=req.email,
        team_id=team.id,
        api_key=api_key,
    )
    repo.create_member(member)

    return MemberRegisterResponse(
        email=member.email,
        team_id=member.team_id,
        api_key=member.api_key,
        created_at=member.created_at,
    )


@router.get(
    "/me",
    response_model=MemberResponse,
    summary="Get authenticated member profile",
)
def get_my_profile(
    member: Annotated[Member, Depends(get_current_member)],
) -> MemberResponse:
    """Returns the current member profile."""
    return MemberResponse(
        email=member.email,
        team_id=member.team_id,
        created_at=member.created_at,
    )


@router.post(
    "/me/api-key",
    response_model=ApiKeyResponse,
    summary="Regenerate member personal API key",
)
def regenerate_my_api_key(
    member: Annotated[Member, Depends(get_current_member)],
    repo: Annotated[BaseMetadataRepository, Depends(get_repository)],
) -> ApiKeyResponse:
    """Rolls a new API key for the authenticated member."""
    new_api_key = f"rag_m_{secrets.token_urlsafe(24)}"
    updated = repo.update_member_api_key(member.email, new_api_key)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update member API key",
        )
    return ApiKeyResponse(api_key=new_api_key)

