"""Retrieval endpoints supporting Semantic, Full-Text, and Hybrid RRF."""

from typing import Annotated

from fastapi import APIRouter, Depends

from rag_ingest.api.dependencies import get_current_team, get_search_service
from rag_ingest.api.schemas import RetrievalRequest, RetrievalResponse
from rag_ingest.domain.entities import Team
from rag_ingest.services.search import SearchService

router = APIRouter(prefix="/v1", tags=["Retrieval"])


@router.post(
    "/retrieval",
    response_model=RetrievalResponse,
    summary="Query knowledge base using Semantic, Full-Text, or Hybrid search",
)
def retrieve_documents(
    req: RetrievalRequest,
    team: Annotated[Team, Depends(get_current_team)],
    search_service: Annotated[SearchService, Depends(get_search_service)],
) -> RetrievalResponse:
    """Performs retrieval over the tenant's indexed knowledge base."""
    hits = search_service.search(
        query=req.query,
        team_id=team.id,
        search_mode=req.search_mode,
        top_k=req.top_k,
        dataset_id=req.dataset_id,
        score_threshold=req.score_threshold,
    )
    return RetrievalResponse(
        query=req.query,
        search_mode=req.search_mode,
        results=hits,
    )
