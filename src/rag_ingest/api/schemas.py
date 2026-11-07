"""Pydantic API request and response contracts."""

from pydantic import BaseModel, Field

from rag_ingest.domain.entities import SearchMode, SearchResult


class IngestResponse(BaseModel):
    """Response returned upon document ingestion."""

    status: str
    document_id: str
    file_name: str | None = None
    total_chunks: int | None = None
    job_id: str | None = None


class JobResponse(BaseModel):
    """Response representing the status of an asynchronous ingestion job."""

    job_id: str
    document_id: str | None = None
    status: str
    progress_percentage: int
    error_message: str | None = None


class RetrievalRequest(BaseModel):
    """Request payload for multi-mode search."""

    query: str = Field(..., min_length=1, description="Search query string")
    search_mode: SearchMode = Field(default=SearchMode.HYBRID)
    dataset_id: str | None = Field(default=None, description="Optional dataset scope filter")
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0)


class RetrievalResponse(BaseModel):
    """Response payload containing ranked search hits."""

    query: str
    search_mode: SearchMode
    results: list[SearchResult]


class DocumentResponse(BaseModel):
    """Document metadata model for listing."""

    id: str
    team_id: str
    dataset_id: str
    file_name: str
    file_type: str
    file_size_bytes: int
    file_hash: str
    total_chunks: int
    status: str
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class MemberRegisterRequest(BaseModel):
    """Payload to register a new member and specify their team."""

    email: str = Field(..., min_length=3, description="Member unique ID (email)")
    team_id: str = Field(..., min_length=1, description="Target team identifier")
    team_name: str | None = Field(
        default=None,
        description="Optional team name if creating a new team",
    )


class MemberResponse(BaseModel):
    """Public team member information."""

    email: str
    team_id: str
    created_at: str | None = None


class MemberRegisterResponse(BaseModel):
    """Response returned upon successful member registration including API key."""

    email: str
    team_id: str
    api_key: str
    created_at: str | None = None


class ApiKeyResponse(BaseModel):
    """Response returned when an API key is regenerated."""

    api_key: str


class TeamCreateRequest(BaseModel):
    """Payload to register a new tenant team."""

    name: str = Field(..., min_length=1)
    id: str | None = None
    default_chunk_size: int = 800
    default_chunk_overlap: int = 100


class TeamResponse(BaseModel):
    """Tenant team information."""

    id: str
    name: str
    default_chunk_size: int
    default_chunk_overlap: int
    created_at: str | None = None


class DatasetCreateRequest(BaseModel):
    """Payload to create a knowledge base dataset."""

    name: str = Field(..., min_length=1)
    id: str | None = None
    description: str | None = None


class DatasetResponse(BaseModel):
    """Dataset information."""

    id: str
    team_id: str
    name: str
    description: str | None = None
    created_at: str | None = None
