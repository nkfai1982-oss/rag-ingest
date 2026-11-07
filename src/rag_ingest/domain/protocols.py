"""Domain protocols and abstract interfaces enforcing Dependency Inversion."""

from abc import ABC, abstractmethod

from rag_ingest.domain.entities import (
    Dataset,
    Document,
    DocumentChunk,
    ExtractedChunk,
    IngestionJob,
    Member,
    SearchResult,
    Team,
)


class BaseExtractor(ABC):
    """Abstract interface for file content extractors."""

    @abstractmethod
    def extract(self, file_path: str) -> list[ExtractedChunk]:
        """Extract text chunks and metadata from file."""
        pass


class BaseVectorStore(ABC):
    """Abstract interface for vector database storage adapters."""

    @abstractmethod
    def init_collection(self, collection_name: str, vector_size: int) -> None:
        """Initialize collection and payload indexes."""
        pass

    @abstractmethod
    def upsert_points(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        """Batch upsert points into vector collection."""
        pass

    @abstractmethod
    def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        """Purge all chunks belonging to a document."""
        pass

    @abstractmethod
    def search_vector(
        self,
        collection_name: str,
        query_vector: list[float],
        team_id: str,
        top_k: int,
        dataset_id: str | None = None,
    ) -> list[SearchResult]:
        """Dense similarity vector search."""
        pass

    @abstractmethod
    def search_text(
        self,
        collection_name: str,
        query_text: str,
        team_id: str,
        top_k: int,
        dataset_id: str | None = None,
    ) -> list[SearchResult]:
        """Tokenized lexical full-text search."""
        pass


class BaseMetadataRepository(ABC):
    """Abstract interface for metadata persistence (SQLite/PostgreSQL)."""

    @abstractmethod
    def init_db(self) -> None:
        """Create tables and indexes."""
        pass

    # Teams
    @abstractmethod
    def create_team(self, team: Team) -> Team:
        """Create a new team record."""
        pass

    @abstractmethod
    def get_team_by_id(self, team_id: str) -> Team | None:
        """Retrieve team by ID."""
        pass

    @abstractmethod
    def list_teams(self) -> list[Team]:
        """List all registered teams."""
        pass

    # Members
    @abstractmethod
    def create_member(self, member: Member) -> Member:
        """Register a new team member."""
        pass

    @abstractmethod
    def get_member_by_email(self, email: str) -> Member | None:
        """Retrieve a member by email (member ID)."""
        pass

    @abstractmethod
    def get_member_by_api_key(self, api_key: str) -> Member | None:
        """Retrieve a member by API key."""
        pass

    @abstractmethod
    def list_members_by_team(self, team_id: str) -> list[Member]:
        """List all members belonging to a team."""
        pass

    @abstractmethod
    def update_member_api_key(self, email: str, new_api_key: str) -> Member | None:
        """Regenerate/update a member's API key."""
        pass

    @abstractmethod
    def delete_member(self, email: str, team_id: str) -> bool:
        """Remove a member from a team."""
        pass

    # Datasets
    @abstractmethod
    def create_dataset(self, dataset: Dataset) -> Dataset:
        """Create a new dataset."""
        pass

    @abstractmethod
    def get_dataset(self, dataset_id: str, team_id: str) -> Dataset | None:
        """Get dataset by ID and team."""
        pass

    @abstractmethod
    def list_datasets(self, team_id: str) -> list[Dataset]:
        """List datasets for a team."""
        pass

    # Documents
    @abstractmethod
    def create_document(self, document: Document) -> Document:
        """Register a new document."""
        pass

    @abstractmethod
    def get_document_by_id(self, document_id: str, team_id: str | None = None) -> Document | None:
        """Retrieve document by ID."""
        pass

    @abstractmethod
    def get_document_by_hash(self, file_hash: str, team_id: str) -> Document | None:
        """Retrieve document by SHA-256 hash for tenant deduplication."""
        pass

    @abstractmethod
    def get_document_by_name(
        self, file_name: str, dataset_id: str, team_id: str
    ) -> Document | None:
        """Retrieve document by file name within a dataset."""
        pass

    @abstractmethod
    def update_document_status(
        self,
        document_id: str,
        status: str,
        total_chunks: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update document processing state."""
        pass

    @abstractmethod
    def list_documents(self, team_id: str, dataset_id: str | None = None) -> list[Document]:
        """List documents for a team with optional dataset filter."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str, team_id: str) -> bool:
        """Delete document record."""
        pass

    # Ingestion Jobs
    @abstractmethod
    def create_job(self, job: IngestionJob) -> IngestionJob:
        """Create a background ingestion job."""
        pass

    @abstractmethod
    def get_job(self, job_id: str, team_id: str | None = None) -> IngestionJob | None:
        """Retrieve job state."""
        pass

    @abstractmethod
    def update_job(
        self,
        job_id: str,
        status: str,
        progress_percentage: int | None = None,
        error_message: str | None = None,
        document_id: str | None = None,
    ) -> None:
        """Update job state and progress."""
        pass


class BaseEmbedder(ABC):
    """Abstract interface for generating vector embeddings."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single search query."""
        pass
