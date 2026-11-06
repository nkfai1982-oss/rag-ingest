"""Domain exceptions for the RAG ingestion and retrieval service."""


class RagIngestError(Exception):
    """Base exception for all service errors."""

    pass


class AuthenticationError(RagIngestError):
    """Invalid or missing team API key."""

    pass


class NotFoundError(RagIngestError):
    """Requested entity was not found."""

    pass


class DocumentDuplicateError(RagIngestError):
    """Identical file hash already indexed."""

    def __init__(self, document_id: str, message: str | None = None) -> None:
        super().__init__(message or f"Document already indexed with id: {document_id}")
        self.document_id = document_id


class UnsupportedFileTypeError(RagIngestError):
    """File extension not supported."""

    pass


class ExtractionError(RagIngestError):
    """Error during document text extraction."""

    pass


class EmbeddingError(RagIngestError):
    """Error calling embedding provider API."""

    pass


class VectorStoreError(RagIngestError):
    """Error during Qdrant operations."""

    pass
