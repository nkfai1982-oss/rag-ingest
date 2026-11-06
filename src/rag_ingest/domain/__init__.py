"""Domain layer containing core entities and protocols."""

from rag_ingest.domain.entities import (
    Dataset,
    Document,
    DocumentChunk,
    ExtractedChunk,
    IngestionJob,
    SearchMode,
    SearchResult,
    Team,
)
from rag_ingest.domain.protocols import (
    BaseEmbedder,
    BaseExtractor,
    BaseMetadataRepository,
    BaseVectorStore,
)

__all__ = [
    "Dataset",
    "Document",
    "DocumentChunk",
    "ExtractedChunk",
    "IngestionJob",
    "SearchMode",
    "SearchResult",
    "Team",
    "BaseEmbedder",
    "BaseExtractor",
    "BaseMetadataRepository",
    "BaseVectorStore",
]
