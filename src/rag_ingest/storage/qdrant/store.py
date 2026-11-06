"""Qdrant vector store adapter implementing BaseVectorStore."""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_ingest.domain.entities import DocumentChunk, SearchResult
from rag_ingest.domain.protocols import BaseVectorStore
from rag_ingest.exceptions import VectorStoreError


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store supporting dense vector search and full-text payload search."""

    def __init__(self, client: QdrantClient) -> None:
        self.client = client

    def init_collection(self, collection_name: str, vector_size: int) -> None:
        """Create Qdrant collection if not exists and create required payload indexes."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )

            # Create payload indexes
            # 1. page_content for full-text search
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="page_content",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.MULTILINGUAL,
                    lowercase=True,
                ),
            )
            # 2. team_id
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="team_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            # 3. dataset_id
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="dataset_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            # 4. document_id
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to initialize Qdrant collection {collection_name}: {e}"
            ) from e

    def upsert_points(self, collection_name: str, chunks: list[DocumentChunk]) -> None:
        """Batch upsert chunks with vector and payload."""
        if not chunks:
            return

        points: list[models.PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise VectorStoreError(f"Chunk {chunk.chunk_id} has no vector embedding.")

            payload: dict[str, Any] = {
                "page_content": chunk.text,
                "document_id": chunk.document_id,
                "team_id": chunk.team_id,
                "dataset_id": chunk.dataset_id,
                "chunk_index": chunk.chunk_index,
                "file_name": chunk.file_name,
                "metadata": chunk.metadata,
            }

            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload=payload,
                )
            )

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert points to Qdrant: {e}") from e

    def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        """Purge all chunks belonging to a document."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete points for document {document_id}: {e}"
            ) from e

    def search_vector(
        self,
        collection_name: str,
        query_vector: list[float],
        team_id: str,
        top_k: int,
        dataset_id: str | None = None,
    ) -> list[SearchResult]:
        """Dense similarity vector search filtered by team_id and optional dataset_id."""
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="team_id",
                match=models.MatchValue(value=team_id),
            )
        ]
        if dataset_id:
            conditions.append(
                models.FieldCondition(
                    key="dataset_id",
                    match=models.MatchValue(value=dataset_id),
                )
            )

        try:
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=models.Filter(must=conditions),
                limit=top_k,
                with_payload=True,
            )

            results: list[SearchResult] = []
            for hit in response.points:
                payload = hit.payload or {}
                results.append(
                    SearchResult(
                        chunk_id=str(hit.id),
                        content=str(payload.get("page_content", "")),
                        score=float(hit.score),
                        document_id=str(payload.get("document_id", "")),
                        team_id=str(payload.get("team_id", team_id)),
                        file_name=str(payload.get("file_name", "")),
                        metadata=dict(payload.get("metadata", {})),
                    )
                )
            return results
        except Exception as e:
            raise VectorStoreError(f"Error during vector search in Qdrant: {e}") from e

    def search_text(
        self,
        collection_name: str,
        query_text: str,
        team_id: str,
        top_k: int,
        dataset_id: str | None = None,
    ) -> list[SearchResult]:
        """Tokenized lexical full-text search on page_content payload."""
        conditions: list[models.Condition] = [
            models.FieldCondition(
                key="page_content",
                match=models.MatchText(text=query_text),
            ),
            models.FieldCondition(
                key="team_id",
                match=models.MatchValue(value=team_id),
            ),
        ]
        if dataset_id:
            conditions.append(
                models.FieldCondition(
                    key="dataset_id",
                    match=models.MatchValue(value=dataset_id),
                )
            )

        try:
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(must=conditions),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            results: list[SearchResult] = []
            for idx, record in enumerate(records):
                payload = record.payload or {}
                # Score assigned decreasing based on lexical retrieval rank
                rank_score = 1.0 / (1.0 + idx)
                results.append(
                    SearchResult(
                        chunk_id=str(record.id),
                        content=str(payload.get("page_content", "")),
                        score=rank_score,
                        document_id=str(payload.get("document_id", "")),
                        team_id=str(payload.get("team_id", team_id)),
                        file_name=str(payload.get("file_name", "")),
                        metadata=dict(payload.get("metadata", {})),
                    )
                )
            return results
        except Exception as e:
            raise VectorStoreError(f"Error during full-text search in Qdrant: {e}") from e
