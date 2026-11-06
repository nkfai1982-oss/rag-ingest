"""Search service supporting Semantic, Full-Text, and Application-Level Hybrid RRF."""

from rag_ingest.config import settings
from rag_ingest.domain.entities import SearchMode, SearchResult
from rag_ingest.domain.protocols import BaseEmbedder, BaseVectorStore


class SearchService:
    """Orchestrates multi-mode search and Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        collection_name: str | None = None,
        rrf_constant: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.rrf_constant = rrf_constant

    def search(
        self,
        query: str,
        team_id: str,
        search_mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 5,
        dataset_id: str | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Execute search query in the specified mode."""
        if search_mode == SearchMode.SEMANTIC:
            return self._search_semantic(query, team_id, top_k, dataset_id, score_threshold)
        elif search_mode == SearchMode.FULL_TEXT:
            return self._search_full_text(query, team_id, top_k, dataset_id, score_threshold)
        elif search_mode == SearchMode.HYBRID:
            return self._search_hybrid_rrf(query, team_id, top_k, dataset_id, score_threshold)
        else:
            raise ValueError(f"Unsupported search mode: {search_mode}")

    def _search_semantic(
        self,
        query: str,
        team_id: str,
        top_k: int,
        dataset_id: str | None,
        score_threshold: float,
    ) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        hits = self.vector_store.search_vector(
            collection_name=self.collection_name,
            query_vector=query_vector,
            team_id=team_id,
            top_k=top_k,
            dataset_id=dataset_id,
        )
        return [h for h in hits if h.score >= score_threshold]

    def _search_full_text(
        self,
        query: str,
        team_id: str,
        top_k: int,
        dataset_id: str | None,
        score_threshold: float,
    ) -> list[SearchResult]:
        hits = self.vector_store.search_text(
            collection_name=self.collection_name,
            query_text=query,
            team_id=team_id,
            top_k=top_k,
            dataset_id=dataset_id,
        )
        return [h for h in hits if h.score >= score_threshold]

    def _search_hybrid_rrf(
        self,
        query: str,
        team_id: str,
        top_k: int,
        dataset_id: str | None,
        score_threshold: float,
    ) -> list[SearchResult]:
        """Fuses dense vector and lexical text search results using Reciprocal Rank Fusion."""
        candidate_k = max(top_k * 2, 10)

        # 1. Dense branch
        query_vector = self.embedder.embed_query(query)
        dense_hits = self.vector_store.search_vector(
            collection_name=self.collection_name,
            query_vector=query_vector,
            team_id=team_id,
            top_k=candidate_k,
            dataset_id=dataset_id,
        )

        # 2. Lexical branch
        text_hits = self.vector_store.search_text(
            collection_name=self.collection_name,
            query_text=query,
            team_id=team_id,
            top_k=candidate_k,
            dataset_id=dataset_id,
        )

        # 3. Application-Level RRF Fusion: score = sum(1 / (k + rank))
        # Keep map of chunk_id -> SearchResult template
        chunk_map: dict[str, SearchResult] = {}
        rrf_scores: dict[str, float] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            chunk_map[hit.chunk_id] = hit
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (
                1.0 / (self.rrf_constant + rank)
            )

        for rank, hit in enumerate(text_hits, start=1):
            if hit.chunk_id not in chunk_map:
                chunk_map[hit.chunk_id] = hit
            rrf_scores[hit.chunk_id] = rrf_scores.get(hit.chunk_id, 0.0) + (
                1.0 / (self.rrf_constant + rank)
            )

        # 4. Sort descending by combined RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        fused_results: list[SearchResult] = []
        for cid in sorted_chunk_ids:
            score = rrf_scores[cid]
            if score < score_threshold:
                continue

            orig_hit = chunk_map[cid]
            fused_results.append(
                SearchResult(
                    chunk_id=orig_hit.chunk_id,
                    content=orig_hit.content,
                    score=round(score, 6),
                    document_id=orig_hit.document_id,
                    team_id=orig_hit.team_id,
                    file_name=orig_hit.file_name,
                    metadata=orig_hit.metadata,
                )
            )
            if len(fused_results) >= top_k:
                break

        return fused_results
