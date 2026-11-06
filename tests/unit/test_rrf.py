"""Unit tests for Application-Level Reciprocal Rank Fusion (RRF)."""

from unittest.mock import MagicMock

import pytest

from rag_ingest.domain.entities import SearchMode, SearchResult
from rag_ingest.services.search import SearchService


def test_rrf_scoring_and_ranking() -> None:
    vector_store = MagicMock()
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 1536

    # Create dummy hits
    # Chunk A: Rank 1 in dense, Rank 2 in text
    # Chunk B: Rank 2 in dense, absent in text
    # Chunk C: absent in dense, Rank 1 in text
    hit_a = SearchResult(
        chunk_id="c_A",
        content="Content A",
        score=0.9,
        document_id="d1",
        team_id="team1",
        file_name="f.md",
    )
    hit_b = SearchResult(
        chunk_id="c_B",
        content="Content B",
        score=0.8,
        document_id="d1",
        team_id="team1",
        file_name="f.md",
    )
    hit_c = SearchResult(
        chunk_id="c_C",
        content="Content C",
        score=0.7,
        document_id="d1",
        team_id="team1",
        file_name="f.md",
    )

    vector_store.search_vector.return_value = [hit_a, hit_b]
    vector_store.search_text.return_value = [hit_c, hit_a]

    service = SearchService(
        vector_store=vector_store,
        embedder=embedder,
        collection_name="test_kb",
        rrf_constant=60,
    )

    results = service.search(
        query="test query",
        team_id="team1",
        search_mode=SearchMode.HYBRID,
        top_k=5,
    )

    # Chunk A score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # Chunk C score: 1/(60+1) = 0.016393
    # Chunk B score: 1/(60+2) = 0.016129
    assert len(results) == 3
    assert results[0].chunk_id == "c_A"
    assert results[0].score == pytest.approx(0.032522, rel=1e-3)
    assert results[1].chunk_id == "c_C"
    assert results[2].chunk_id == "c_B"
