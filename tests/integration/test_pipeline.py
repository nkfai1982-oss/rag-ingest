"""Integration tests for IngestionPipeline (sync, dedup, overwrite, compensating transaction)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_ingest.exceptions import DocumentDuplicateError, RagIngestError
from rag_ingest.pipeline.orchestrator import IngestionPipeline
from rag_ingest.storage.qdrant.store import QdrantVectorStore
from rag_ingest.storage.sqlite.repository import SQLiteMetadataRepository


def test_pipeline_multi_format_ingestion(
    test_pipeline: IngestionPipeline,
    sqlite_repo: SQLiteMetadataRepository,
    sample_files: dict[str, Path],
) -> None:
    team = sqlite_repo.get_team_by_id("test_team_1")
    assert team is not None

    for _ext, file_path in sample_files.items():
        status, doc_id, chunk_count = test_pipeline.run(
            file_content=file_path.read_bytes(),
            file_name=file_path.name,
            team=team,
            dataset_id="test_dataset_1",
        )
        assert status == "SUCCESS"
        assert chunk_count >= 1

        # Check DB record
        doc = sqlite_repo.get_document_by_id(doc_id, team.id)
        assert doc is not None
        assert doc.status == "COMPLETED"
        assert doc.total_chunks == chunk_count


def test_pipeline_deduplication(
    test_pipeline: IngestionPipeline,
    sqlite_repo: SQLiteMetadataRepository,
    sample_files: dict[str, Path],
) -> None:
    team = sqlite_repo.get_team_by_id("test_team_1")
    assert team is not None
    content = sample_files["md"].read_bytes()

    # First upload
    status1, doc_id1, chunks1 = test_pipeline.run(
        file_content=content,
        file_name="first_run.md",
        team=team,
        dataset_id="test_dataset_1",
    )
    assert status1 == "SUCCESS"

    # Second upload of identical content under different file name
    status2, doc_id2, chunks2 = test_pipeline.run(
        file_content=content,
        file_name="duplicate.md",
        team=team,
        dataset_id="test_dataset_1",
    )
    assert status2 == "SKIPPED_DUPLICATE"
    assert doc_id2 == doc_id1
    assert chunks2 == chunks1


def test_pipeline_controlled_overwrite(
    test_pipeline: IngestionPipeline,
    sqlite_repo: SQLiteMetadataRepository,
) -> None:
    team = sqlite_repo.get_team_by_id("test_team_1")
    assert team is not None

    v1_content = b"# Version 1 content"
    v2_content = b"# Version 2 completely updated content"

    # Ingest version 1
    test_pipeline.run(
        file_content=v1_content,
        file_name="versioned.md",
        team=team,
        dataset_id="test_dataset_1",
    )

    # Ingest version 2 with overwrite=False -> should raise DocumentDuplicateError
    with pytest.raises(DocumentDuplicateError):
        test_pipeline.run(
            file_content=v2_content,
            file_name="versioned.md",
            team=team,
            dataset_id="test_dataset_1",
            overwrite=False,
        )

    # Ingest version 2 with overwrite=True -> succeeds
    status, doc_id2, chunks2 = test_pipeline.run(
        file_content=v2_content,
        file_name="versioned.md",
        team=team,
        dataset_id="test_dataset_1",
        overwrite=True,
    )
    assert status == "SUCCESS"
    doc2 = sqlite_repo.get_document_by_id(doc_id2, team.id)
    assert doc2 is not None
    assert doc2.status == "COMPLETED"


def test_pipeline_compensating_transaction(
    sqlite_repo: SQLiteMetadataRepository,
    in_memory_qdrant: QdrantVectorStore,
    sample_files: dict[str, Path],
    tmp_path: Path,
) -> None:
    team = sqlite_repo.get_team_by_id("test_team_1")
    assert team is not None

    mock_repo = MagicMock(wraps=sqlite_repo)
    # Simulate database crash right after vector store upsert
    mock_repo.update_document_status.side_effect = RuntimeError("Database disk failure")

    pipeline = IngestionPipeline(
        repository=mock_repo,
        vector_store=in_memory_qdrant,
        embedder=MagicMock(embed_texts=lambda texts: [[0.1] * 1536 for _ in texts]),
        storage_dir=tmp_path / "uploads",
        collection_name="knowledge_base",
    )

    with pytest.raises(RagIngestError):
        pipeline.run(
            file_content=sample_files["md"].read_bytes(),
            file_name="fail_test.md",
            team=team,
            dataset_id="test_dataset_1",
        )

    # Verify no points were left orphaned in Qdrant
    hits = in_memory_qdrant.search_vector(
        collection_name="knowledge_base",
        query_vector=[0.1] * 1536,
        team_id=team.id,
        top_k=10,
    )
    assert len(hits) == 0
