"""Integration tests for SQLite repository and QdrantVectorStore."""

import uuid

from rag_ingest.domain.entities import Dataset, Document, DocumentChunk, Team
from rag_ingest.storage.qdrant.store import QdrantVectorStore
from rag_ingest.storage.sqlite.repository import SQLiteMetadataRepository


def test_sqlite_repository_crud(sqlite_repo: SQLiteMetadataRepository) -> None:
    # 1. Team CRUD
    team = Team(
        id="team_test_repo",
        name="Repo Team",
        api_key="secret_repo_key",
        default_chunk_size=600,
        default_chunk_overlap=60,
    )
    sqlite_repo.create_team(team)

    retrieved_team = sqlite_repo.get_team_by_api_key("secret_repo_key")
    assert retrieved_team is not None
    assert retrieved_team.id == "team_test_repo"

    # 2. Dataset CRUD
    ds = Dataset(
        id="ds_test_repo",
        team_id="team_test_repo",
        name="Test Dataset",
    )
    sqlite_repo.create_dataset(ds)

    datasets = sqlite_repo.list_datasets("team_test_repo")
    assert len(datasets) == 1
    assert datasets[0].name == "Test Dataset"

    # 3. Document CRUD
    doc = Document(
        id="doc_123",
        team_id="team_test_repo",
        dataset_id="ds_test_repo",
        file_name="test.md",
        file_type="md",
        file_size_bytes=100,
        file_hash="hash_abc_123",
        storage_path="/path/test.md",
        status="PENDING",
    )
    sqlite_repo.create_document(doc)

    by_hash = sqlite_repo.get_document_by_hash("hash_abc_123", "team_test_repo")
    assert by_hash is not None
    assert by_hash.id == "doc_123"

    by_name = sqlite_repo.get_document_by_name("test.md", "ds_test_repo", "team_test_repo")
    assert by_name is not None
    assert by_name.id == "doc_123"

    sqlite_repo.update_document_status("doc_123", status="COMPLETED", total_chunks=5)
    updated = sqlite_repo.get_document_by_id("doc_123", "team_test_repo")
    assert updated is not None
    assert updated.status == "COMPLETED"
    assert updated.total_chunks == 5

    # 4. Delete
    deleted = sqlite_repo.delete_document("doc_123", "team_test_repo")
    assert deleted is True
    assert sqlite_repo.get_document_by_id("doc_123", "team_test_repo") is None


def test_qdrant_store_vector_and_text_search(in_memory_qdrant: QdrantVectorStore) -> None:
    collection = "knowledge_base"
    doc_id = str(uuid.uuid4())
    team_id = "team_alpha"
    dataset_id = "ds_alpha"

    # Insert two chunks
    chunk1 = DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=doc_id,
        team_id=team_id,
        dataset_id=dataset_id,
        chunk_index=0,
        text="FastAPI authentication and security configuration with Bearer tokens.",
        embedding=[0.1] * 1536,
        file_name="security.md",
    )
    chunk2 = DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=doc_id,
        team_id=team_id,
        dataset_id=dataset_id,
        chunk_index=1,
        text="Docker container deployment strategies for production environments.",
        embedding=[-0.1] * 1536,
        file_name="docker.md",
    )

    in_memory_qdrant.upsert_points(collection, [chunk1, chunk2])

    # 1. Vector Search
    vec_results = in_memory_qdrant.search_vector(
        collection_name=collection,
        query_vector=[0.1] * 1536,
        team_id=team_id,
        top_k=2,
    )
    assert len(vec_results) == 2
    assert vec_results[0].chunk_id == chunk1.chunk_id

    # 2. Lexical Text Search
    text_results = in_memory_qdrant.search_text(
        collection_name=collection,
        query_text="authentication",
        team_id=team_id,
        top_k=5,
    )
    assert len(text_results) >= 1
    assert any("authentication" in r.content.lower() for r in text_results)

    # 3. Purge by document id
    in_memory_qdrant.delete_by_document_id(collection, doc_id)
    post_delete_vec = in_memory_qdrant.search_vector(
        collection_name=collection,
        query_vector=[0.1] * 1536,
        team_id=team_id,
        top_k=5,
    )
    assert len(post_delete_vec) == 0
