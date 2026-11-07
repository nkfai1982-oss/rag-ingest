"""Pytest fixtures for unit and integration testing."""

import hashlib
from collections.abc import Generator
from pathlib import Path

import docx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from rag_ingest.api.dependencies import (
    get_embedder,
    get_pipeline,
    get_repository,
    get_search_service,
    get_vector_store,
)
from rag_ingest.api.main import create_app
from rag_ingest.domain.entities import Dataset, Member, Team
from rag_ingest.domain.protocols import BaseEmbedder
from rag_ingest.pipeline.cleaner import CleanProcessor
from rag_ingest.pipeline.orchestrator import IngestionPipeline
from rag_ingest.services.search import SearchService
from rag_ingest.storage.qdrant.store import QdrantVectorStore
from rag_ingest.storage.sqlite.connection import SQLiteConnection
from rag_ingest.storage.sqlite.repository import SQLiteMetadataRepository


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder generating repeatable unit vectors."""

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def _generate_vector(self, text: str) -> list[float]:
        # Hash text to generate repeatable pseudorandom float vector
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        vec = [((h >> (i % 64)) & 0xFF) / 255.0 for i in range(self.dim)]
        # Normalize vector
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._generate_vector(text)


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=1536)


@pytest.fixture
def sqlite_repo() -> Generator[SQLiteMetadataRepository, None, None]:
    conn_provider = SQLiteConnection(":memory:")
    repo = SQLiteMetadataRepository(conn_provider)
    repo.init_db()

    # Pre-seed test team, member, and dataset
    test_team = Team(
        id="test_team_1",
        name="Engineering Team",
        default_chunk_size=500,
        default_chunk_overlap=50,
    )
    repo.create_team(test_team)

    test_member = Member(
        email="admin@test.com",
        team_id="test_team_1",
        api_key="test_api_key_123",
    )
    repo.create_member(test_member)

    test_dataset = Dataset(
        id="test_dataset_1",
        team_id="test_team_1",
        name="Documentation",
        description="Docs dataset",
    )
    repo.create_dataset(test_dataset)

    yield repo


@pytest.fixture
def in_memory_qdrant() -> Generator[QdrantVectorStore, None, None]:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client)
    store.init_collection("knowledge_base", 1536)
    yield store


@pytest.fixture
def test_pipeline(
    sqlite_repo: SQLiteMetadataRepository,
    in_memory_qdrant: QdrantVectorStore,
    mock_embedder: MockEmbedder,
    tmp_path: Path,
) -> IngestionPipeline:
    return IngestionPipeline(
        repository=sqlite_repo,
        vector_store=in_memory_qdrant,
        embedder=mock_embedder,
        cleaner=CleanProcessor(),
        storage_dir=tmp_path / "uploads",
        collection_name="knowledge_base",
    )


@pytest.fixture
def search_service(
    in_memory_qdrant: QdrantVectorStore,
    mock_embedder: MockEmbedder,
) -> SearchService:
    return SearchService(
        vector_store=in_memory_qdrant,
        embedder=mock_embedder,
        collection_name="knowledge_base",
    )


@pytest.fixture
def test_client(
    sqlite_repo: SQLiteMetadataRepository,
    in_memory_qdrant: QdrantVectorStore,
    mock_embedder: MockEmbedder,
    test_pipeline: IngestionPipeline,
    search_service: SearchService,
) -> Generator[TestClient, None, None]:
    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_repository] = lambda: sqlite_repo
    app.dependency_overrides[get_vector_store] = lambda: in_memory_qdrant
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_pipeline] = lambda: test_pipeline
    app.dependency_overrides[get_search_service] = lambda: search_service

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_files(tmp_path: Path) -> dict[str, Path]:
    """Generates real sample files (.md, .html, .docx, .pdf) for test fixtures."""
    files: dict[str, Path] = {}

    # 1. Markdown
    md_path = tmp_path / "sample.md"
    md_path.write_text(
        "# Title\n\nThis is a sample document for RAG ingestion.\n\n"
        "| Name | Role |\n|---|---|\n| Alice | Dev |\n\n"
        "```python\nprint('hello world')\n```",
        encoding="utf-8",
    )
    files["md"] = md_path

    # 2. HTML
    html_path = tmp_path / "sample.html"
    html_path.write_text(
        "<html><head><title>Web Page</title><style>body{color:red;}</style></head>"
        "<body><h1>Header One</h1><script>alert(1);</script>"
        "<p>This is an HTML test document paragraph.</p></body></html>",
        encoding="utf-8",
    )
    files["html"] = html_path

    # 3. DOCX
    docx_path = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_heading("Docx Document", level=1)
    doc.add_paragraph("This is a Word document test paragraph for ingestion testing.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Key"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Status"
    table.cell(1, 1).text = "Active"
    doc.save(str(docx_path))
    files["docx"] = docx_path

    # 4. PDF
    pdf_path = tmp_path / "sample.pdf"
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 55>>stream\n"
        b"BT /F1 12 Tf 72 712 Td (This is a PDF test document page content.) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \n"
        b"0000000220 00000 n \n0000000326 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n393\n%%EOF"
    )
    pdf_path.write_bytes(pdf_bytes)
    files["pdf"] = pdf_path

    return files
