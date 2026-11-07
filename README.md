# RAG Ingestion & Retrieval Service

A production-grade, modular, multi-tenant Data Ingestion and Retrieval Pipeline supporting **Markdown, PDF, HTML, and Word (.docx)** documents with **Semantic, Full-Text, and Hybrid (RRF)** search on **Qdrant** (with an abstraction for **pgvector**).

---

## Architecture & Engineering Standards

The comprehensive architecture design, engineering blueprints, database schemas, and Architecture Decision Records (ADRs) are documented in:

* [`docs/DESIGN_INGESTION_RETRIEVAL_PIPELINE.md`](docs/DESIGN_INGESTION_RETRIEVAL_PIPELINE.md)

### Key Architectural Highlights
* **Python Runtime & Standards**: Python 3.12+, `uv` package manager, `ruff` (linting & formatting), `mypy --strict`.
* **Clean Layered Architecture**: Strict `src/` layout with domain isolation (`domain/`, `extractors/`, `pipeline/`, `storage/`, `services/`, `api/`).
* **Dependency Inversion (DIP)**: Core services depend on abstract protocols (`BaseVectorStore`, `BaseMetadataRepository`, `BaseExtractor`, `BaseEmbedder`), allowing 100% testability with mocks and seamless addition of other backends (such as `pgvector`).
* **Resilience**: Exponential backoff retries (`tenacity`) for external embedding APIs; compensating transactions to automatically purge orphan vectors on metadata failures.
* **Lean Dual-Store Model (Option A)**:
  * **SQLite (`knowledge.db`)**: Single-file metadata database in WAL mode managing multi-tenant teams, team members (identified by email with individual API keys), datasets, document registry, SHA-256 deduplication, and async job queues.
  * **Qdrant**: Vector embeddings + chunk text payloads (`page_content`), enabling single-hop sub-50ms retrieval.
  * **Local File Storage**: Preserves raw files under `./data/uploads/{team_id}/` for download and verification.
* **Portable Hybrid Search**: Application-level **Reciprocal Rank Fusion (RRF)** in Python, fusing dense similarity and lexical text matching without vendor lock-in.
* **Dual Execution Modes**: Supports both synchronous (`?async_mode=false`) and asynchronous background tasks (`?async_mode=true`).

---

## Quickstart

### Prerequisites
* Python 3.12+
* [`uv`](https://docs.astral.sh/uv/) package manager
* Docker & Docker Compose (for running Qdrant)

### 1. Installation
Clone the repository and sync dependencies:
```bash
uv sync
```

### 2. Environment Configuration
Copy `.env.example` or create a `.env` file:
```bash
# Service
SERVICE_PORT=8000
LOG_LEVEL=INFO

# Storage
SQLITE_PATH=data/knowledge.db
STORAGE_DIR=data/uploads

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=knowledge_base

# Embeddings (OpenAI compatible)
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

### 3. Run Locally with Docker Qdrant
Start Qdrant:
```bash
docker compose up qdrant -d
```

Start the FastAPI application:
```bash
uv run uvicorn rag_ingest.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at: `http://localhost:8000/docs`

---

## API Usage Walkthrough

### 1. Register a Member & Specify Team
Registration is at member level. The member's ID is their email address. If the team does not exist yet, it is automatically provisioned:
```bash
curl -X POST http://localhost:8000/v1/members/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "alice@engineering.com",
    "team_id": "eng_team",
    "team_name": "Core Engineering Team"
  }'
```
*Response returns the member's `email`, `team_id`, and their individual `api_key`:*
```json
{
  "email": "alice@engineering.com",
  "team_id": "eng_team",
  "api_key": "rag_m_abc123...",
  "created_at": "2026-11-07T08:00:00"
}
```

Other members can register to the same team:
```bash
curl -X POST http://localhost:8000/v1/members/register \
  -H "Content-Type: application/json" \
  -d '{"email": "bob@engineering.com", "team_id": "eng_team"}'
```

### 2. Check Member Profile & Team Members
Check current member profile:
```bash
curl -X GET http://localhost:8000/v1/members/me \
  -H "Authorization: Bearer <member_api_key>"
```

List all members of your team (ready for the future UI):
```bash
curl -X GET http://localhost:8000/v1/teams/eng_team/members \
  -H "Authorization: Bearer <member_api_key>"
```

### 3. Create a Dataset
```bash
curl -X POST http://localhost:8000/v1/datasets \
  -H "Authorization: Bearer <member_api_key>" \
  -H "Content-Type: application/json" \
  -d '{"id": "api_reference", "name": "API Reference Documentation"}'
```

### 4. Ingest a Document (Synchronous)
```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "Authorization: Bearer <member_api_key>" \
  -F "file=@./docs/DESIGN_INGESTION_RETRIEVAL_PIPELINE.md" \
  -F "dataset_id=api_reference" \
  -F "async_mode=false"
```

### 5. Ingest a Document (Asynchronous with BackgroundTasks)
```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H "Authorization: Bearer <member_api_key>" \
  -F "file=@./manual.pdf" \
  -F "dataset_id=api_reference" \
  -F "async_mode=true"
```
*Returns `202 Accepted` with a `job_id`.*

Check job progress:
```bash
curl -X GET http://localhost:8000/v1/jobs/<job_id> \
  -H "Authorization: Bearer <member_api_key>"
```

### 6. Triple-Mode Retrieval (Hybrid RRF)
```bash
curl -X POST http://localhost:8000/v1/retrieval \
  -H "Authorization: Bearer <member_api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Reciprocal Rank Fusion work?",
    "search_mode": "hybrid",
    "dataset_id": "api_reference",
    "top_k": 5
  }'
```

*(You can also set `search_mode` to `"semantic"` or `"full_text"`).*

### 7. Document Management & Download
List all documents:
```bash
curl -X GET http://localhost:8000/v1/documents \
  -H "Authorization: Bearer <member_api_key>"
```

Download raw stored file:
```bash
curl -X GET http://localhost:8000/v1/documents/<document_id>/download \
  -H "Authorization: Bearer <member_api_key>" \
  --output downloaded_file.md
```

Delete document and purge vectors:
```bash
curl -X DELETE http://localhost:8000/v1/documents/<document_id> \
  -H "Authorization: Bearer <member_api_key>"
```

### 8. Regenerate Member API Key
```bash
curl -X POST http://localhost:8000/v1/members/me/api-key \
  -H "Authorization: Bearer <member_api_key>"
```

---

## Testing & Quality Assurance

Run code formatting and lint check:
```bash
uv run ruff check .
uv run ruff format --check .
```

Run static type checker:
```bash
uv run mypy src
```

Run test suite with coverage:
```bash
uv run pytest tests -v --cov=rag_ingest --cov-report=term-missing
```
