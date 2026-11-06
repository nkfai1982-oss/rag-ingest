# RAG Ingestion & Retrieval Service

A production-grade, modular, multi-tenant Data Ingestion and Retrieval Pipeline supporting **Markdown, PDF, HTML, and Word (.docx)** documents with **Semantic, Full-Text, and Hybrid (RRF)** search on **Qdrant** (with an abstraction for **pgvector**).

---

## Architecture & Engineering Standards

The comprehensive architecture design, engineering blueprints, database schemas, and Architecture Decision Records (ADRs) are documented in:

* [`DESIGN_INGESTION_RETRIEVAL_PIPELINE.md`](DESIGN_INGESTION_RETRIEVAL_PIPELINE.md)

---

## Engineering & Coding Standards Applied

* **Runtime & Tooling**: Python 3.12+, `uv` package manager, `ruff` (linting & formatting), `mypy --strict`.
* **Clean Layered Architecture**: Strict `src/` layout with domain isolation (`domain/`, `extractors/`, `pipeline/`, `storage/`, `services/`, `api/`).
* **Dependency Inversion (DIP)**: Core services depend on abstract protocols (`BaseVectorStore`, `BaseExtractor`), allowing 100% testability with mocks and seamless addition of `pgvector`.
* **Resilience**: Exponential backoff retries (`tenacity`) for external embedding APIs; compensating transactions to prevent orphan vectors.
* **Storage Model (Lean Dual-Store)**:
  * **SQLite (`knowledge.db`)**: Multi-tenant teams (API keys), datasets, document registry, SHA-256 deduplication, and job tracking in WAL mode.
  * **Qdrant**: Vector embeddings + chunk text payloads (`page_content`), enabling single-hop sub-50ms retrieval.
  * **Local File Storage**: Preserves raw files under `./data/uploads/{team_id}/` for future UI downloads.
* **Portable Hybrid Search**: Application-level **Reciprocal Rank Fusion (RRF)** in Python, ensuring zero vendor lock-in to Qdrant proprietary features when adding `pgvector`.
* **Dual Execution Modes**: Supports both synchronous (`?async=false`) and asynchronous background tasks (`?async=true`).

