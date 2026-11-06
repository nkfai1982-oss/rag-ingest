"""SQLite implementation of the MetadataRepository."""

import sqlite3
from datetime import UTC, datetime

from rag_ingest.domain.entities import Dataset, Document, IngestionJob, Team
from rag_ingest.domain.protocols import BaseMetadataRepository
from rag_ingest.storage.sqlite.connection import SQLiteConnection


class SQLiteMetadataRepository(BaseMetadataRepository):
    """Metadata repository managing teams, datasets, documents, and jobs in SQLite."""

    def __init__(self, connection_provider: SQLiteConnection) -> None:
        self._provider = connection_provider

    def init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(
                """
                -- 1. Tenant / Team Registry
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL,
                    default_chunk_size INTEGER DEFAULT 800,
                    default_chunk_overlap INTEGER DEFAULT 100,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- 2. Datasets (Knowledge Bases)
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    team_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE
                );

                -- 3. Document Registry
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    team_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER,
                    file_hash TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'PENDING',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE
                );

                -- 4. Ingestion Jobs
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    team_id TEXT NOT NULL,
                    status TEXT DEFAULT 'RUNNING',
                    progress_percentage INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_docs_team_dataset ON documents(team_id, dataset_id);
                CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(team_id, file_hash);
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(status);
                """
            )

    # Teams
    def create_team(self, team: Team) -> Team:
        now = datetime.now(UTC).isoformat()
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO teams (
                    id, name, api_key, default_chunk_size, default_chunk_overlap, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    team.id,
                    team.name,
                    team.api_key,
                    team.default_chunk_size,
                    team.default_chunk_overlap,
                    team.created_at or now,
                ),
            )
            return team

    def get_team_by_api_key(self, api_key: str) -> Team | None:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teams WHERE api_key = ?", (api_key,))
            row = cursor.fetchone()
            if not row:
                return None
            return Team(
                id=row["id"],
                name=row["name"],
                api_key=row["api_key"],
                default_chunk_size=row["default_chunk_size"],
                default_chunk_overlap=row["default_chunk_overlap"],
                created_at=row["created_at"],
            )

    def get_team_by_id(self, team_id: str) -> Team | None:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teams WHERE id = ?", (team_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Team(
                id=row["id"],
                name=row["name"],
                api_key=row["api_key"],
                default_chunk_size=row["default_chunk_size"],
                default_chunk_overlap=row["default_chunk_overlap"],
                created_at=row["created_at"],
            )

    def list_teams(self) -> list[Team]:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM teams ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [
                Team(
                    id=row["id"],
                    name=row["name"],
                    api_key=row["api_key"],
                    default_chunk_size=row["default_chunk_size"],
                    default_chunk_overlap=row["default_chunk_overlap"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    # Datasets
    def create_dataset(self, dataset: Dataset) -> Dataset:
        now = datetime.now(UTC).isoformat()
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO datasets (id, team_id, name, description, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    dataset.id,
                    dataset.team_id,
                    dataset.name,
                    dataset.description,
                    dataset.created_at or now,
                ),
            )
            return dataset

    def get_dataset(self, dataset_id: str, team_id: str) -> Dataset | None:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM datasets WHERE id = ? AND team_id = ?",
                (dataset_id, team_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return Dataset(
                id=row["id"],
                team_id=row["team_id"],
                name=row["name"],
                description=row["description"],
                created_at=row["created_at"],
            )

    def list_datasets(self, team_id: str) -> list[Dataset]:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM datasets WHERE team_id = ? ORDER BY created_at DESC",
                (team_id,),
            )
            rows = cursor.fetchall()
            return [
                Dataset(
                    id=row["id"],
                    team_id=row["team_id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    # Documents
    def create_document(self, document: Document) -> Document:
        now = datetime.now(UTC).isoformat()
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents (
                    id, team_id, dataset_id, file_name, file_type, file_size_bytes,
                    file_hash, storage_path, total_chunks, status, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.team_id,
                    document.dataset_id,
                    document.file_name,
                    document.file_type,
                    document.file_size_bytes,
                    document.file_hash,
                    document.storage_path,
                    document.total_chunks,
                    document.status,
                    document.error_message,
                    document.created_at or now,
                    document.updated_at or now,
                ),
            )
            return document

    def get_document_by_id(self, document_id: str, team_id: str | None = None) -> Document | None:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            if team_id:
                cursor.execute(
                    "SELECT * FROM documents WHERE id = ? AND team_id = ?",
                    (document_id, team_id),
                )
            else:
                cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_document(row)

    def get_document_by_hash(self, file_hash: str, team_id: str) -> Document | None:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE file_hash = ? AND team_id = ?",
                (file_hash, team_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_document(row)

    def get_document_by_name(
        self, file_name: str, dataset_id: str, team_id: str
    ) -> Document | None:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM documents WHERE file_name = ? AND dataset_id = ? AND team_id = ?",
                (file_name, dataset_id, team_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_document(row)

    def update_document_status(
        self,
        document_id: str,
        status: str,
        total_chunks: int | None = None,
        error_message: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            if total_chunks is not None:
                cursor.execute(
                    """
                    UPDATE documents
                    SET status = ?, total_chunks = ?, error_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, total_chunks, error_message, now, document_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE documents
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, error_message, now, document_id),
                )

    def list_documents(self, team_id: str, dataset_id: str | None = None) -> list[Document]:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            if dataset_id:
                cursor.execute(
                    """
                    SELECT * FROM documents
                    WHERE team_id = ? AND dataset_id = ?
                    ORDER BY created_at DESC
                    """,
                    (team_id, dataset_id),
                )
            else:
                cursor.execute(
                    "SELECT * FROM documents WHERE team_id = ? ORDER BY created_at DESC",
                    (team_id,),
                )
            rows = cursor.fetchall()
            return [self._row_to_document(row) for row in rows]

    def delete_document(self, document_id: str, team_id: str) -> bool:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM documents WHERE id = ? AND team_id = ?",
                (document_id, team_id),
            )
            return cursor.rowcount > 0

    # Ingestion Jobs
    def create_job(self, job: IngestionJob) -> IngestionJob:
        now = datetime.now(UTC).isoformat()
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ingestion_jobs (
                    id, document_id, team_id, status, progress_percentage,
                    error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.document_id,
                    job.team_id,
                    job.status,
                    job.progress_percentage,
                    job.error_message,
                    job.created_at or now,
                    job.updated_at or now,
                ),
            )
            return job

    def get_job(self, job_id: str, team_id: str | None = None) -> IngestionJob | None:
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            if team_id:
                cursor.execute(
                    "SELECT * FROM ingestion_jobs WHERE id = ? AND team_id = ?",
                    (job_id, team_id),
                )
            else:
                cursor.execute(
                    "SELECT * FROM ingestion_jobs WHERE id = ?",
                    (job_id,),
                )
            row = cursor.fetchone()
            if not row:
                return None
            return IngestionJob(
                id=row["id"],
                document_id=row["document_id"],
                team_id=row["team_id"],
                status=row["status"],
                progress_percentage=row["progress_percentage"],
                error_message=row["error_message"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def update_job(
        self,
        job_id: str,
        status: str,
        progress_percentage: int | None = None,
        error_message: str | None = None,
        document_id: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._provider.get_connection() as conn:
            cursor = conn.cursor()
            query = "UPDATE ingestion_jobs SET status = ?, error_message = ?, updated_at = ?"
            params: list[object] = [status, error_message, now]

            if progress_percentage is not None:
                query += ", progress_percentage = ?"
                params.append(progress_percentage)

            if document_id is not None:
                query += ", document_id = ?"
                params.append(document_id)

            query += " WHERE id = ?"
            params.append(job_id)

            cursor.execute(query, tuple(params))

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            team_id=row["team_id"],
            dataset_id=row["dataset_id"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            file_size_bytes=row["file_size_bytes"],
            file_hash=row["file_hash"],
            storage_path=row["storage_path"],
            total_chunks=row["total_chunks"],
            status=row["status"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
