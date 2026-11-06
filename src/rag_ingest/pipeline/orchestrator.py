"""Ingestion pipeline orchestrator coordinating extraction, chunking, and storage."""

import hashlib
import logging
import uuid
from pathlib import Path

from rag_ingest.config import settings
from rag_ingest.domain.entities import Document, DocumentChunk, Team
from rag_ingest.domain.protocols import BaseEmbedder, BaseMetadataRepository, BaseVectorStore
from rag_ingest.exceptions import DocumentDuplicateError, RagIngestError
from rag_ingest.extractors.factory import ExtractorFactory
from rag_ingest.pipeline.cleaner import CleanProcessor
from rag_ingest.pipeline.splitter import RecursiveTextSplitter

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """ETL orchestrator for ingesting, chunking, embedding, and indexing documents."""

    def __init__(
        self,
        repository: BaseMetadataRepository,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        cleaner: CleanProcessor | None = None,
        storage_dir: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.repo = repository
        self.vector_store = vector_store
        self.embedder = embedder
        self.cleaner = cleaner or CleanProcessor()
        self.storage_dir = Path(storage_dir or settings.STORAGE_DIR).resolve()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION

    def run(
        self,
        file_content: bytes,
        file_name: str,
        team: Team,
        dataset_id: str,
        overwrite: bool = False,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        job_id: str | None = None,
    ) -> tuple[str, str, int]:
        """Runs the ingestion pipeline.

        Returns:
            tuple[status, document_id, total_chunks]
        """
        # 1. Calculate SHA-256 hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Check identical content hash
        existing_by_hash = self.repo.get_document_by_hash(file_hash, team.id)
        if existing_by_hash:
            logger.info("File %s matches existing hash (%s), skipping.", file_name, file_hash)
            if job_id:
                self.repo.update_job(job_id, status="SUCCESS", progress_percentage=100)
            return "SKIPPED_DUPLICATE", existing_by_hash.id, existing_by_hash.total_chunks

        # Check existing file with same name in dataset
        existing_by_name = self.repo.get_document_by_name(file_name, dataset_id, team.id)
        if existing_by_name:
            if not overwrite:
                raise DocumentDuplicateError(
                    existing_by_name.id,
                    f"Document '{file_name}' already exists in dataset '{dataset_id}'. "
                    f"Set overwrite=true to replace.",
                )
            # Overwrite enabled: purge previous chunks and metadata
            logger.info("Purging existing document %s for overwrite.", existing_by_name.id)
            self.vector_store.delete_by_document_id(self.collection_name, existing_by_name.id)
            self.repo.delete_document(existing_by_name.id, team.id)

        document_id = str(uuid.uuid4())
        ext = Path(file_name).suffix.lower()

        # 2. Persist raw binary to storage
        team_upload_dir = self.storage_dir / team.id
        team_upload_dir.mkdir(parents=True, exist_ok=True)
        storage_file_path = team_upload_dir / f"{document_id}{ext}"
        storage_file_path.write_bytes(file_content)

        # 3. Create document record in SQLite
        doc = Document(
            id=document_id,
            team_id=team.id,
            dataset_id=dataset_id,
            file_name=file_name,
            file_type=ext.lstrip("."),
            file_size_bytes=len(file_content),
            file_hash=file_hash,
            storage_path=str(storage_file_path),
            status="PROCESSING",
            total_chunks=0,
        )
        self.repo.create_document(doc)

        upsert_completed = False
        try:
            if job_id:
                self.repo.update_job(
                    job_id,
                    status="RUNNING",
                    progress_percentage=20,
                    document_id=document_id,
                )

            # 4. Extract content
            extractor = ExtractorFactory.get_extractor(storage_file_path)
            extracted_chunks = extractor.extract(str(storage_file_path))

            if job_id:
                self.repo.update_job(job_id, status="RUNNING", progress_percentage=40)

            # 5. Clean & Split
            effective_chunk_size = chunk_size or team.default_chunk_size
            effective_chunk_overlap = chunk_overlap or team.default_chunk_overlap
            splitter = RecursiveTextSplitter(
                chunk_size=effective_chunk_size,
                chunk_overlap=effective_chunk_overlap,
            )

            processed_chunks: list[DocumentChunk] = []
            chunk_index = 0

            for ext_chunk in extracted_chunks:
                cleaned_text = self.cleaner.clean(ext_chunk.text)
                if not cleaned_text:
                    continue

                split_pieces = splitter.split_text(cleaned_text)
                for piece in split_pieces:
                    metadata = dict(ext_chunk.metadata)
                    metadata["file_name"] = file_name
                    metadata["chunk_index"] = chunk_index
                    metadata["dataset_id"] = dataset_id

                    processed_chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            document_id=document_id,
                            team_id=team.id,
                            dataset_id=dataset_id,
                            chunk_index=chunk_index,
                            text=piece,
                            file_name=file_name,
                            metadata=metadata,
                        )
                    )
                    chunk_index += 1

            if job_id:
                self.repo.update_job(job_id, status="RUNNING", progress_percentage=60)

            # 6. Generate Vector Embeddings
            texts_to_embed = [c.text for c in processed_chunks]
            embeddings = self.embedder.embed_texts(texts_to_embed)
            for c, emb in zip(processed_chunks, embeddings, strict=True):
                c.embedding = emb

            if job_id:
                self.repo.update_job(job_id, status="RUNNING", progress_percentage=80)

            # 7. Upsert to Vector Store
            self.vector_store.upsert_points(self.collection_name, processed_chunks)
            upsert_completed = True

            # 8. Update Document Status in SQLite
            self.repo.update_document_status(
                document_id=document_id,
                status="COMPLETED",
                total_chunks=len(processed_chunks),
            )

            if job_id:
                self.repo.update_job(job_id, status="SUCCESS", progress_percentage=100)

            return "SUCCESS", document_id, len(processed_chunks)

        except Exception as e:
            logger.error("Ingestion failed for doc %s: %s", document_id, e, exc_info=True)
            # Compensating transaction: purge orphan points in Qdrant if upsert succeeded
            if upsert_completed:
                logger.warning(
                    "Executing compensating transaction: purging vectors for doc %s", document_id
                )
                try:
                    self.vector_store.delete_by_document_id(self.collection_name, document_id)
                except Exception as rollback_err:
                    logger.critical(
                        "Compensating transaction failed for doc %s: %s", document_id, rollback_err
                    )

            try:
                self.repo.update_document_status(
                    document_id=document_id,
                    status="FAILED",
                    error_message=str(e),
                )
            except Exception as doc_err:
                logger.error("Could not record document failure status: %s", doc_err)

            if job_id:
                try:
                    self.repo.update_job(
                        job_id,
                        status="FAILED",
                        error_message=str(e),
                    )
                except Exception as job_err:
                    logger.error("Could not record job failure status: %s", job_err)

            raise RagIngestError(f"Ingestion failed: {e}") from e
