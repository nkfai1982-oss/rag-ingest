"""FastAPI application factory, lifespan management, and global exception handlers."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_ingest.api.dependencies import get_repository, get_settings, get_vector_store
from rag_ingest.api.routes import documents, ingest, members, retrieval, teams
from rag_ingest.config import settings
from rag_ingest.exceptions import (
    AuthenticationError,
    DocumentDuplicateError,
    NotFoundError,
    RagIngestError,
    UnsupportedFileTypeError,
)

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    logger.info("Initializing RAG Ingestion Service...")
    cfg = get_settings()

    # Ensure uploads directory exists
    cfg.uploads_dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize SQLite tables
    repo = get_repository()
    repo.init_db()
    logger.info("SQLite metadata database initialized at %s", cfg.SQLITE_PATH)

    # Initialize Qdrant collection & indexes
    try:
        store = get_vector_store()
        store.init_collection(cfg.QDRANT_COLLECTION, cfg.EMBEDDING_DIM)
        logger.info(
            "Qdrant collection '%s' initialized (dim=%d)",
            cfg.QDRANT_COLLECTION,
            cfg.EMBEDDING_DIM,
        )
    except Exception as e:
        logger.warning("Could not connect to Qdrant during startup (will retry on requests): %s", e)

    yield
    logger.info("RAG Ingestion Service shutting down.")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="RAG Ingestion & Retrieval Service",
        version="0.1.0",
        description="Modular, multi-tenant Data Ingestion and Triple-Mode Retrieval API",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": str(exc)},
        )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(DocumentDuplicateError)
    async def duplicate_handler(request: Request, exc: DocumentDuplicateError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"detail": str(exc), "document_id": exc.document_id},
        )

    @app.exception_handler(UnsupportedFileTypeError)
    async def unsupported_file_handler(
        request: Request, exc: UnsupportedFileTypeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(RagIngestError)
    async def rag_error_handler(request: Request, exc: RagIngestError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    # Register Routers
    app.include_router(members.router)
    app.include_router(teams.router)
    app.include_router(ingest.router)
    app.include_router(retrieval.router)
    app.include_router(documents.router)

    @app.get("/health", tags=["Health"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
