"""Markdown document extractor."""

from pathlib import Path

from rag_ingest.domain.entities import ExtractedChunk
from rag_ingest.domain.protocols import BaseExtractor
from rag_ingest.exceptions import ExtractionError


class MarkdownExtractor(BaseExtractor):
    """Extracts text from Markdown (.md) documents."""

    def extract(self, file_path: str) -> list[ExtractedChunk]:
        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="latin-1")
            except Exception as e:
                raise ExtractionError(f"Failed to read markdown file {file_path}: {e}") from e
        except Exception as e:
            raise ExtractionError(f"Failed to read markdown file {file_path}: {e}") from e

        cleaned = content.strip()
        if not cleaned:
            return []

        return [
            ExtractedChunk(
                text=cleaned,
                metadata={
                    "file_name": path.name,
                    "file_type": "md",
                    "file_size": path.stat().st_size,
                },
            )
        ]
