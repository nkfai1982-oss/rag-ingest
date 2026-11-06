"""PDF document extractor using pypdfium2."""

from pathlib import Path

import pypdfium2 as pdfium

from rag_ingest.domain.entities import ExtractedChunk
from rag_ingest.domain.protocols import BaseExtractor
from rag_ingest.exceptions import ExtractionError


class PdfExtractor(BaseExtractor):
    """Extracts text page-by-page from PDF documents using pypdfium2."""

    def extract(self, file_path: str) -> list[ExtractedChunk]:
        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        try:
            pdf = pdfium.PdfDocument(str(path))
            chunks: list[ExtractedChunk] = []

            for page_index in range(len(pdf)):
                page = pdf[page_index]
                text_page = page.get_textpage()
                page_text = text_page.get_text_range().strip()

                if page_text:
                    chunks.append(
                        ExtractedChunk(
                            text=page_text,
                            metadata={
                                "file_name": path.name,
                                "file_type": "pdf",
                                "page": page_index + 1,
                                "total_pages": len(pdf),
                            },
                        )
                    )

            return chunks
        except Exception as e:
            raise ExtractionError(f"Failed to extract text from PDF {file_path}: {e}") from e
