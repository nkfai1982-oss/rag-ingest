"""Word (.docx) document extractor using python-docx."""

from pathlib import Path

import docx

from rag_ingest.domain.entities import ExtractedChunk
from rag_ingest.domain.protocols import BaseExtractor
from rag_ingest.exceptions import ExtractionError


class DocxExtractor(BaseExtractor):
    """Extracts text paragraphs and tables from Word (.docx) documents."""

    def extract(self, file_path: str) -> list[ExtractedChunk]:
        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        try:
            doc = docx.Document(str(path))
            lines: list[str] = []

            # Extract paragraphs
            for paragraph in doc.paragraphs:
                p_text = paragraph.text.strip()
                if p_text:
                    lines.append(p_text)

            # Extract tables
            for table in doc.tables:
                table_lines: list[str] = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    if any(row_data):
                        table_lines.append(" | ".join(row_data))
                if table_lines:
                    lines.append("\n".join(table_lines))

            full_text = "\n\n".join(lines).strip()
            if not full_text:
                return []

            return [
                ExtractedChunk(
                    text=full_text,
                    metadata={
                        "file_name": path.name,
                        "file_type": "docx",
                        "paragraphs_count": len(doc.paragraphs),
                        "tables_count": len(doc.tables),
                    },
                )
            ]
        except Exception as e:
            raise ExtractionError(f"Failed to extract text from DOCX {file_path}: {e}") from e
