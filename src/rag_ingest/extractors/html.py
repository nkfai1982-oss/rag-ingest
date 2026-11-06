"""HTML document extractor using BeautifulSoup4."""

from pathlib import Path

from bs4 import BeautifulSoup

from rag_ingest.domain.entities import ExtractedChunk
from rag_ingest.domain.protocols import BaseExtractor
from rag_ingest.exceptions import ExtractionError


class HtmlExtractor(BaseExtractor):
    """Extracts clean text and structural elements from HTML files."""

    def extract(self, file_path: str) -> list[ExtractedChunk]:
        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        try:
            content = path.read_bytes()
            soup = BeautifulSoup(content, "lxml")

            # Remove non-content elements
            for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
                tag.decompose()

            # Extract title if present
            title = soup.title.string.strip() if soup.title and soup.title.string else ""

            # Extract clean text with newline separation between block elements
            text = soup.get_text(separator="\n\n", strip=True)

            if not text:
                return []

            return [
                ExtractedChunk(
                    text=text,
                    metadata={
                        "file_name": path.name,
                        "file_type": "html",
                        "title": title,
                        "file_size": path.stat().st_size,
                    },
                )
            ]
        except Exception as e:
            raise ExtractionError(f"Failed to extract text from HTML {file_path}: {e}") from e
