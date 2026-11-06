"""Factory for selecting appropriate document extractor by file extension."""

from pathlib import Path

from rag_ingest.domain.protocols import BaseExtractor
from rag_ingest.exceptions import UnsupportedFileTypeError
from rag_ingest.extractors.docx import DocxExtractor
from rag_ingest.extractors.html import HtmlExtractor
from rag_ingest.extractors.markdown import MarkdownExtractor
from rag_ingest.extractors.pdf import PdfExtractor


class ExtractorFactory:
    """Factory resolving file extension to the matching BaseExtractor implementation."""

    _EXTRACTORS: dict[str, type[BaseExtractor]] = {
        ".md": MarkdownExtractor,
        ".markdown": MarkdownExtractor,
        ".txt": MarkdownExtractor,
        ".pdf": PdfExtractor,
        ".html": HtmlExtractor,
        ".htm": HtmlExtractor,
        ".docx": DocxExtractor,
    }

    @classmethod
    def get_extractor(cls, file_path: str | Path) -> BaseExtractor:
        """Returns an instance of BaseExtractor for the file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()
        extractor_cls = cls._EXTRACTORS.get(ext)
        if extractor_cls is None:
            supported = ", ".join(sorted(cls._EXTRACTORS.keys()))
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{ext}'. Supported extensions are: {supported}"
            )
        return extractor_cls()

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return list of all supported file extensions."""
        return list(cls._EXTRACTORS.keys())
