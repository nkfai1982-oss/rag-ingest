"""Unit tests for document extractors (MD, HTML, DOCX, PDF) and ExtractorFactory."""

from pathlib import Path

import pytest

from rag_ingest.exceptions import ExtractionError, UnsupportedFileTypeError
from rag_ingest.extractors.docx import DocxExtractor
from rag_ingest.extractors.factory import ExtractorFactory
from rag_ingest.extractors.html import HtmlExtractor
from rag_ingest.extractors.markdown import MarkdownExtractor
from rag_ingest.extractors.pdf import PdfExtractor


def test_extractor_factory_resolution() -> None:
    assert isinstance(ExtractorFactory.get_extractor("file.md"), MarkdownExtractor)
    assert isinstance(ExtractorFactory.get_extractor("file.pdf"), PdfExtractor)
    assert isinstance(ExtractorFactory.get_extractor("file.html"), HtmlExtractor)
    assert isinstance(ExtractorFactory.get_extractor("file.docx"), DocxExtractor)

    with pytest.raises(UnsupportedFileTypeError):
        ExtractorFactory.get_extractor("file.xyz")


def test_markdown_extractor(sample_files: dict[str, Path]) -> None:
    extractor = MarkdownExtractor()
    chunks = extractor.extract(str(sample_files["md"]))
    assert len(chunks) == 1
    assert "# Title" in chunks[0].text
    assert "Alice" in chunks[0].text
    assert chunks[0].metadata["file_type"] == "md"


def test_html_extractor(sample_files: dict[str, Path]) -> None:
    extractor = HtmlExtractor()
    chunks = extractor.extract(str(sample_files["html"]))
    assert len(chunks) == 1
    text = chunks[0].text
    assert "Header One" in text
    assert "HTML test document" in text
    # Non-content should be stripped
    assert "alert" not in text
    assert "color:red" not in text
    assert chunks[0].metadata["title"] == "Web Page"


def test_docx_extractor(sample_files: dict[str, Path]) -> None:
    extractor = DocxExtractor()
    chunks = extractor.extract(str(sample_files["docx"]))
    assert len(chunks) == 1
    text = chunks[0].text
    assert "Docx Document" in text
    assert "Key | Value" in text
    assert "Status | Active" in text
    assert chunks[0].metadata["file_type"] == "docx"


def test_pdf_extractor(sample_files: dict[str, Path]) -> None:
    extractor = PdfExtractor()
    chunks = extractor.extract(str(sample_files["pdf"]))
    assert len(chunks) >= 1
    assert "PDF test document" in chunks[0].text
    assert chunks[0].metadata["page"] == 1


def test_extractor_non_existent_file() -> None:
    extractor = MarkdownExtractor()
    with pytest.raises(ExtractionError):
        extractor.extract("non_existent_file.md")
