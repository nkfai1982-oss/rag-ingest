"""Unit tests for CleanProcessor."""

from rag_ingest.pipeline.cleaner import CleanProcessor


def test_cleaner_empty_input() -> None:
    cleaner = CleanProcessor()
    assert cleaner.clean("") == ""
    assert cleaner.clean("   ") == ""


def test_cleaner_strips_control_characters() -> None:
    cleaner = CleanProcessor()
    text = "Hello\x00 World\x07!\x1f"
    assert cleaner.clean(text) == "Hello World!"


def test_cleaner_collapses_excessive_newlines() -> None:
    cleaner = CleanProcessor()
    text = "Paragraph 1\n\n\n\n\nParagraph 2\n\n\nParagraph 3"
    result = cleaner.clean(text)
    assert result == "Paragraph 1\n\nParagraph 2\n\nParagraph 3"


def test_cleaner_normalizes_spaces_outside_code_blocks() -> None:
    cleaner = CleanProcessor()
    text = "Sentence   with    multiple     spaces."
    assert cleaner.clean(text) == "Sentence with multiple spaces."


def test_cleaner_preserves_code_blocks() -> None:
    cleaner = CleanProcessor()
    text = "Some text\n```python\n    def func():\n        x = 1\n```\nAfter code."
    cleaned = cleaner.clean(text)
    assert "    def func():" in cleaned
    assert "        x = 1" in cleaned


def test_cleaner_preserves_markdown_tables() -> None:
    cleaner = CleanProcessor()
    text = "| Col 1 | Col 2 |\n|---|---|\n|  Val 1  |  Val 2  |"
    cleaned = cleaner.clean(text)
    assert "| Col 1 | Col 2 |" in cleaned
    assert "|---|---|" in cleaned
