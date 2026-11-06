"""Unit tests for RecursiveTextSplitter."""

import pytest

from rag_ingest.pipeline.splitter import RecursiveTextSplitter


def test_splitter_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        RecursiveTextSplitter(chunk_size=100, chunk_overlap=100)


def test_splitter_empty_text() -> None:
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
    assert splitter.split_text("") == []
    assert splitter.split_text("   ") == []


def test_splitter_short_text() -> None:
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
    text = "Short text under chunk size."
    chunks = splitter.split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_splitter_splits_on_paragraphs() -> None:
    splitter = RecursiveTextSplitter(chunk_size=60, chunk_overlap=10)
    text = (
        "First paragraph with some words.\n\n"
        "Second paragraph with more interesting details.\n\n"
        "Third paragraph concluding the piece."
    )
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 60


def test_splitter_overlap_continuity() -> None:
    splitter = RecursiveTextSplitter(chunk_size=40, chunk_overlap=15)
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron."
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    # Check that adjacent chunks share overlapping text
    for i in range(len(chunks) - 1):
        assert any(word in chunks[i + 1] for word in chunks[i].split()[-2:])
