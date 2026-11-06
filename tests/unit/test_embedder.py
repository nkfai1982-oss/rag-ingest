"""Unit tests for EmbedderClient."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag_ingest.exceptions import EmbeddingError
from rag_ingest.services.embedder import EmbedderClient


def test_embedder_empty_input() -> None:
    client = EmbedderClient()
    assert client.embed_texts([]) == []


@patch("httpx.Client.post")
def test_embedder_success(mock_post: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]
    }
    mock_post.return_value = mock_resp

    client = EmbedderClient(batch_size=2)
    vectors = client.embed_texts(["first", "second"])
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2]
    assert vectors[1] == [0.3, 0.4]


@patch("httpx.Client.post")
def test_embedder_retry_and_failure(mock_post: MagicMock) -> None:
    mock_post.side_effect = httpx.RequestError("Connection refused")

    client = EmbedderClient()
    with pytest.raises(EmbeddingError):
        client.embed_texts(["test string"])
