"""OpenAI-compatible embeddings client with tenacity retry resilience."""

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_ingest.domain.protocols import BaseEmbedder
from rag_ingest.exceptions import EmbeddingError


class EmbedderClient(BaseEmbedder):
    """Client for generating text embeddings using an OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "test",
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    @retry(
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                # Sort by index to preserve order
                embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in embeddings_data]
        except Exception as e:
            raise EmbeddingError(f"Embedding API request failed: {e}") from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch generate embeddings for multiple texts."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        results = self.embed_texts([text])
        if not results:
            raise EmbeddingError("Failed to embed search query: empty response")
        return results[0]
