"""Optional OpenAI-compatible embedding backend for hybrid knowledge retrieval."""

from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for every input text in the same order."""


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if len(vectors) != len(texts):
            raise ValueError("embedding response count does not match input count")
        return vectors


def embedding_provider_from_env() -> EmbeddingProvider | None:
    """Build the optional provider without reusing a chat completion key."""
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").strip()
    if not api_key or not model or not base_url:
        return None
    return OpenAIEmbeddingProvider(api_key=api_key, base_url=base_url, model=model)
