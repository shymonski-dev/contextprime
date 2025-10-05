"""Wrapper around OpenAI embedding API with a simple encode interface."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence

from openai import OpenAI


class OpenAIEmbeddingModel:
    """Thin adapter exposing an ``encode`` method compatible with SentenceTransformer."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        client_kwargs = {}
        if api_key or os.getenv("OPENAI_API_KEY"):
            client_kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")
        if base_url or os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name

    def encode(self, texts: Sequence[str], show_progress_bar: bool = False) -> List[List[float]]:
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        if len(texts) == 0:
            return []

        response = self.client.embeddings.create(model=self.model_name, input=list(texts))
        # Response items are in the same order as inputs
        return [item.embedding for item in response.data]
