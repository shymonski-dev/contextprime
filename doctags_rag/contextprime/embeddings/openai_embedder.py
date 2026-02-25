"""Wrapper around OpenAI embedding API with a simple encode interface."""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence
from collections import OrderedDict
from threading import Lock

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
        try:
            self._cache_size = max(0, int(os.getenv("DOCTAGS_EMBEDDING_CACHE_SIZE", "2048")))
        except ValueError:
            self._cache_size = 2048
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = Lock()

    def encode(self, texts: Sequence[str], show_progress_bar: bool = False) -> List[List[float]]:
        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")
        if len(texts) == 0:
            return []

        normalized_texts = [str(item) for item in texts]
        if self._cache_size <= 0:
            response = self.client.embeddings.create(model=self.model_name, input=normalized_texts)
            return [item.embedding for item in response.data]

        output: List[List[float] | None] = [None] * len(normalized_texts)
        misses: List[str] = []
        miss_indices: List[int] = []

        with self._lock:
            for idx, text in enumerate(normalized_texts):
                cached = self._cache.get(text)
                if cached is None:
                    misses.append(text)
                    miss_indices.append(idx)
                    continue
                self._cache.move_to_end(text)
                output[idx] = list(cached)

        if misses:
            unique_misses = list(dict.fromkeys(misses))
            response = self.client.embeddings.create(model=self.model_name, input=unique_misses)
            miss_vectors = {
                text: list(item.embedding)
                for text, item in zip(unique_misses, response.data)
            }

            with self._lock:
                for text in unique_misses:
                    vector = miss_vectors[text]
                    self._cache[text] = vector
                    self._cache.move_to_end(text)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

            for idx, text in zip(miss_indices, misses):
                output[idx] = list(miss_vectors[text])

        return [item if item is not None else [] for item in output]
