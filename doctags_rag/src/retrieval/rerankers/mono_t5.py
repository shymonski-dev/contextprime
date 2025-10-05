"""MonoT5 reranker implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, TYPE_CHECKING

from loguru import logger

try:  # pragma: no cover - optional heavy dependency
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None

try:  # pragma: no cover - optional heavy dependency
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:  # pragma: no cover
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from ..hybrid_retriever import HybridSearchResult


@dataclass
class _MonoT5Config:
    model_name: str
    device: Optional[str] = None
    max_length: int = 512


class MonoT5Reranker:
    """Apply monoT5 scoring to reorder hybrid retrieval results."""

    def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 512) -> None:
        self.config = _MonoT5Config(model_name=model_name, device=device, max_length=max_length)
        if torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise RuntimeError("MonoT5 reranker requires torch and transformers to be installed")
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        logger.info("Loading MonoT5 reranker (%s) on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Pre-compute token ids for true/false
        self.true_token_id = self.tokenizer.encode(" true", add_special_tokens=False)[0]
        self.false_token_id = self.tokenizer.encode(" false", add_special_tokens=False)[0]

    def rerank(
        self,
        query: str,
        results: List["HybridSearchResult"],
        top_k: Optional[int] = None,
    ) -> List[HybridSearchResult]:
        if not results:
            return results

        limit = top_k or len(results)
        candidates = results[:limit]
        scores = self._score_candidates(query, candidates)

        paired = list(zip(candidates, scores))
        paired.sort(key=lambda item: item[1], reverse=True)

        reranked: List[HybridSearchResult] = []
        for original, score in paired:
            prob = float(score)
            new_score = max(original.score, prob)
            new_conf = max(original.confidence, prob)
            reranked.append(
                replace(
                    original,
                    score=new_score,
                    confidence=min(1.0, new_conf),
                )
            )

        if limit < len(results):
            reranked.extend(results[limit:])

        return reranked

    def _score_candidates(self, query: str, candidates: Iterable["HybridSearchResult"]) -> List[float]:
        inputs = [self._format_input(query, c) for c in candidates]
        encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        decoder_input_ids = torch.full(
            (len(inputs), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings.get("attention_mask"),
                decoder_input_ids=decoder_input_ids,
            )

        logits = outputs.logits[:, 0, :]
        candidate_logits = logits[:, [self.true_token_id, self.false_token_id]]
        probs = torch.nn.functional.softmax(candidate_logits, dim=-1)

        return probs[:, 0].tolist()

    def _format_input(self, query: str, result: "HybridSearchResult") -> str:
        document = result.content or result.metadata.get("text", "")
        return f"Query: {query} Document: {document} Relevant:"
