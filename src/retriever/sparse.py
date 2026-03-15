"""BM25-based sparse (keyword) retriever.

Replaces VecDash text search with rank_bm25.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from src.retriever.indexer import Passage

from config.settings import settings


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


@dataclass
class SparseRetriever:
    """BM25Okapi keyword retriever."""

    bm25: BM25Okapi | None = field(default=None, repr=False)
    passage_ids: list[str] = field(default_factory=list)
    _corpus_tokens: list[list[str]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def build_index(self, passages: list[Passage]) -> None:
        """Build BM25 index from passages."""
        self._corpus_tokens = [_tokenize(p.content) for p in passages]
        self.passage_ids = [p.id for p in passages]
        self.bm25 = BM25Okapi(self._corpus_tokens)
        logger.info(f"Sparse index built: {len(self.passage_ids)} documents")

    def save(self, path: Path) -> None:
        """Save BM25 index data to disk."""
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "passage_ids": self.passage_ids,
            "corpus_tokens": self._corpus_tokens,
        }
        with open(path / "bm25_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """Load BM25 index data from disk."""
        with open(path / "bm25_data.json", encoding="utf-8") as f:
            data = json.load(f)
        self.passage_ids = data["passage_ids"]
        self._corpus_tokens = data["corpus_tokens"]
        self.bm25 = BM25Okapi(self._corpus_tokens)
        logger.info(f"Sparse index loaded: {len(self.passage_ids)} documents")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search the BM25 index.

        Returns list of (passage_id, score) sorted by descending score.
        """
        if self.bm25 is None:
            return []

        top_k = top_k or settings.retrieval.text_top_k
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[tuple[str, float]] = []
        for idx, score in ranked:
            pid = self.passage_ids[idx]
            if exclude_ids and pid in exclude_ids:
                continue
            if score <= 0:
                continue
            results.append((pid, float(score)))
            if len(results) >= top_k:
                break

        return results
