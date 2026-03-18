"""Hybrid retriever using Reciprocal Rank Fusion (RRF).

Combines dense (FAISS) and sparse (BM25) retrieval results,
replicating the VecDash RRF query method.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import Passage

from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.retriever.dense import DenseRetriever
from agentic_rag.retriever.sparse import SparseRetriever


@dataclass
class HybridRetriever:
    """RRF-based hybrid retriever combining dense + sparse search."""

    dense: DenseRetriever = field(default_factory=DenseRetriever)
    sparse: SparseRetriever = field(default_factory=SparseRetriever)
    rrf_k: int = 60  # RRF constant (standard value)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def build_index(self, passages: list[Passage]) -> None:
        """Build both dense and sparse indices."""
        self.dense.build_index(passages)
        self.sparse.build_index(passages)

    def save(self, path: Path) -> None:
        """Save both indices."""
        self.dense.save(path / "dense")
        self.sparse.save(path / "sparse")

    def load(self, path: Path) -> None:
        """Load both indices. Falls back to dense-only if sparse index is absent."""
        self.dense.load(path / "dense")
        sparse_path = path / "sparse"
        if sparse_path.exists():
            self.sparse.load(sparse_path)
        else:
            logger.warning(f"Sparse index not found at {sparse_path} — using dense-only retrieval")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        exclude_ids: set[str] | None = None,
        method: str | None = None,
    ) -> list[tuple[str, float]]:
        """Hybrid search with configurable fusion method.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            exclude_ids: Passage IDs to exclude (for accumulation).
            method: Override query method. One of:
                    "rrf" (default), "vector_only", "text_only", "combined".

        Returns:
            List of (passage_id, rrf_score) sorted by descending score.
        """
        top_k = top_k or settings.retrieval.top_k
        method = method or settings.retrieval.query_method
        exclude = exclude_ids or set()

        if method == "vector_only":
            return self.dense.search(query, top_k, exclude)

        if method == "text_only":
            return self.sparse.search(query, top_k, exclude)

        # RRF or combined — fetch from both; fall back to dense-only if sparse unavailable
        dense_results = self.dense.search(query, top_k, exclude)
        if self.sparse.bm25 is None:
            return dense_results
        sparse_results = self.sparse.search(query, top_k, exclude)

        if method == "combined":
            return self._weighted_combine(dense_results, sparse_results, top_k)

        # Default: RRF
        return self._rrf_fuse(dense_results, sparse_results, top_k)

    def _rrf_fuse(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion.

        RRF_score(d) = Σ  1 / (k + rank_i(d))
        where k is a constant (default 60) and rank_i is the rank in list i.
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        for rank, (pid, _score) in enumerate(dense_results, start=1):
            rrf_scores[pid] += 1.0 / (self.rrf_k + rank)

        for rank, (pid, _score) in enumerate(sparse_results, start=1):
            rrf_scores[pid] += 1.0 / (self.rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _weighted_combine(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Weighted linear combination using hybrid_weight.

        combined_score = w * dense_norm + (1-w) * sparse_norm
        """
        w = settings.retrieval.hybrid_weight

        # Normalize scores to [0, 1]
        def _normalize(results: list[tuple[str, float]]) -> dict[str, float]:
            if not results:
                return {}
            max_s = max(s for _, s in results)
            min_s = min(s for _, s in results)
            rng = max_s - min_s if max_s > min_s else 1.0
            return {pid: (s - min_s) / rng for pid, s in results}

        dense_norm = _normalize(dense_results)
        sparse_norm = _normalize(sparse_results)

        all_pids = set(dense_norm) | set(sparse_norm)
        combined: dict[str, float] = {}
        for pid in all_pids:
            d = dense_norm.get(pid, 0.0)
            s = sparse_norm.get(pid, 0.0)
            combined[pid] = w * d + (1 - w) * s

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
