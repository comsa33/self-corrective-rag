"""FAISS-based dense (vector) retriever.

Replaces VecDash vector search with a local FAISS index + OpenAI or
sentence-transformers embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import Passage

from agentic_rag.config.settings import settings


@dataclass
class DenseRetriever:
    """FAISS flat (exact) inner-product search."""

    index: faiss.IndexFlatIP | None = field(default=None, repr=False)
    passage_ids: list[str] = field(default_factory=list)
    dimension: int = settings.model.embedding_dimension
    _embed_fn: object = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _get_embed_fn(self):
        """Lazy-load the embedding function.

        Supports three embedding backends:
        - OpenAI: model names starting with "text-embedding"
        - litellm: model names with "/" (e.g., "gemini/text-embedding-004")
        - sentence-transformers: all other model names (local, free)
        """
        if self._embed_fn is not None:
            return self._embed_fn

        model_name = settings.model.embedding_model

        if model_name.startswith("text-embedding"):
            # OpenAI embeddings
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)

            def _openai_embed(texts: list[str]) -> np.ndarray:
                resp = client.embeddings.create(input=texts, model=model_name)
                vecs = [e.embedding for e in resp.data]
                return np.array(vecs, dtype=np.float32)

            self._embed_fn = _openai_embed

        elif "/" in model_name:
            # litellm-compatible embeddings (Gemini, Cohere, etc.)
            import litellm

            def _litellm_embed(texts: list[str]) -> np.ndarray:
                resp = litellm.embedding(model=model_name, input=texts)
                vecs = [e["embedding"] for e in resp.data]
                return np.array(vecs, dtype=np.float32)

            self._embed_fn = _litellm_embed

        else:
            # sentence-transformers (all-MiniLM-L6-v2, e5-large-v2, etc.)
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(model_name)
            self.dimension = st_model.get_sentence_embedding_dimension()

            def _st_embed(texts: list[str]) -> np.ndarray:
                return st_model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                ).astype(np.float32)

            self._embed_fn = _st_embed

        return self._embed_fn

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
        fn = self._get_embed_fn()
        return fn(texts)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def build_index(self, passages: list[Passage]) -> None:
        """Build FAISS index from passages."""
        texts = [p.content for p in passages]
        ids = [p.id for p in passages]

        logger.info(f"Embedding {len(texts)} passages for dense index...")
        vectors = self.embed(texts)
        self.dimension = vectors.shape[1]

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        self.passage_ids = ids
        logger.info(f"Dense index built: {self.index.ntotal} vectors, dim={self.dimension}")

    def save(self, path: Path) -> None:
        """Save FAISS index and passage IDs to disk."""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        np.save(path / "passage_ids.npy", np.array(self.passage_ids))

    def load(self, path: Path) -> None:
        """Load FAISS index and passage IDs from disk."""
        self.index = faiss.read_index(str(path / "faiss.index"))
        self.passage_ids = np.load(path / "passage_ids.npy").tolist()
        self.dimension = self.index.d
        logger.info(f"Dense index loaded: {self.index.ntotal} vectors")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search the FAISS index.

        Returns list of (passage_id, score) sorted by descending score.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = top_k or settings.retrieval.top_k
        # Over-fetch to account for excluded IDs
        fetch_k = top_k + (len(exclude_ids) if exclude_ids else 0)
        fetch_k = min(fetch_k, self.index.ntotal)

        q_vec = self.embed([query])
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, fetch_k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            pid = self.passage_ids[idx]
            if exclude_ids and pid in exclude_ids:
                continue
            results.append((pid, float(score)))
            if len(results) >= top_k:
                break

        return results
