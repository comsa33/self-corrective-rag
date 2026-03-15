"""Tests for retriever modules.

Tests BM25 sparse retrieval, RRF fusion, and passage formatting.
Dense retrieval tests require embeddings and are marked as slow.
"""

from __future__ import annotations

import pytest

from src.retriever.hybrid import HybridRetriever
from src.retriever.indexer import Passage
from src.retriever.sparse import SparseRetriever


@pytest.fixture
def passages() -> list[Passage]:
    """Create Passage objects for testing."""
    return [
        Passage(
            id="doc_1",
            title="Python Basics",
            content="Python is a high-level programming language known for its readability.",
        ),
        Passage(
            id="doc_2",
            title="Machine Learning",
            content="Machine learning is a subset of AI that learns from data.",
        ),
        Passage(
            id="doc_3",
            title="RAG Systems",
            content="Retrieval-Augmented Generation combines retrieval with LLM generation.",
        ),
        Passage(
            id="doc_4",
            title="DSPy Framework",
            content="DSPy is a framework for programming foundation models declaratively.",
        ),
        Passage(
            id="doc_5",
            title="Vector Search",
            content="FAISS is a library for efficient similarity search and clustering.",
        ),
    ]


# ---------------------------------------------------------------
# BM25 Sparse Retriever
# ---------------------------------------------------------------
class TestSparseRetriever:
    def test_build_and_search(self, passages):
        retriever = SparseRetriever()
        retriever.build_index(passages)

        results = retriever.search("programming language", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_empty_query(self, passages):
        retriever = SparseRetriever()
        retriever.build_index(passages)
        results = retriever.search("", top_k=3)
        assert isinstance(results, list)

    def test_exclude_ids(self, passages):
        retriever = SparseRetriever()
        retriever.build_index(passages)

        results = retriever.search("Python programming", top_k=5, exclude_ids={"doc_1"})
        result_ids = {r[0] for r in results}
        assert "doc_1" not in result_ids

    def test_returns_scores(self, passages):
        retriever = SparseRetriever()
        retriever.build_index(passages)

        results = retriever.search("machine learning AI", top_k=3)
        for pid, score in results:
            assert isinstance(pid, str)
            assert isinstance(score, int | float)


# ---------------------------------------------------------------
# RRF Fusion (internal method of HybridRetriever)
# ---------------------------------------------------------------
class TestRRF:
    def test_basic_fusion(self):
        """Test RRF fusion via HybridRetriever._rrf_fuse."""
        hybrid = HybridRetriever()
        dense_results = [
            ("a", 0.9),
            ("b", 0.7),
            ("c", 0.5),
        ]
        sparse_results = [
            ("b", 5.0),
            ("d", 3.0),
            ("a", 1.0),
        ]

        fused = hybrid._rrf_fuse(dense_results, sparse_results, top_k=5)
        assert len(fused) <= 5
        fused_ids = [pid for pid, _score in fused]
        # "b" appears in both lists at good positions
        assert "b" in fused_ids[:2]

    def test_single_list(self):
        hybrid = HybridRetriever()
        results = [("x", 1.0)]
        fused = hybrid._rrf_fuse(results, [], top_k=5)
        assert len(fused) == 1
        assert fused[0][0] == "x"

    def test_empty_lists(self):
        hybrid = HybridRetriever()
        fused = hybrid._rrf_fuse([], [], top_k=5)
        assert fused == []

    def test_deduplication(self):
        """Passages appearing in multiple lists should appear once in output."""
        hybrid = HybridRetriever()
        list1 = [("a", 1.0)]
        list2 = [("a", 2.0)]
        fused = hybrid._rrf_fuse(list1, list2, top_k=5)
        ids = [pid for pid, _ in fused]
        assert ids.count("a") == 1
