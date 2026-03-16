"""Backward-compatibility shim — use agentic_rag.tools instead."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import dspy

from agentic_rag.tools import create_tools

if TYPE_CHECKING:
    from agentic_rag.retriever.hybrid import HybridRetriever
    from agentic_rag.retriever.indexer import DocumentIndexer


def create_rlm_tools(
    retriever: HybridRetriever,
    indexer: DocumentIndexer,
    evaluator: dspy.Predict,
) -> list[Callable]:
    """Create RLM tools. Delegates to agentic_rag.tools.create_tools()."""
    return create_tools(retriever, indexer, evaluator)
