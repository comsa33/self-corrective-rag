"""Tool registry for agentic retrieval refinement.

Each tool wraps existing pipeline components (retriever, indexer, evaluator)
via closures. All tools return JSON strings and handle exceptions internally
to prevent REPL sandbox crashes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import dspy

from agentic_rag.tools.calculate import make_calculate
from agentic_rag.tools.decompose import make_decompose_query
from agentic_rag.tools.evaluate import make_evaluate_passages
from agentic_rag.tools.inspect import make_get_passage_detail
from agentic_rag.tools.search import make_search_passages
from agentic_rag.tools.structure import make_list_document_sections
from agentic_rag.tools.terminology import make_get_terminology

if TYPE_CHECKING:
    from agentic_rag.retriever.hybrid import HybridRetriever
    from agentic_rag.retriever.indexer import DocumentIndexer

# Tool name → factory function mapping for ablation
TOOL_REGISTRY = {
    "search": make_search_passages,
    "structure": make_list_document_sections,
    "terminology": make_get_terminology,
    "evaluate": make_evaluate_passages,
    "inspect": make_get_passage_detail,
    "decompose": make_decompose_query,
    "calculate": make_calculate,
}


def create_tools(
    retriever: HybridRetriever,
    indexer: DocumentIndexer,
    evaluator: dspy.Predict,
    enabled_tools: list[str] | None = None,
) -> list[Callable]:
    """Create tool functions as closures over pipeline components.

    Args:
        retriever: Hybrid retriever for search tool.
        indexer: Document indexer for all tools.
        evaluator: DSPy evaluator for evaluate tool.
        enabled_tools: List of tool names to enable. None = all tools.
            Valid names: search, structure, terminology, evaluate, inspect.

    Returns:
        List of callable tools for the ReAct agent.
    """
    tool_names = enabled_tools or list(TOOL_REGISTRY.keys())
    tools = []

    for name in tool_names:
        if name == "search":
            tools.append(make_search_passages(retriever, indexer))
        elif name == "structure":
            tools.append(make_list_document_sections(indexer))
        elif name == "terminology":
            tools.append(make_get_terminology(indexer))
        elif name == "evaluate":
            tools.append(make_evaluate_passages(indexer, evaluator))
        elif name == "inspect":
            tools.append(make_get_passage_detail(indexer))
        elif name == "decompose":
            tools.append(make_decompose_query())
        elif name == "calculate":
            tools.append(make_calculate())

    return tools


__all__ = [
    "TOOL_REGISTRY",
    "create_tools",
    "make_calculate",
    "make_decompose_query",
    "make_evaluate_passages",
    "make_get_passage_detail",
    "make_get_terminology",
    "make_list_document_sections",
    "make_search_passages",
]
