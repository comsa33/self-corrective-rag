"""Terminology mapping tool for agentic retrieval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import DocumentIndexer


def make_get_terminology(indexer: DocumentIndexer):
    """Create a get_terminology tool closure."""

    def get_terminology(user_term: str) -> str:
        """Map user language to document-specific terminology.

        Args:
            user_term: A term from the user's question.

        Returns:
            JSON string: list of matching document terms.
        """
        try:
            results = indexer.term_index.lookup(user_term, top_k=5)
            logger.debug(f"[Agent:get_terminology] '{user_term}' → {results}")
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return get_terminology
