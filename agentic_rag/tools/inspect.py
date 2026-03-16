"""Passage inspection tool for agentic retrieval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import DocumentIndexer


def make_get_passage_detail(indexer: DocumentIndexer):
    """Create a get_passage_detail tool closure."""

    def get_passage_detail(passage_id: str) -> str:
        """Read the full content of a specific passage.

        Args:
            passage_id: The ID of the passage to retrieve.

        Returns:
            JSON string: {id, title, content, source} or error.
        """
        try:
            p = indexer.get_passage(passage_id)
            if p is None:
                return json.dumps({"error": f"Passage '{passage_id}' not found"})
            logger.debug(f"[Agent:get_passage_detail] id={passage_id}")
            return json.dumps(
                {
                    "id": p.id,
                    "title": p.title,
                    "content": p.content,
                    "source": p.source,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    return get_passage_detail
