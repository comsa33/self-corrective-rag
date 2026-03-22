"""Passage inspection tool for agentic retrieval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import DocumentIndexer


def make_get_passage_detail(indexer: DocumentIndexer):
    """Create a get_passage_detail tool closure."""

    def get_passage_detail(passage_id: str, include_adjacent: bool = False) -> str:
        """Read the full content of a specific passage.

        Use this after search_passages to read the complete text of
        promising passages. Search only returns short previews.

        Args:
            passage_id: The ID of the passage to retrieve.
            include_adjacent: If True, also return adjacent passages
                from the same document source for context.

        Returns:
            JSON string: {id, title, content, source, adjacent?} or error.
        """
        try:
            p = indexer.get_passage(passage_id)
            if p is None:
                return json.dumps({"error": f"Passage '{passage_id}' not found"})
            logger.debug(f"[Agent:get_passage_detail] id={passage_id}, adjacent={include_adjacent}")
            result: dict = {
                "id": p.id,
                "title": p.title,
                "content": p.content,
                "source": p.source,
            }

            if include_adjacent:
                # Find other passages from the same source
                adjacent = [
                    {"id": ap.id, "title": ap.title, "content_preview": ap.content[:100]}
                    for ap in indexer.passages
                    if ap.source == p.source and ap.id != p.id
                ]
                result["adjacent_passages"] = adjacent[:5]

            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return get_passage_detail
