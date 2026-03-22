"""Search tool for agentic retrieval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from agentic_rag.retriever.hybrid import HybridRetriever
    from agentic_rag.retriever.indexer import DocumentIndexer


def make_search_passages(
    retriever: HybridRetriever,
    indexer: DocumentIndexer,
):
    """Create a search_passages tool closure."""

    def search_passages(query: str, top_k: int = 10) -> str:
        """Search internal documents with a query.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return (default: 10).

        Returns:
            JSON string: list of {id, title, content_preview, score}
        """
        try:
            results = retriever.search(query=query, top_k=top_k)
            passages = indexer.get_passages([pid for pid, _ in results])
            score_map = dict(results)

            output = []
            for p in passages:
                output.append(
                    {
                        "id": p.id,
                        "title": p.title,
                        "content_preview": p.content[:100] if p.content else "",
                        "score": round(score_map.get(p.id, 0.0), 4),
                    }
                )

            logger.debug(f"[Agent:search_passages] query='{query}', results={len(output)}")
            return json.dumps(output, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return search_passages
