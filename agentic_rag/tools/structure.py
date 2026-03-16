"""Document structure browsing tool for RLM agentic retrieval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import DocumentIndexer


def make_list_document_sections(indexer: DocumentIndexer):
    """Create a list_document_sections tool closure."""

    def list_document_sections(keyword: str = "") -> str:
        """Browse document table of contents / section structure.

        Args:
            keyword: Search term to match against section titles.
                     Empty string returns all sections.

        Returns:
            JSON string: list of {source, title, passage_count}
        """
        try:
            results = indexer.section_index.search(keyword)
            logger.debug(f"[RLM:list_sections] keyword='{keyword}', results={len(results)}")
            return json.dumps(results[:20], ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return list_document_sections
