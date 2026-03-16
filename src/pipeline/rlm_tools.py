"""RLM tool factory for agentic retrieval refinement (C6).

Creates callable tool functions that are injected into the dspy.RLM
sandbox. Each tool wraps existing pipeline components (retriever, indexer,
evaluator) via closures, keeping the RLM agent's interface simple.

All tools return JSON strings and catch exceptions internally to prevent
REPL crashes in the sandbox.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import dspy
from loguru import logger

from config.settings import settings
from src.pipeline.base import BasePipeline

if TYPE_CHECKING:
    from src.retriever.hybrid import HybridRetriever
    from src.retriever.indexer import DocumentIndexer


def create_rlm_tools(
    retriever: HybridRetriever,
    indexer: DocumentIndexer,
    evaluator: dspy.Predict,
) -> list[Callable]:
    """Create tool functions as closures over pipeline components.

    Returns a list of 5 callable tools for the RLM sandbox:
    - search_passages
    - list_document_sections
    - get_terminology
    - evaluate_passages
    - get_passage_detail
    """

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
                        "content_preview": p.content[:200]
                        + ("..." if len(p.content) > 200 else ""),
                        "score": round(score_map.get(p.id, 0.0), 4),
                    }
                )

            logger.debug(f"[RLM:search_passages] query='{query}', results={len(output)}")
            return json.dumps(output, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

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
            return json.dumps(results[:20], ensure_ascii=False)  # cap at 20
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_terminology(user_term: str) -> str:
        """Map user language to document-specific terminology.

        Args:
            user_term: A term from the user's question.

        Returns:
            JSON string: list of matching document terms.
        """
        try:
            results = indexer.term_index.lookup(user_term, top_k=5)
            logger.debug(f"[RLM:get_terminology] '{user_term}' → {results}")
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def evaluate_passages(question: str, passage_ids_json: str) -> str:
        """Run 4D quality evaluation on selected passages.

        Args:
            question: The user's question.
            passage_ids_json: JSON array of passage IDs to evaluate,
                              e.g. '["id1", "id2", "id3"]'

        Returns:
            JSON string: {relevance, coverage, specificity, sufficiency,
                          total, action, reasoning, keywords_to_add,
                          keywords_to_remove, suggested_query}
        """
        try:
            passage_ids = json.loads(passage_ids_json)
            passages = indexer.get_passages(passage_ids)

            if not passages:
                return json.dumps(
                    {
                        "error": "No passages found for the given IDs",
                        "total": 0,
                        "action": "refine",
                    }
                )

            context = BasePipeline.format_passages(passages)

            with dspy.context(lm=dspy.LM(settings.model.evaluate_model)):
                eval_result = evaluator(
                    question=question,
                    passages=context,
                    retry_count=0,
                    max_retry=settings.evaluation.max_retry_count,
                )

            score_dict = {
                "relevance": int(eval_result.relevance_score),
                "coverage": int(eval_result.coverage_score),
                "specificity": int(eval_result.specificity_score),
                "sufficiency": int(eval_result.sufficiency_score),
                "total": int(eval_result.total_score),
                "action": eval_result.action,
                "reasoning": eval_result.reasoning,
                "keywords_to_add": eval_result.keywords_to_add,
                "keywords_to_remove": eval_result.keywords_to_remove,
                "suggested_query": eval_result.suggested_query,
            }

            logger.debug(
                f"[RLM:evaluate_passages] total={score_dict['total']}, "
                f"action={score_dict['action']}"
            )
            return json.dumps(score_dict, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "total": 0, "action": "refine"})

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

    return [
        search_passages,
        list_document_sections,
        get_terminology,
        evaluate_passages,
        get_passage_detail,
    ]
