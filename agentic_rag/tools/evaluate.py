"""4D quality evaluation tool for RLM agentic retrieval."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import dspy
from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.pipeline.base import BasePipeline

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import DocumentIndexer


def make_evaluate_passages(
    indexer: DocumentIndexer,
    evaluator: dspy.Predict,
):
    """Create an evaluate_passages tool closure."""

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

    return evaluate_passages
