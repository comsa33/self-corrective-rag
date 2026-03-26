"""Quality evaluation tool for agentic retrieval (4D or 1D mode)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import dspy
from loguru import logger

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.pipeline.base import BasePipeline

if TYPE_CHECKING:
    from agentic_rag.retriever.indexer import DocumentIndexer


_DOCSTRING_4D = """Run multi-dimensional quality evaluation on selected passages.

Evaluates passages across 4 quality dimensions: relevance, coverage,
specificity, and sufficiency.  Returns per-dimension scores plus
targeted refinement feedback when quality is insufficient.

Args:
    question: The user's question.
    passage_ids_json: JSON array of passage IDs to evaluate,
                      e.g. '["id1", "id2", "id3"]'
    retry_count: Current retry iteration (0-based). Higher values
                 apply progressive leniency to the quality threshold.

Returns:
    JSON string: {relevance, coverage, specificity, sufficiency,
                  total, action, reasoning, keywords_to_add,
                  keywords_to_remove, suggested_query}
"""

_DOCSTRING_1D = """Run quality evaluation on selected passages.

Returns a single overall quality score (0-100) and a next-action
decision ("output" or "refine") without per-dimension breakdown,
plus targeted refinement feedback when quality is insufficient.

Args:
    question: The user's question.
    passage_ids_json: JSON array of passage IDs to evaluate,
                      e.g. '["id1", "id2", "id3"]'
    retry_count: Current retry iteration (0-based). Higher values
                 apply progressive leniency to the quality threshold.

Returns:
    JSON string: {total, action, reasoning, keywords_to_add,
                  keywords_to_remove, suggested_query}
"""


def make_evaluate_passages(
    indexer: DocumentIndexer,
    evaluator: dspy.Predict,
):
    """Create an evaluate_passages tool closure."""

    is_4d = settings.experiment.enable_4d_evaluation
    use_dspy = settings.experiment.enable_dspy

    def evaluate_passages(question: str, passage_ids_json: str, retry_count: int = 0) -> str:
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

            if use_dspy:
                with dspy.context(lm=make_lm(settings.model.evaluate_model)):
                    eval_result = evaluator(
                        question=question,
                        passages=context,
                        retry_count=retry_count,
                        max_retry=settings.evaluation.max_retry_count,
                    )
            else:
                eval_result = evaluator(
                    question=question,
                    passages=context,
                    retry_count=retry_count,
                    max_retry=settings.evaluation.max_retry_count,
                )

            # Build score dict — branch on DSPy vs Manual attribute names
            if use_dspy:
                # DSPy Predict result: attributes match signature field names
                if is_4d:
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
                else:
                    score_dict = {
                        "total": int(eval_result.total_score),
                        "action": eval_result.action,
                        "reasoning": eval_result.reasoning,
                        "keywords_to_add": eval_result.keywords_to_add,
                        "keywords_to_remove": eval_result.keywords_to_remove,
                        "suggested_query": eval_result.suggested_query,
                    }
            else:
                # ManualEvaluator result: dataclass with short attribute names
                score_dict = {
                    "relevance": int(eval_result.relevance),
                    "coverage": int(eval_result.coverage),
                    "specificity": int(eval_result.specificity),
                    "sufficiency": int(eval_result.sufficiency),
                    "total": int(eval_result.total),
                    "action": eval_result.action,
                    "reasoning": eval_result.reasoning,
                    "keywords_to_add": eval_result.keywords_to_add,
                    "keywords_to_remove": eval_result.keywords_to_remove,
                    "suggested_query": eval_result.suggested_query,
                }

            logger.debug(
                f"[Agent:evaluate_passages] total={score_dict['total']}, "
                f"action={score_dict['action']}"
            )
            return json.dumps(score_dict, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "total": 0, "action": "refine"})

    # Set docstring based on evaluation mode so ReAct agent sees accurate tool description
    evaluate_passages.__doc__ = _DOCSTRING_4D if is_4d else _DOCSTRING_1D

    return evaluate_passages
