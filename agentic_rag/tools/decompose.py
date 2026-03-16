"""Query decomposition tool for RLM agentic retrieval."""

from __future__ import annotations

import json

import dspy
from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.signatures.decompose import DecomposeQuerySignature


def make_decompose_query():
    """Create a decompose_query tool closure."""

    decomposer = dspy.ChainOfThought(DecomposeQuerySignature)

    def decompose_query(question: str) -> str:
        """Decompose a complex multi-hop question into simpler sub-questions.

        Use this when a question requires multiple reasoning steps or
        involves relationships between entities (e.g. "Who is the spouse
        of the director of Film X?").

        Args:
            question: The complex question to decompose.

        Returns:
            JSON string: {is_multi_hop, sub_questions, reasoning}
        """
        try:
            with dspy.context(lm=dspy.LM(settings.model.preprocess_model)):
                result = decomposer(question=question)

            output = {
                "is_multi_hop": bool(result.is_multi_hop),
                "sub_questions": list(result.sub_questions),
                "reasoning": result.reasoning,
            }

            logger.debug(
                f"[RLM:decompose_query] multi_hop={output['is_multi_hop']}, "
                f"sub_qs={len(output['sub_questions'])}"
            )
            return json.dumps(output, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "is_multi_hop": False, "sub_questions": [question]})

    return decompose_query
