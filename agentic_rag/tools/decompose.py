"""Query decomposition tool for agentic retrieval."""

from __future__ import annotations

import json
import re

import dspy
from loguru import logger

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.signatures.decompose import DecomposeQuerySignature

_DECOMPOSE_PROMPT = """\
Decompose a complex multi-hop question into simpler sub-questions.

Rules:
1. Each sub-question should be self-contained and answerable independently.
2. Sub-questions should cover different aspects needed to answer the original.
3. For simple, single-hop questions, return just the original question.
4. Order sub-questions by dependency (answer earlier ones first if needed).

Question: {question}

Respond with ONLY valid JSON:
{{"is_multi_hop": true/false, "sub_questions": ["q1", "q2"], "reasoning": "brief explanation"}}"""


def _call_lm_for_decompose(question: str) -> dict:
    """Call LLM directly for manual decomposition."""
    import litellm

    model = settings.model.preprocess_model.replace("dspy/", "")
    prompt = _DECOMPOSE_PROMPT.format(question=question)
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.model.temperature,
        num_retries=settings.model.num_retries,
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


def make_decompose_query():
    """Create a decompose_query tool closure."""

    use_dspy = settings.experiment.enable_dspy

    if use_dspy:
        decomposer = dspy.ChainOfThought(DecomposeQuerySignature)
    else:
        decomposer = None

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
            if use_dspy:
                with dspy.context(lm=make_lm(settings.model.preprocess_model)):
                    result = decomposer(question=question)

                output = {
                    "is_multi_hop": bool(result.is_multi_hop),
                    "sub_questions": list(result.sub_questions),
                    "reasoning": result.reasoning,
                }
            else:
                data = _call_lm_for_decompose(question)
                output = {
                    "is_multi_hop": bool(data.get("is_multi_hop", False)),
                    "sub_questions": data.get("sub_questions", [question]),
                    "reasoning": data.get("reasoning", ""),
                }

            logger.debug(
                f"[Agent:decompose_query] multi_hop={output['is_multi_hop']}, "
                f"sub_qs={len(output['sub_questions'])}"
            )
            return json.dumps(output, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "is_multi_hop": False, "sub_questions": [question]})

    return decompose_query
