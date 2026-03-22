"""Manual prompt-based modules for non-DSPy pipeline variant (RQ5).

When enable_dspy=False, these replace dspy.ChainOfThought/Predict modules
with direct litellm calls using hardcoded prompt templates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from loguru import logger

from agentic_rag.config.settings import settings


# ---------------------------------------------------------------------------
# Result containers (mimic DSPy output interface)
# ---------------------------------------------------------------------------
@dataclass
class PreprocessResult:
    rephrased_question: str = ""
    topic_category: str = "General"
    search_keywords: list[str] = field(default_factory=list)
    recommended_questions: list[str] = field(default_factory=list)


@dataclass
class GenerateResult:
    answer: str = ""
    footnotes: str = ""
    recommended_questions: list[str] = field(default_factory=list)


@dataclass
class EvaluateResult:
    relevance: int = 0
    coverage: int = 0
    specificity: int = 0
    sufficiency: int = 0
    total: int = 0
    action: str = "output"
    suggested_query: str = ""
    keywords_to_add: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_PREPROCESS_PROMPT = """\
You are a search query optimizer. Given a user question, produce a JSON object with:
- "rephrased_question": the question rephrased as a standalone query
- "topic_category": broad domain (e.g. science, history, finance)
- "search_keywords": list of 5-10 keywords for retrieval
- "recommended_questions": list of 3 follow-up questions

User question: {question}
Conversation history: {history}

Respond with ONLY valid JSON, no markdown fences."""

_GENERATE_PROMPT = """\
{system_prompt}

Question: {question}

Passages:
{passages}

Answer format rules:
- For factoid questions (who/what/when/where/how many), answer with ONLY the key fact.
- For yes/no + explain questions, answer with verdict AND explanation with numbers.
- For analytical questions, provide finding + supporting data.
- NEVER start with "The answer is" or "Based on".

Respond with ONLY a JSON object:
{{"answer": "your answer", "footnotes": "passage references", "recommended_questions": ["q1", "q2", "q3"]}}"""

_EVALUATE_PROMPT = """\
Evaluate passage quality for answering a question. Score each dimension 0-25:
- Relevance: How relevant are the passages to the question?
- Coverage: Do passages cover all aspects of the question?
- Specificity: Do passages contain specific facts/numbers needed?
- Sufficiency: Are passages sufficient to fully answer the question?

Question: {question}
Passages:
{passages}

Respond with ONLY valid JSON:
{{"relevance": N, "coverage": N, "specificity": N, "sufficiency": N, "total": N, "action": "output|refine", "suggested_query": "refined query if action=refine", "keywords_to_add": ["kw1"]}}"""


# ---------------------------------------------------------------------------
# Manual modules (callable, same interface as DSPy modules)
# ---------------------------------------------------------------------------
def _call_lm(model: str, prompt: str) -> str:
    """Call LLM via litellm and return raw text."""
    import litellm

    response = litellm.completion(
        model=model.replace("dspy/", ""),
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.model.temperature,
        num_retries=settings.model.num_retries,
    )
    return response.choices[0].message.content.strip()


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"[ManualPrompt] Failed to parse JSON: {text[:200]}")
        return {}


class ManualPreprocessor:
    """Manual prompt-based preprocessor replacing dspy.ChainOfThought(PreprocessSignature)."""

    def __call__(self, user_question: str, conversation_history: str = "") -> PreprocessResult:
        prompt = _PREPROCESS_PROMPT.format(question=user_question, history=conversation_history)
        raw = _call_lm(settings.model.preprocess_model, prompt)
        data = _parse_json(raw)
        return PreprocessResult(
            rephrased_question=data.get("rephrased_question", user_question),
            topic_category=data.get("topic_category", "General"),
            search_keywords=data.get("search_keywords", []),
            recommended_questions=data.get("recommended_questions", []),
        )


class ManualGenerator:
    """Manual prompt-based generator replacing dspy.ChainOfThought(QnAGenerateSignature)."""

    def __call__(self, question: str, passages: str, system_prompt: str = "") -> GenerateResult:
        prompt = _GENERATE_PROMPT.format(
            system_prompt=system_prompt or "You are a helpful knowledge assistant.",
            question=question,
            passages=passages,
        )
        raw = _call_lm(settings.model.generate_model, prompt)
        data = _parse_json(raw)

        if not data:
            # Fallback: treat entire response as answer
            return GenerateResult(answer=raw, footnotes="", recommended_questions=[])

        return GenerateResult(
            answer=data.get("answer", raw),
            footnotes=data.get("footnotes", ""),
            recommended_questions=data.get("recommended_questions", []),
        )


class ManualEvaluator:
    """Manual prompt-based evaluator replacing dspy.Predict(EvaluationSignature)."""

    def __call__(self, question: str, passages: str, **kwargs) -> EvaluateResult:
        prompt = _EVALUATE_PROMPT.format(question=question, passages=passages)
        raw = _call_lm(settings.model.evaluate_model, prompt)
        data = _parse_json(raw)
        return EvaluateResult(
            relevance=data.get("relevance", 0),
            coverage=data.get("coverage", 0),
            specificity=data.get("specificity", 0),
            sufficiency=data.get("sufficiency", 0),
            total=data.get("total", 0),
            action=data.get("action", "output"),
            suggested_query=data.get("suggested_query", ""),
            keywords_to_add=data.get("keywords_to_add", []),
        )
