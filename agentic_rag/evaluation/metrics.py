"""Automated evaluation metrics for RAG experiments.

Implements standard NLP evaluation metrics used across baselines:
  - Accuracy (exact match)
  - F1 score (token-level overlap)
  - ROUGE-L (longest common subsequence)
  - BERTScore (semantic similarity)
  - FactScore-style faithfulness (LLM-based)

All metrics follow the same interface: (prediction, reference) → float.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _strip_footnotes(text: str) -> str:
    """Remove footnote markers like [1], [1, 2], [hotpotqa_Florida Senate]."""
    # Remove bracketed references: [1], [1, 2], [hotpotqa_xxx], [source_id]
    text = re.sub(r"\s*\[[^\]]*\]", "", text)
    return text.strip()


def _normalize_text(text: str) -> str:
    """Lowercase, strip footnotes/punctuation/articles/whitespace for matching."""
    text = _strip_footnotes(text)
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization after normalization."""
    return _normalize_text(text).split()


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def exact_match(prediction: str, reference: str) -> float:
    """Exact match accuracy (0 or 1) after normalization."""
    return float(_normalize_text(prediction) == _normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    common = sum((pred_counter & ref_counter).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score (longest common subsequence)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def bert_score(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, list[float]]:
    """Compute BERTScore for a batch of predictions/references.

    Returns dict with 'precision', 'recall', 'f1' lists.
    Requires: pip install bert-score
    """
    from bert_score import score as bs_score

    P, R, F1 = bs_score(
        predictions,
        references,
        model_type=model_type,
        verbose=False,
    )
    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
    }


def rouge_score_batch(
    predictions: list[str],
    references: list[str],
) -> dict[str, list[float]]:
    """Compute ROUGE scores using the rouge-score library.

    Returns dict with 'rouge1', 'rouge2', 'rougeL' F1 scores.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results: dict[str, list[float]] = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
    }
    for pred, ref in zip(predictions, references, strict=False):
        scores = scorer.score(ref, pred)
        for key in results:
            results[key].append(scores[key].fmeasure)
    return results


# ---------------------------------------------------------------------------
# LLM-based faithfulness (FactScore-style)
# ---------------------------------------------------------------------------
def faithfulness_score(
    answer: str,
    passages: str,
    question: str = "",
) -> dict[str, float]:
    """LLM-based faithfulness evaluation.

    Checks whether the answer is grounded in the provided passages
    (no hallucination). Returns a score between 0 and 1.
    """
    import dspy

    from agentic_rag.config.settings import make_lm, settings

    class FaithfulnessSignature(dspy.Signature):
        """Evaluate whether the answer is faithfully grounded in passages."""

        question: str = dspy.InputField()
        answer: str = dspy.InputField()
        passages: str = dspy.InputField()

        is_faithful: bool = dspy.OutputField(
            desc="True if the answer is fully supported by the passages."
        )
        faithfulness_score: float = dspy.OutputField(
            desc="Score 0.0-1.0: fraction of claims supported by passages."
        )
        unsupported_claims: str = dspy.OutputField(
            desc="List of claims not supported by the passages, if any."
        )

    evaluator = dspy.Predict(FaithfulnessSignature)

    with dspy.context(lm=make_lm(settings.model.evaluate_model)):
        result = evaluator(
            question=question,
            answer=answer,
            passages=passages,
        )

    return {
        "is_faithful": float(result.is_faithful),
        "score": float(result.faithfulness_score),
        "unsupported_claims": result.unsupported_claims,
    }


# ---------------------------------------------------------------------------
# LLM-as-Judge correctness
# ---------------------------------------------------------------------------
def llm_judge_correctness(
    prediction: str,
    reference: str,
    question: str = "",
) -> float:
    """LLM-based correctness evaluation (LLM-as-Judge).

    Uses the evaluate_model to judge whether the prediction is semantically
    correct compared to the reference answer. Returns 1.0 (correct) or
    0.0 (incorrect). Handles cases where the prediction is phrased
    differently but conveys the same answer (e.g., "1572" vs "He died
    in 1572").

    This is the standard approach used in recent agentic RAG papers
    (Test-Time Strategies 2026, Search-o1) as a complement to EM/F1.
    """
    import dspy

    from agentic_rag.config.settings import make_lm, settings

    class CorrectnessJudge(dspy.Signature):
        """Judge whether the predicted answer is correct given the reference.

        The predicted answer may be phrased differently from the reference.
        Focus on whether the core factual content matches, not exact wording.
        A prediction that contains the reference answer is correct.
        A prediction that contradicts the reference is incorrect.
        If the prediction says "I don't know" or fails to answer, it is incorrect.
        """

        question: str = dspy.InputField(desc="The question that was asked.")
        reference_answer: str = dspy.InputField(desc="The gold-standard correct answer.")
        predicted_answer: str = dspy.InputField(desc="The model's predicted answer to judge.")

        is_correct: bool = dspy.OutputField(
            desc="True if the predicted answer is factually correct and matches the reference."
        )

    evaluator = dspy.Predict(CorrectnessJudge)

    try:
        with dspy.context(lm=make_lm(settings.model.evaluate_model)):
            result = evaluator(
                question=question,
                reference_answer=reference,
                predicted_answer=prediction,
            )
        return 1.0 if result.is_correct else 0.0
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return 0.0


def llm_judge_batch(
    predictions: list[str],
    references: list[str],
    questions: list[str] | None = None,
) -> float:
    """Compute mean LLM-as-Judge correctness for a batch.

    Returns accuracy (fraction judged correct).
    """
    questions = questions or [""] * len(predictions)
    scores = []
    for pred, ref, q in zip(predictions, references, questions, strict=False):
        scores.append(llm_judge_correctness(pred, ref, q))
    mean_score = float(np.mean(scores)) if scores else 0.0
    logger.info(f"[LLM-Judge] {sum(scores):.0f}/{len(scores)} correct ({mean_score:.3f})")
    return mean_score


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Container for all evaluation metrics on a single example."""

    question: str
    prediction: str
    reference: str
    exact_match: float = 0.0
    f1: float = 0.0
    rouge_l: float = 0.0
    llm_judge: float = 0.0
    bert_score_f1: float = 0.0
    faithfulness: float = 0.0
    metadata: dict = field(default_factory=dict)


def evaluate_single(
    prediction: str,
    reference: str,
    question: str = "",
    passages: str = "",
    compute_bert_score: bool = False,
    compute_faithfulness: bool = False,
    compute_llm_judge: bool = False,
) -> EvaluationResult:
    """Compute all metrics for a single prediction-reference pair."""
    result = EvaluationResult(
        question=question,
        prediction=prediction,
        reference=reference,
        exact_match=exact_match(prediction, reference),
        f1=token_f1(prediction, reference),
        rouge_l=rouge_l(prediction, reference),
    )

    if compute_llm_judge:
        result.llm_judge = llm_judge_correctness(prediction, reference, question)

    if compute_bert_score:
        bs = bert_score([prediction], [reference])
        result.bert_score_f1 = bs["f1"][0]

    if compute_faithfulness and passages:
        fs = faithfulness_score(prediction, passages, question)
        result.faithfulness = fs["score"]

    return result


def evaluate_batch(
    predictions: list[str],
    references: list[str],
    questions: list[str] | None = None,
    compute_bert_score: bool = True,
    compute_llm_judge: bool = False,
) -> dict[str, float]:
    """Compute aggregate metrics for a batch of predictions.

    Returns mean scores across all examples.
    """
    n = len(predictions)
    questions = questions or [""] * n

    em_scores = [exact_match(p, r) for p, r in zip(predictions, references, strict=False)]
    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references, strict=False)]
    rl_scores = [rouge_l(p, r) for p, r in zip(predictions, references, strict=False)]

    results = {
        "exact_match": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "rouge_l": float(np.mean(rl_scores)),
        "n": n,
    }

    if compute_llm_judge and n > 0:
        results["llm_judge"] = llm_judge_batch(predictions, references, questions)

    if compute_bert_score and n > 0:
        bs = bert_score(predictions, references)
        results["bert_score_f1"] = float(np.mean(bs["f1"]))

    # Also compute ROUGE-1/2 via rouge-score library
    if n > 0:
        rouge = rouge_score_batch(predictions, references)
        results["rouge1"] = float(np.mean(rouge["rouge1"]))
        results["rouge2"] = float(np.mean(rouge["rouge2"]))

    logger.info(
        f"[Metrics] n={n}, EM={results['exact_match']:.3f}, "
        f"F1={results['f1']:.3f}, ROUGE-L={results['rouge_l']:.3f}"
        + (f", LLM-Judge={results['llm_judge']:.3f}" if "llm_judge" in results else "")
    )
    return results
