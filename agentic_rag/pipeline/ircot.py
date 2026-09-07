"""IRCoT — Interleaving Retrieval with Chain-of-Thought (Trivedi et al., ACL 2023).

An external multi-step retrieval method, re-implemented on this project's
retriever and final generation signature so the controlled protocol can be
applied to a method that is not ours. No training; prompting only.

Algorithm (paper §3, Fig. 1):
  1. retrieve K paragraphs with the question;
  2. reason-step: generate the next CoT sentence from the question and the
     paragraphs collected so far;
  3. retrieve-step: use that sentence as the query, add the K new paragraphs
     to the collection (deduplicated);
  4. repeat 2-3 until the CoT sentence contains "answer is" or the maximum
     number of steps is reached; then answer from the collected paragraphs.

Paper defaults, taken from the ACL 2023 text (§3, §4.1, footnotes 4-5):
  - termination: the generated CoT sentence has the "answer is:" string, or
    the maximum number of steps, set to 8, has been reached;
  - K, paragraphs retrieved per step: chosen per dataset from {2, 4, 6, 8}
    on a dev set;
  - the retrieval result is capped at 15 collected paragraphs;
  - retriever: BM25 (Elasticsearch) over Wikipedia; reader: CoT or Direct
    prompting over the collected paragraphs.

What differs here, and is written to the run manifest (ircot_params):
  - retriever: this project's hybrid FAISS + BM25 (RRF) index, the same one
    every other pipeline uses;
  - K defaults to 8, the top of the paper's range, instead of a per-dataset
    tuned value (no dev-set tuning is done for any pipeline here);
  - the collection cap is this project's M (`retrieval.max_passages`, 30)
    rather than 15, so IRCoT sees the same context budget as the
    accumulating pipelines;
  - the reader is `QnAGenerateSignature`, the generation signature shared by
    every pipeline, so the comparison isolates the retrieval strategy;
  - `experiment.llm_call_budget`, if set, bounds the number of reasoning
    steps (see `effective_max_steps`).
"""

from __future__ import annotations

import re

import dspy
from loguru import logger

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.pipeline.base import BasePipeline, PipelineResult
from agentic_rag.retriever.hybrid import HybridRetriever
from agentic_rag.retriever.indexer import DocumentIndexer, Passage
from agentic_rag.signatures.generate import QnAGenerateSignature
from agentic_rag.signatures.ircot import IRCoTStepSignature

# Paper values, kept next to ours so the manifest can state the difference.
PAPER_DEFAULTS = {
    "max_steps": 8,
    "per_step_k": "tuned per dataset from {2, 4, 6, 8}",
    "max_paragraphs": 15,
    "retriever": "BM25 (Elasticsearch)",
    "source": "Trivedi et al., ACL 2023, §3-4.1, footnotes 4-5",
}

# The paper stops on the "answer is:" string; accept the colon-less form too.
ANSWER_PATTERN = re.compile(r"\banswer is\b", re.IGNORECASE)

# Calls a run always makes besides the reasoning steps: the final generation.
IRCOT_FIXED_CALLS = 1


class IRCoTPipeline(BasePipeline):
    """Retrieve → (reason-step → retrieve-step)* → generate."""

    def __init__(self, retriever: HybridRetriever, indexer: DocumentIndexer):
        super().__init__(retriever, indexer)
        self.reasoner = dspy.Predict(IRCoTStepSignature)
        self.generator = dspy.ChainOfThought(QnAGenerateSignature)

    # ------------------------------------------------------------------
    # Parameters as they apply to this run
    # ------------------------------------------------------------------
    @classmethod
    def effective_max_steps(cls) -> int:
        """Reasoning steps allowed: the configured maximum, bounded by the budget.

        Each step is one LLM call and the final generation is one more, so a
        budget of B calls affords B - 1 steps; at least one step is always
        taken so the method remains IRCoT rather than one-step retrieval.
        """
        configured = settings.ircot.max_steps
        budget = settings.experiment.llm_call_budget
        if budget is None:
            return configured
        allowed = max(1, budget - IRCOT_FIXED_CALLS)
        if allowed < configured:
            logger.info(
                f"[IRCoT] Budget mode: {budget} calls -> max_steps {allowed} "
                f"(configured {configured})"
            )
        return min(configured, allowed)

    @classmethod
    def params(cls) -> dict:
        """Parameters of this run next to the paper's, for the manifest."""
        return {
            "max_steps": cls.effective_max_steps(),
            "per_step_k": settings.ircot.per_step_k,
            "max_paragraphs": cls.passage_cap(),
            "retriever": "hybrid FAISS+BM25 (RRF)",
            "reader": "QnAGenerateSignature (shared)",
            "paper": PAPER_DEFAULTS,
        }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, question: str, system_prompt: str = "", **kwargs) -> PipelineResult:
        system_prompt = system_prompt or (
            "You are a helpful knowledge assistant. Answer based on the provided passages."
        )
        k = settings.ircot.per_step_k
        max_steps = self.effective_max_steps()

        collected: list[Passage] = []
        seen: set[str] = set()
        action_history: list[str] = []
        sentences: list[str] = []
        llm_calls = 0

        def _retrieve(query: str, label: str) -> None:
            results = self.retriever.search(query=query, top_k=k, exclude_ids=seen)
            new = [
                p
                for p in self.indexer.get_passages([pid for pid, _ in results])
                if p.id not in seen
            ]
            for p in new:
                seen.add(p.id)
                collected.append(p)
            action_history.append(label)
            logger.info(f"[IRCoT] {label}: +{len(new)} passages, {len(collected)} collected")

        # 1. initial retrieval with the question
        _retrieve(question, "retrieve(q0)")

        # 2-4. interleave reasoning and retrieval
        terminated = False
        for step in range(1, max_steps + 1):
            context = self.format_passages(self.cap_passages(collected))
            with dspy.context(lm=make_lm(settings.model.generate_model)):
                out = self.reasoner(
                    question=question,
                    passages=context,
                    reasoning_so_far=" ".join(sentences),
                )
            llm_calls += 1
            sentence = (out.next_sentence or "").strip()
            sentences.append(sentence)
            action_history.append(f"cot({step})")
            logger.info(f"[IRCoT] cot({step}): {sentence[:120]}")

            if ANSWER_PATTERN.search(sentence):
                terminated = True
                break
            if step < max_steps:
                _retrieve(sentence or question, f"retrieve(cot{step})")

        # 5. answer from the collected paragraphs with the shared reader
        passages = self.cap_passages(collected)
        with dspy.context(lm=make_lm(settings.model.generate_model)):
            gen = self.generator(
                question=question,
                passages=self.format_passages(passages),
                system_prompt=system_prompt,
            )
        llm_calls += 1
        action_history.append("output")

        return PipelineResult(
            question=question,
            answer=gen.answer,
            footnotes=gen.footnotes,
            recommended_questions=gen.recommended_questions,
            passages_used=passages,
            total_passages_retrieved=len(collected),
            retry_count=len(sentences) - 1,
            evaluation_scores=[
                {"step": i + 1, "cot": s, "terminated": terminated and i == len(sentences) - 1}
                for i, s in enumerate(sentences)
            ],
            action_history=action_history,
            llm_calls=llm_calls,
        )
