"""CRAG Replica pipeline — Baseline 2.

Reproduces the core CRAG (Corrective RAG) mechanism from:
  Yan et al., "Corrective Retrieval Augmented Generation" (2024)

Key differences from our proposed method:
  - Binary relevance evaluation (Correct/Incorrect/Ambiguous) instead of 4D
  - Single-pass correction (no iterative loop)
  - Web search fallback instead of targeted query refinement
  - Knowledge refinement via strip decomposition
  - No agent routing

Note: Original CRAG uses a fine-tuned T5-large evaluator.
We use LLM-based evaluation for fair comparison (same compute class).
"""

from __future__ import annotations

import dspy
from loguru import logger

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.pipeline.base import BasePipeline, PipelineResult
from agentic_rag.retriever.hybrid import HybridRetriever
from agentic_rag.retriever.indexer import DocumentIndexer, Passage
from agentic_rag.signatures.generate import QnAGenerateSignature


# ---------------------------------------------------------------------------
# CRAG-specific DSPy Signatures
# ---------------------------------------------------------------------------
class CRAGEvaluationSignature(dspy.Signature):
    """Binary relevance evaluation for retrieved documents (CRAG-style).

    Evaluate whether the retrieved passages are relevant to the question.
    This is a simplified, single-dimension evaluation compared to our
    proposed 4D assessment.

    Decision:
      - "correct":   passages are clearly relevant → use them directly
      - "incorrect": passages are irrelevant → trigger web search fallback
      - "ambiguous": uncertain relevance → refine + supplement with web search
    """

    question: str = dspy.InputField(desc="The user's question.")
    passages: str = dspy.InputField(desc="Retrieved passages as context string.")

    relevance_judgment: str = dspy.OutputField(desc='One of: "correct", "incorrect", "ambiguous".')
    confidence: float = dspy.OutputField(desc="Confidence score (0.0-1.0) in the judgment.")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the relevance judgment.")


class KnowledgeRefinementSignature(dspy.Signature):
    """CRAG Knowledge Refinement — decompose and filter document strips.

    Decomposes each passage into fine-grained "strips" (sentences/clauses),
    evaluates each strip's relevance, and reassembles only the relevant ones.
    This produces a cleaner, more focused context for generation.
    """

    question: str = dspy.InputField(desc="The user's question.")
    passage: str = dspy.InputField(desc="A single passage to refine.")

    refined_content: str = dspy.OutputField(
        desc="Only the relevant sentences/strips from the passage, concatenated."
    )
    relevance_ratio: float = dspy.OutputField(
        desc="Fraction of original content deemed relevant (0.0-1.0)."
    )


class QueryRewriteSignature(dspy.Signature):
    """Rewrite a query for improved retrieval when initial passages are insufficient.

    Given a question and reasoning about why initial passages were irrelevant,
    produce a rewritten query that targets different aspects or uses alternative
    terms to find relevant passages in the same corpus.
    """

    question: str = dspy.InputField(desc="The user's question.")
    reasoning: str = dspy.InputField(desc="Why the initial retrieval was insufficient.")

    rewritten_query: str = dspy.OutputField(
        desc="An improved search query targeting different aspects or alternative terms."
    )


class CRAGReplicaPipeline(BasePipeline):
    """CRAG reproduction: Evaluate → Correct (1-pass) → Generate.

    Implements the three CRAG actions:
      1. Correct:   passages relevant → knowledge refinement → generate
      2. Incorrect: passages irrelevant → web search → generate
      3. Ambiguous:  combine refined passages + web search → generate
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        indexer: DocumentIndexer,
        use_real_web_search: bool = False,
    ):
        super().__init__(retriever, indexer)
        self.use_real_web_search = use_real_web_search

        # DSPy modules
        self.evaluator = dspy.Predict(CRAGEvaluationSignature)
        self.refiner = dspy.Predict(KnowledgeRefinementSignature)
        self.query_rewriter = dspy.Predict(QueryRewriteSignature)
        self.generator = dspy.ChainOfThought(QnAGenerateSignature)

    def run(
        self,
        question: str,
        system_prompt: str = "",
        top_k: int | None = None,
    ) -> PipelineResult:
        """Execute CRAG pipeline: Retrieve → Evaluate → Correct → Generate."""
        top_k = top_k or settings.retrieval.top_k
        system_prompt = system_prompt or (
            "You are a helpful knowledge assistant. Answer based on the provided context."
        )
        llm_calls = 0

        # ==============================================================
        # Step 1: Retrieval (same as proposed method for fair comparison)
        # ==============================================================
        search_results = self.retriever.search(query=question, top_k=top_k)
        passage_ids = [pid for pid, _ in search_results]
        passages = self.indexer.get_passages(passage_ids)

        logger.info(f"[CRAG] Retrieved {len(passages)} passages")

        # ==============================================================
        # Step 2: Retrieval Evaluation (binary judgment)
        # ==============================================================
        context = self.format_passages(passages)

        with dspy.context(lm=make_lm(settings.model.evaluate_model)):
            eval_result = self.evaluator(
                question=question,
                passages=context,
            )
        llm_calls += 1

        judgment = eval_result.relevance_judgment.lower().strip()
        logger.info(f"[CRAG] Judgment: {judgment} (confidence={eval_result.confidence})")

        # ==============================================================
        # Step 3: Corrective Action (1-pass, no iteration)
        # ==============================================================
        final_context: str
        action_history = [judgment]
        eval_scores = [
            {
                "judgment": judgment,
                "confidence": float(eval_result.confidence),
                "reasoning": eval_result.reasoning,
            }
        ]

        if judgment == "correct":
            # Knowledge refinement on retrieved passages
            final_context = self._refine_passages(question, passages)
            llm_calls += len(passages)

        elif judgment == "incorrect":
            # Re-retrieval with rewritten query (same corpus, no LLM memory)
            re_passages = self._rewrite_and_retrieve(
                question, eval_result.reasoning, exclude_ids=set(passage_ids)
            )
            llm_calls += 1
            if re_passages:
                passages = re_passages  # replace with new passages
            final_context = self.format_passages(passages)

        else:  # ambiguous
            # Combine refined passages + re-retrieved passages
            refined = self._refine_passages(question, passages)
            llm_calls += len(passages)
            re_passages = self._rewrite_and_retrieve(
                question, eval_result.reasoning, exclude_ids=set(passage_ids)
            )
            llm_calls += 1
            re_context = self.format_passages(re_passages) if re_passages else ""
            final_context = f"{refined}\n\n{re_context}".strip()

        # ==============================================================
        # Step 4: Generation
        # ==============================================================
        with dspy.context(lm=make_lm(settings.model.generate_model)):
            gen_result = self.generator(
                question=question,
                passages=final_context,
                system_prompt=system_prompt,
            )
        llm_calls += 1

        return PipelineResult(
            question=question,
            answer=gen_result.answer,
            footnotes=gen_result.footnotes,
            recommended_questions=gen_result.recommended_questions,
            passages_used=passages,
            total_passages_retrieved=len(passages),
            retry_count=0,  # CRAG is always single-pass
            evaluation_scores=eval_scores,
            action_history=action_history,
            llm_calls=llm_calls,
        )

    # ------------------------------------------------------------------
    # Knowledge Refinement (CRAG core mechanism)
    # ------------------------------------------------------------------
    def _refine_passages(self, question: str, passages: list[Passage]) -> str:
        """Decompose passages into strips and keep only relevant ones."""
        refined_parts: list[str] = []

        with dspy.context(lm=make_lm(settings.model.evaluate_model)):
            for p in passages:
                result = self.refiner(
                    question=question,
                    passage=p.content,
                )
                if result.refined_content.strip():
                    refined_parts.append(f"[{p.id}, {p.title}]\n{result.refined_content}")

        if not refined_parts:
            # Fallback: use original passages if refinement removes everything
            return self.format_passages(passages)

        return "\n\n".join(refined_parts)

    # ------------------------------------------------------------------
    # Re-retrieval with Query Rewrite
    # ------------------------------------------------------------------
    def _rewrite_and_retrieve(
        self,
        question: str,
        reasoning: str,
        exclude_ids: set[str] | None = None,
    ) -> list[Passage]:
        """Rewrite the query and re-retrieve from the same corpus.

        Instead of web search (which uses LLM parametric knowledge and creates
        an unfair advantage), this rewrites the query based on the evaluator's
        reasoning and retrieves again from the same corpus.
        """
        with dspy.context(lm=make_lm(settings.model.evaluate_model)):
            rewrite_result = self.query_rewriter(
                question=question,
                reasoning=reasoning,
            )

        new_query = rewrite_result.rewritten_query
        logger.info(f"[CRAG] Re-retrieval with rewritten query: '{new_query}'")

        search_results = self.retriever.search(
            query=new_query,
            exclude_ids=exclude_ids or set(),
        )
        return self.indexer.get_passages([pid for pid, _ in search_results])
