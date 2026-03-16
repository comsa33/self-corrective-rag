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

from agentic_rag.config.settings import settings
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


class WebSearchSimulatorSignature(dspy.Signature):
    """Simulate web search fallback for CRAG.

    In the original CRAG, web search provides supplementary knowledge
    when retrieved passages are insufficient. For controlled experiments,
    we simulate this by generating plausible web search results from LLM
    parametric knowledge, ensuring reproducibility.

    For real web search experiments, this can be replaced with actual
    search API calls (e.g., Tavily, Serper).
    """

    question: str = dspy.InputField(desc="The user's question.")

    web_knowledge: str = dspy.OutputField(
        desc="Simulated web search results — factual knowledge about the topic."
    )
    sources: str = dspy.OutputField(desc="Hypothetical source descriptions for the web results.")


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
        self.web_searcher = dspy.Predict(WebSearchSimulatorSignature)
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

        with dspy.context(lm=dspy.LM(settings.model.evaluate_model)):
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
            # Web search fallback (discard retrieved passages)
            web_context = self._web_search(question)
            llm_calls += 1
            final_context = web_context

        else:  # ambiguous
            # Combine refined passages + web search
            refined = self._refine_passages(question, passages)
            llm_calls += len(passages)
            web_context = self._web_search(question)
            llm_calls += 1
            final_context = f"{refined}\n\n--- Web Search Results ---\n{web_context}"

        # ==============================================================
        # Step 4: Generation
        # ==============================================================
        with dspy.context(lm=dspy.LM(settings.model.generate_model)):
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

        with dspy.context(lm=dspy.LM(settings.model.evaluate_model)):
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
    # Web Search Fallback
    # ------------------------------------------------------------------
    def _web_search(self, question: str) -> str:
        """Simulate or perform web search for supplementary knowledge."""
        if self.use_real_web_search:
            return self._real_web_search(question)

        # Simulated web search via LLM parametric knowledge
        with dspy.context(lm=dspy.LM(settings.model.agent_model)):
            result = self.web_searcher(question=question)

        return f"{result.web_knowledge}\n\nSources: {result.sources}"

    @staticmethod
    def _real_web_search(question: str) -> str:
        """Placeholder for real web search API integration.

        To use, install and configure a search provider:
          - Tavily: pip install tavily-python
          - Serper: pip install google-serper
        """
        raise NotImplementedError(
            "Real web search not configured. "
            "Set use_real_web_search=False or implement a search provider."
        )
