"""Shared logic for self-corrective RAG pipelines.

Preprocessing, generation, and agent routing are identical across
the agentic (ReAct) and loop (for-loop) pipeline variants. This mixin
keeps that logic in one place.
"""

from __future__ import annotations

import dspy
from loguru import logger

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.pipeline.base import BasePipeline, PipelineResult
from agentic_rag.retriever.hybrid import HybridRetriever
from agentic_rag.retriever.indexer import DocumentIndexer, Passage
from agentic_rag.signatures.agents import (
    ClarificationSignature,
    DomainExpertSignature,
    FallbackSignature,
)
from agentic_rag.signatures.evaluate import Evaluation1DSignature, EvaluationSignature
from agentic_rag.signatures.generate import QnAGenerateSignature
from agentic_rag.signatures.preprocess import (
    HyDEPreprocessSignature,
    PreprocessSignature,
)


class SelfCorrectiveMixin(BasePipeline):
    """Common init, preprocessing, generation, and agent routing."""

    def __init__(
        self,
        retriever: HybridRetriever,
        indexer: DocumentIndexer,
        use_hyde: bool = False,
    ):
        super().__init__(retriever, indexer)
        self.use_hyde = use_hyde

        # --- DSPy modules ---
        if use_hyde:
            self.preprocessor = dspy.ChainOfThought(HyDEPreprocessSignature)
        else:
            self.preprocessor = dspy.ChainOfThought(PreprocessSignature)

        if settings.experiment.enable_4d_evaluation:
            self.evaluator = dspy.Predict(EvaluationSignature)
        else:
            self.evaluator = dspy.Predict(Evaluation1DSignature)
        self.generator = dspy.ChainOfThought(QnAGenerateSignature)

        # Agents
        self.clarification_agent = dspy.Predict(ClarificationSignature)
        self.domain_expert_agent = dspy.ChainOfThought(DomainExpertSignature)
        self.fallback_agent = dspy.Predict(FallbackSignature)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(
        self,
        question: str,
        conversation_history: str = "",
    ) -> tuple[str, list[str], str | None, str]:
        """Run preprocessing and return (query, keywords, hyde_query, topic).

        Returns:
            search_query: Rephrased standalone question.
            search_keywords: Union of product/keyword/subject keywords.
            hyde_query: Hypothetical answer for dense retrieval (or None).
            topic: Topic category string.
        """
        with dspy.context(lm=make_lm(settings.model.preprocess_model)):
            prep_result = self.preprocessor(
                user_question=question,
                conversation_history=conversation_history,
            )

        search_query = prep_result.rephrased_question
        search_keywords = list(set(prep_result.search_keywords))

        hyde_query = None
        if self.use_hyde and hasattr(prep_result, "hypothetical_answer"):
            hyde_query = prep_result.hypothetical_answer

        logger.info(
            f"[Preprocess] query='{search_query}', "
            f"keywords={search_keywords[:5]}, topic={prep_result.topic_category}"
        )

        return search_query, search_keywords, hyde_query, prep_result.topic_category

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(
        self,
        question: str,
        passages: list[Passage],
        system_prompt: str,
    ) -> tuple[str, str, list[str]]:
        """Run answer generation. Returns (answer, footnotes, rec_questions)."""
        context = self.format_passages(passages)
        with dspy.context(lm=make_lm(settings.model.generate_model)):
            gen_result = self.generator(
                question=question,
                passages=context,
                system_prompt=system_prompt,
            )
        return gen_result.answer, gen_result.footnotes, gen_result.recommended_questions

    # ------------------------------------------------------------------
    # Agent routing (C4)
    # ------------------------------------------------------------------
    def _route_to_agent(
        self,
        question: str,
        context: str,
    ) -> tuple[str, str, list[str], str]:
        """Route to the appropriate agent based on question characteristics.

        Returns (answer, footnotes, recommended_questions, agent_type).
        """
        agent_type = self._classify_agent_type(question)

        with dspy.context(lm=make_lm(settings.model.agent_model)):
            if agent_type == "clarification":
                result = self.clarification_agent(question=question, passages=context)
                return (result.clarification_question, "", [], "clarification")

            elif agent_type == "domain_expert":
                result = self.domain_expert_agent(question=question, passages=context)
                return (result.expert_answer, "", [], "domain_expert")

            else:
                result = self.fallback_agent(question=question, passages=context)
                return (
                    result.best_effort_answer,
                    f"Limitations: {result.limitations}\nAlternatives: {result.alternatives}",
                    [],
                    "fallback",
                )

    @staticmethod
    def _classify_agent_type(question: str) -> str:
        """Simple heuristic agent classification."""
        q_lower = question.lower()

        tech_keywords = [
            "api",
            "코드",
            "code",
            "아키텍처",
            "architecture",
            "devops",
            "함수",
            "function",
            "설정",
            "config",
            "구현",
            "implement",
            "에러",
            "error",
            "서버",
            "server",
        ]
        if any(kw in q_lower for kw in tech_keywords):
            return "domain_expert"

        ambiguity_keywords = [
            "어떻게",
            "뭐가",
            "어떤",
            "what kind",
            "which",
            "차이",
            "difference",
            "비교",
            "compare",
        ]
        if any(kw in q_lower for kw in ambiguity_keywords):
            return "clarification"

        return "fallback"

    # ------------------------------------------------------------------
    # Build PipelineResult (shared by both variants)
    # ------------------------------------------------------------------
    def _build_result(
        self,
        question: str,
        search_query: str,
        passages: list[Passage],
        evaluation_scores: list[dict],
        action_history: list[str],
        final_action: str,
        llm_calls: int,
        system_prompt: str,
    ) -> PipelineResult:
        """Generate answer (or route to agent) and build final result."""
        exp = settings.experiment
        agent_type = None

        if final_action == "route_to_agent" and exp.enable_agent_routing and not passages:
            # Only route to agent if we truly have no passages to work with
            context = self.format_passages(passages)
            answer, footnotes, rec_questions, agent_type = self._route_to_agent(question, context)
            llm_calls += 1
        else:
            # Generate from available passages (even if below threshold)
            # Use original question (not search_query) for concise factoid answers
            if final_action == "route_to_agent" and passages:
                logger.info(
                    f"[Pipeline] route_to_agent with {len(passages)} passages → "
                    f"falling back to generation"
                )
            answer, footnotes, rec_questions = self._generate(question, passages, system_prompt)
            llm_calls += 1

        return PipelineResult(
            question=question,
            answer=answer,
            footnotes=footnotes,
            recommended_questions=rec_questions,
            passages_used=passages,
            total_passages_retrieved=len(passages),
            retry_count=len(action_history) - 1,
            evaluation_scores=evaluation_scores,
            action_history=action_history,
            agent_type=agent_type,
            llm_calls=llm_calls,
        )
