"""Self-Corrective RAG pipeline — Proposed Method.

Full pipeline implementing all 5 contributions:
  C1: Iterative self-corrective loop + passage accumulation
  C2: 4-dimensional quality assessment
  C3: Targeted query refinement
  C4: 3-way agent routing
  C5: DSPy-based declarative pipeline

Ablation flags in ExperimentSettings allow disabling each contribution
independently for controlled experiments.
"""

from __future__ import annotations

import dspy
from loguru import logger

from config.settings import settings
from src.pipeline.base import BasePipeline, PipelineResult
from src.pipeline.rlm_tools import create_rlm_tools
from src.retriever.hybrid import HybridRetriever
from src.retriever.indexer import DocumentIndexer, Passage
from src.signatures.agents import (
    ClarificationSignature,
    DomainExpertSignature,
    FallbackSignature,
)
from src.signatures.evaluate import EvaluationSignature
from src.signatures.generate import QnAGenerateSignature
from src.signatures.preprocess import (
    HyDEPreprocessSignature,
    PreprocessSignature,
)
from src.signatures.rlm_refinement import RLMRefinementSignature


class SelfCorrectiveRAGPipeline(BasePipeline):
    """Preprocess → Retrieve → 4D Evaluate → Refine Loop → Agent/Generate."""

    def __init__(
        self,
        retriever: HybridRetriever,
        indexer: DocumentIndexer,
        use_hyde: bool = False,
    ):
        super().__init__(retriever, indexer)
        self.use_hyde = use_hyde

        # --- DSPy modules ---
        # Preprocess
        if use_hyde:
            self.preprocessor = dspy.ChainOfThought(HyDEPreprocessSignature)
        else:
            self.preprocessor = dspy.ChainOfThought(PreprocessSignature)

        # Evaluate (Predict — no CoT, to keep evaluation concise)
        self.evaluator = dspy.Predict(EvaluationSignature)

        # Generate (ChainOfThought — explicit reasoning)
        self.generator = dspy.ChainOfThought(QnAGenerateSignature)

        # Agents
        self.clarification_agent = dspy.Predict(ClarificationSignature)
        self.domain_expert_agent = dspy.ChainOfThought(DomainExpertSignature)
        self.fallback_agent = dspy.Predict(FallbackSignature)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        question: str,
        conversation_history: str = "",
        system_prompt: str = "",
    ) -> PipelineResult:
        """Execute the full Self-Corrective RAG pipeline."""
        exp = settings.experiment
        system_prompt = system_prompt or (
            "You are a helpful knowledge assistant. Answer based on the provided passages."
        )

        llm_calls = 0

        # ============================================================
        # STEP 1: Preprocessing (C5: DSPy-based)
        # ============================================================
        with dspy.context(lm=dspy.LM(settings.model.preprocess_model)):
            prep_result = self.preprocessor(
                user_question=question,
                conversation_history=conversation_history,
            )
        llm_calls += 1

        search_query = prep_result.rephrased_question
        search_keywords = list(
            set(prep_result.product_keywords)
            | set(prep_result.keyword_words)
            | set(prep_result.subject_keywords)
        )

        # HyDE: use hypothetical answer as the dense search query
        hyde_query = None
        if self.use_hyde and hasattr(prep_result, "hypothetical_answer"):
            hyde_query = prep_result.hypothetical_answer

        logger.info(
            f"[SelfCorrectiveRAG] Preprocessed: "
            f"query='{search_query}', "
            f"keywords={search_keywords[:5]}, "
            f"topic={prep_result.topic_category}"
        )

        # ============================================================
        # STEP 2-3: Self-Corrective Loop (C1 + C2 + C3) or RLM (C6)
        # ============================================================
        if exp.enable_rlm_refinement:
            # --- C6: RLM-based Agentic Retrieval Refinement ---
            (
                accumulated_passages,
                evaluation_scores,
                action_history,
                final_action,
                rlm_llm_calls,
            ) = self._run_rlm_refinement(search_query, search_keywords, hyde_query)
            llm_calls += rlm_llm_calls
        else:
            # --- Standard for-loop refinement (C1 + C2 + C3) ---
            (
                accumulated_passages,
                evaluation_scores,
                action_history,
                final_action,
                loop_llm_calls,
            ) = self._run_loop_refinement(search_query, search_keywords, hyde_query)
            llm_calls += loop_llm_calls

        # ============================================================
        # STEP 4: Generation or Agent Routing (C4)
        # ============================================================
        context = self.format_passages(accumulated_passages)
        agent_type = None

        if final_action == "route_to_agent" and exp.enable_agent_routing:
            # C4: 3-Way Agent Routing
            answer, footnotes, rec_questions, agent_type = self._route_to_agent(
                search_query, context
            )
            llm_calls += 1
        else:
            # Standard generation
            with dspy.context(lm=dspy.LM(settings.model.generate_model)):
                gen_result = self.generator(
                    question=search_query,
                    passages=context,
                    system_prompt=system_prompt,
                )
            llm_calls += 1
            answer = gen_result.answer
            footnotes = gen_result.footnotes
            rec_questions = gen_result.recommended_questions

        return PipelineResult(
            question=question,
            answer=answer,
            footnotes=footnotes,
            recommended_questions=rec_questions,
            passages_used=accumulated_passages,
            total_passages_retrieved=len(accumulated_passages),
            retry_count=len(action_history) - 1,
            evaluation_scores=evaluation_scores,
            action_history=action_history,
            agent_type=agent_type,
            llm_calls=llm_calls,
        )

    # ------------------------------------------------------------------
    # Standard for-loop refinement (C1 + C2 + C3)
    # ------------------------------------------------------------------
    def _run_loop_refinement(
        self,
        search_query: str,
        search_keywords: list[str],
        hyde_query: str | None,
    ) -> tuple[list[Passage], list[dict], list[str], str, int]:
        """Execute the standard for-loop self-corrective refinement.

        Returns (accumulated_passages, evaluation_scores, action_history,
                 final_action, llm_calls).
        """
        exp = settings.experiment
        eval_cfg = settings.evaluation
        llm_calls = 0

        accumulated_passages: list[Passage] = []
        used_passage_ids: set[str] = set()
        evaluation_scores: list[dict] = []
        action_history: list[str] = []

        max_retry = eval_cfg.max_retry_count if exp.enable_iteration else 0
        final_action = "output"

        for retry in range(max_retry + 1):
            # --- C3: Targeted Query Refinement (on retry) ---
            if retry > 0 and exp.enable_refinement and evaluation_scores:
                prev_eval = evaluation_scores[-1]
                kw_to_add = prev_eval.get("keywords_to_add", [])
                kw_to_remove = prev_eval.get("keywords_to_remove", [])
                suggested = prev_eval.get("suggested_query", "")

                search_keywords = [
                    kw for kw in search_keywords if kw not in kw_to_remove
                ] + kw_to_add

                if suggested:
                    search_query = suggested

                logger.info(
                    f"[SelfCorrectiveRAG] Retry {retry}: "
                    f"refined query='{search_query}', +{kw_to_add}, -{kw_to_remove}"
                )

            # --- Retrieval ---
            combined_query = (f"{search_query} {' '.join(search_keywords)}").strip()
            actual_query = hyde_query if hyde_query else combined_query

            exclude = used_passage_ids if exp.enable_accumulation else set()
            search_results = self.retriever.search(
                query=actual_query,
                exclude_ids=exclude,
            )

            # --- C1: Passage Accumulation ---
            new_passages = self.indexer.get_passages([pid for pid, _ in search_results])

            if exp.enable_accumulation:
                for p in new_passages:
                    if p.id not in used_passage_ids:
                        accumulated_passages.append(p)
                        used_passage_ids.add(p.id)

                # FIFO eviction
                max_p = settings.retrieval.max_passages
                if len(accumulated_passages) > max_p:
                    evicted = accumulated_passages[:-max_p]
                    accumulated_passages = accumulated_passages[-max_p:]
                    for ep in evicted:
                        used_passage_ids.discard(ep.id)
            else:
                accumulated_passages = new_passages
                used_passage_ids = {p.id for p in new_passages}

            context = self.format_passages(accumulated_passages)

            logger.info(
                f"[SelfCorrectiveRAG] Retry {retry}: "
                f"{len(new_passages)} new, "
                f"{len(accumulated_passages)} accumulated"
            )

            # --- C2: 4D Quality Evaluation ---
            if exp.enable_4d_evaluation:
                with dspy.context(lm=dspy.LM(settings.model.evaluate_model)):
                    eval_result = self.evaluator(
                        question=search_query,
                        passages=context,
                        retry_count=retry,
                        max_retry=max_retry,
                    )
                llm_calls += 1

                score_dict = {
                    "retry": retry,
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
                evaluation_scores.append(score_dict)
                action = eval_result.action
            else:
                action = "output"
                evaluation_scores.append({"retry": retry, "action": action})

            action_history.append(action)
            logger.info(
                f"[SelfCorrectiveRAG] Retry {retry}: action={action}, "
                f"score={evaluation_scores[-1].get('total', 'N/A')}"
            )

            if action == "output":
                final_action = "output"
                break
            elif action == "route_to_agent":
                final_action = "route_to_agent"
                break
            # else: "refine" → continue loop

        return accumulated_passages, evaluation_scores, action_history, final_action, llm_calls

    # ------------------------------------------------------------------
    # RLM-based agentic refinement (C6)
    # ------------------------------------------------------------------
    def _run_rlm_refinement(
        self,
        search_query: str,
        search_keywords: list[str],
        hyde_query: str | None,
    ) -> tuple[list[Passage], list[dict], list[str], str, int]:
        """Execute RLM-based agentic retrieval refinement.

        Instead of a fixed for-loop, an RLM agent autonomously decides
        which tools to use (search, browse sections, map terminology,
        evaluate) and in what order to optimize retrieval quality.

        Returns (accumulated_passages, evaluation_scores, action_history,
                 final_action, llm_calls).
        """
        rlm_cfg = settings.rlm

        # Create tools that close over pipeline components
        tools = create_rlm_tools(
            retriever=self.retriever,
            indexer=self.indexer,
            evaluator=self.evaluator,
        )

        # Sub-LM for llm_query() calls inside the REPL (cheap model)
        sub_lm = dspy.LM(settings.model.evaluate_model)

        # Create RLM instance
        rlm = dspy.RLM(
            RLMRefinementSignature,
            max_iterations=rlm_cfg.max_iterations,
            max_llm_calls=rlm_cfg.max_llm_calls,
            max_output_chars=rlm_cfg.max_output_chars,
            verbose=rlm_cfg.verbose,
            tools=tools,
            sub_lm=sub_lm,
        )

        # Execute RLM with main reasoning model
        logger.info(
            f"[SelfCorrectiveRAG:RLM] Starting agentic refinement: "
            f"query='{search_query}', keywords={search_keywords[:5]}"
        )

        with dspy.context(lm=dspy.LM(settings.model.agent_model)):
            result = rlm(
                question=search_query,
                initial_query=hyde_query or search_query,
                initial_keywords=search_keywords,
                quality_threshold=settings.evaluation.quality_threshold,
                max_passages=settings.retrieval.max_passages,
            )

        # Extract results
        passage_ids = result.final_passages or []
        accumulated_passages = self.indexer.get_passages(passage_ids)
        final_action = result.final_action or "output"
        evaluation_scores = result.evaluation_scores or []
        action_history = result.search_log or []
        total_search_calls = result.total_search_calls or 0

        # Estimate LLM calls from trajectory
        trajectory_steps = len(getattr(result, "trajectory", []))
        llm_calls = trajectory_steps + total_search_calls

        logger.info(
            f"[SelfCorrectiveRAG:RLM] Completed: action={final_action}, "
            f"passages={len(accumulated_passages)}, "
            f"search_calls={total_search_calls}, "
            f"trajectory_steps={trajectory_steps}"
        )

        return accumulated_passages, evaluation_scores, action_history, final_action, llm_calls

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
        # Determine agent type from the last evaluation's reasoning
        agent_type = self._classify_agent_type(question)

        with dspy.context(lm=dspy.LM(settings.model.agent_model)):
            if agent_type == "clarification":
                result = self.clarification_agent(question=question, passages=context)
                return (
                    result.clarification_question,
                    "",
                    [],
                    "clarification",
                )

            elif agent_type == "domain_expert":
                result = self.domain_expert_agent(question=question, passages=context)
                return (
                    result.expert_answer,
                    "",
                    [],
                    "domain_expert",
                )

            else:  # fallback
                result = self.fallback_agent(question=question, passages=context)
                return (
                    result.best_effort_answer,
                    f"Limitations: {result.limitations}\nAlternatives: {result.alternatives}",
                    [],
                    "fallback",
                )

    @staticmethod
    def _classify_agent_type(question: str) -> str:
        """Simple heuristic agent classification.

        In production, this could be an LLM-based classifier.
        For experiments, we use keyword heuristics as a baseline,
        or the evaluation step could include agent_type in its output.
        """
        q_lower = question.lower()

        # Technical domain keywords → domain expert
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

        # Ambiguity indicators → clarification
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
