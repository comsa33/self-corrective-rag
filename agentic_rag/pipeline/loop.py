"""Loop-based Self-Corrective RAG pipeline — Ablation Baseline.

Fixed for-loop refinement implementing C1-C5:
  C1: Iterative self-corrective loop + passage accumulation
  C2: 4-dimensional quality assessment
  C3: Targeted query refinement
  C4: 3-way agent routing
  C5: DSPy-based declarative pipeline

This serves as the primary ablation baseline against the proposed
AgenticRAGPipeline (ReAct-based). Ablation flags in ExperimentSettings
allow disabling each contribution independently.
"""

from __future__ import annotations

import dspy
from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.pipeline._mixin import SelfCorrectiveMixin
from agentic_rag.pipeline.base import PipelineResult
from agentic_rag.retriever.indexer import Passage


class LoopRAGPipeline(SelfCorrectiveMixin):
    """Preprocess → For-Loop Refinement → Generate/Route.

    The refinement loop follows a fixed sequence:
      retrieve → evaluate (4D) → refine query → repeat (up to max_retry)

    This is deterministic and predictable, but cannot adapt its strategy
    based on intermediate findings (unlike the agentic variant).
    """

    def run(
        self,
        question: str,
        conversation_history: str = "",
        system_prompt: str = "",
    ) -> PipelineResult:
        """Execute the loop-based Self-Corrective RAG pipeline."""
        system_prompt = system_prompt or (
            "You are a helpful knowledge assistant. Answer based on the provided passages."
        )

        # STEP 1: Preprocessing
        search_query, search_keywords, hyde_query, _topic = self._preprocess(
            question, conversation_history
        )
        llm_calls = 1

        # STEP 2-3: For-Loop Refinement (C1 + C2 + C3)
        passages, eval_scores, action_history, final_action, loop_calls = self._run_loop_refinement(
            search_query, search_keywords, hyde_query
        )
        llm_calls += loop_calls

        # STEP 4: Generation or Agent Routing (C4)
        return self._build_result(
            question=question,
            search_query=search_query,
            passages=passages,
            evaluation_scores=eval_scores,
            action_history=action_history,
            final_action=final_action,
            llm_calls=llm_calls,
            system_prompt=system_prompt,
        )

    def _run_loop_refinement(
        self,
        search_query: str,
        search_keywords: list[str],
        hyde_query: str | None,
    ) -> tuple[list[Passage], list[dict], list[str], str, int]:
        """Execute the standard for-loop self-corrective refinement.

        Returns:
            (accumulated_passages, evaluation_scores, action_history,
             final_action, llm_calls)
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
                    f"[LoopRAG] Retry {retry}: "
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
                f"[LoopRAG] Retry {retry}: "
                f"{len(new_passages)} new, {len(accumulated_passages)} accumulated"
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
                f"[LoopRAG] Retry {retry}: action={action}, "
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
