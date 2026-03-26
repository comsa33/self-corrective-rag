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

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.pipeline._mixin import SelfCorrectiveMixin
from agentic_rag.pipeline.base import PipelineResult
from agentic_rag.retriever.indexer import Passage
from agentic_rag.signatures.decompose import DecomposeQuerySignature


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

        # Build tool_score_trace from evaluation_scores for mediation analysis
        tool_score_trace = _build_loop_score_trace(eval_scores)

        # STEP 4: Generation or Agent Routing (C4)
        result = self._build_result(
            question=question,
            search_query=search_query,
            passages=passages,
            evaluation_scores=eval_scores,
            action_history=action_history,
            final_action=final_action,
            llm_calls=llm_calls,
            system_prompt=system_prompt,
        )
        result.tool_score_trace = tool_score_trace
        return result

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
        passage_scores: dict[str, float] = {}  # passage_id → retrieval score
        used_passage_ids: set[str] = set()
        evaluation_scores: list[dict] = []
        action_history: list[str] = []

        max_retry = eval_cfg.max_retry_count if exp.enable_iteration else 0
        final_action = "output"

        # Guard: preserve original keywords so refinement cannot remove them
        original_keywords = set(search_keywords)

        # --- Multi-hop Decomposition: retrieve per sub-question ---
        if exp.enable_dspy:
            decomposer = dspy.ChainOfThought(DecomposeQuerySignature)
            with dspy.context(lm=make_lm(settings.model.preprocess_model)):
                decomp = decomposer(question=search_query)
        else:
            from agentic_rag.tools.decompose import _call_lm_for_decompose

            data = _call_lm_for_decompose(search_query)

            class _DecompResult:
                is_multi_hop = bool(data.get("is_multi_hop", False))
                sub_questions = data.get("sub_questions", [search_query])

            decomp = _DecompResult()
        llm_calls += 1

        if decomp.is_multi_hop and len(decomp.sub_questions) > 1:
            logger.info(f"[LoopRAG] Multi-hop detected: {len(decomp.sub_questions)} sub-questions")
            for sq in decomp.sub_questions:
                sq_results = self.retriever.search(query=sq, exclude_ids=used_passage_ids)
                sq_passages = self.indexer.get_passages([pid for pid, _ in sq_results])
                sq_score_map = dict(sq_results)
                for p in sq_passages:
                    if p.id not in used_passage_ids:
                        accumulated_passages.append(p)
                        used_passage_ids.add(p.id)
                        passage_scores[p.id] = sq_score_map.get(p.id, 0.0)

            # Evict to max_passages if needed after sub-question retrieval
            max_p = settings.retrieval.max_passages
            if len(accumulated_passages) > max_p:
                accumulated_passages.sort(key=lambda p: passage_scores.get(p.id, 0.0), reverse=True)
                evicted = accumulated_passages[max_p:]
                accumulated_passages = accumulated_passages[:max_p]
                for ep in evicted:
                    used_passage_ids.discard(ep.id)
                    passage_scores.pop(ep.id, None)

        for retry in range(max_retry + 1):
            # --- C3: Targeted Query Refinement (on retry) ---
            if retry > 0 and exp.enable_refinement and evaluation_scores:
                prev_eval = evaluation_scores[-1]
                kw_to_add = prev_eval.get("keywords_to_add", [])
                kw_to_remove = prev_eval.get("keywords_to_remove", [])
                suggested = prev_eval.get("suggested_query", "")

                # Guard: never remove keywords from the original question
                safe_to_remove = [kw for kw in kw_to_remove if kw not in original_keywords]
                # Guard: limit new keywords to 3 per iteration
                kw_to_add = kw_to_add[:3]

                search_keywords = [
                    kw for kw in search_keywords if kw not in safe_to_remove
                ] + kw_to_add

                # Guard: ignore suggested_query if it drifts too far from original
                if suggested:
                    orig_tokens = set(search_query.lower().split())
                    sugg_tokens = set(suggested.lower().split())
                    overlap = len(orig_tokens & sugg_tokens) / max(len(orig_tokens), 1)
                    if overlap >= 0.4:
                        search_query = suggested
                    else:
                        logger.info(
                            f"[LoopRAG] Ignoring suggested_query (overlap={overlap:.0%}): "
                            f"'{suggested}'"
                        )

                logger.info(
                    f"[LoopRAG] Retry {retry}: "
                    f"refined query='{search_query}', +{kw_to_add}, -{safe_to_remove}"
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
            score_map = dict(search_results)

            if exp.enable_accumulation:
                for p in new_passages:
                    if p.id not in used_passage_ids:
                        accumulated_passages.append(p)
                        used_passage_ids.add(p.id)
                        passage_scores[p.id] = score_map.get(p.id, 0.0)

                # Score-based eviction: drop lowest-scored passages (not FIFO)
                max_p = settings.retrieval.max_passages
                if len(accumulated_passages) > max_p:
                    accumulated_passages.sort(
                        key=lambda p: passage_scores.get(p.id, 0.0), reverse=True
                    )
                    evicted = accumulated_passages[max_p:]
                    accumulated_passages = accumulated_passages[:max_p]
                    for ep in evicted:
                        used_passage_ids.discard(ep.id)
                        passage_scores.pop(ep.id, None)
            else:
                accumulated_passages = new_passages
                used_passage_ids = {p.id for p in new_passages}
                passage_scores = {p.id: score_map.get(p.id, 0.0) for p in new_passages}

            logger.info(
                f"[LoopRAG] Retry {retry}: "
                f"{len(new_passages)} new, {len(accumulated_passages)} accumulated"
            )

            # --- C2: 4D Quality Evaluation ---
            # On retries, evaluate only NEW passages to avoid dilution
            if retry > 0 and new_passages and exp.enable_accumulation:
                eval_context = self.format_passages(new_passages)
            else:
                eval_context = self.format_passages(accumulated_passages)

            if exp.enable_4d_evaluation:
                if exp.enable_dspy:
                    with dspy.context(lm=make_lm(settings.model.evaluate_model)):
                        eval_result = self.evaluator(
                            question=search_query,
                            passages=eval_context,
                            retry_count=retry,
                            max_retry=max_retry,
                        )
                else:
                    eval_result = self.evaluator(
                        question=search_query,
                        passages=eval_context,
                        retry_count=retry,
                        max_retry=max_retry,
                    )
                llm_calls += 1

                if exp.enable_dspy:
                    rel = int(eval_result.relevance_score)
                    cov = int(eval_result.coverage_score)
                    spec = int(eval_result.specificity_score)
                    suf = int(eval_result.sufficiency_score)
                    total = int(eval_result.total_score)
                else:
                    rel = int(eval_result.relevance)
                    cov = int(eval_result.coverage)
                    spec = int(eval_result.specificity)
                    suf = int(eval_result.sufficiency)
                    total = int(eval_result.total)

                # Guard: if total_score is 0 but sub-scores exist, recompute
                computed = rel + cov + spec + suf
                if total == 0 and computed > 0:
                    logger.warning(
                        f"[LoopRAG] total_score=0 but sub-scores sum to {computed}, using computed"
                    )
                    total = computed

                # Progressive leniency: lower effective threshold on later retries
                effective_threshold = max(eval_cfg.quality_threshold - (retry * 5), 20)
                # Override action with programmatic progressive leniency
                if total >= effective_threshold:
                    action_override = "output"
                elif retry >= max_retry:
                    # Always generate on final retry — don't route away
                    action_override = "output"
                else:
                    # Diminishing returns: stop if new passages scored worse
                    if retry > 0 and evaluation_scores:
                        prev_total = evaluation_scores[-1].get("total", 0)
                        if total <= prev_total:
                            logger.info(
                                f"[LoopRAG] Diminishing returns: "
                                f"score {total} <= prev {prev_total}, stopping"
                            )
                            action_override = "output"
                        else:
                            action_override = "refine"
                    else:
                        action_override = "refine"

                score_dict = {
                    "retry": retry,
                    "relevance": rel,
                    "coverage": cov,
                    "specificity": spec,
                    "sufficiency": suf,
                    "total": total,
                    "effective_threshold": effective_threshold,
                    "action": action_override,
                    "reasoning": eval_result.reasoning,
                    "keywords_to_add": eval_result.keywords_to_add,
                    "keywords_to_remove": eval_result.keywords_to_remove,
                    "suggested_query": eval_result.suggested_query,
                }
                evaluation_scores.append(score_dict)
                action = action_override
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


def _build_loop_score_trace(evaluation_scores: list[dict]) -> list[dict]:
    """Build tool_score_trace from loop evaluation_scores for mediation analysis."""
    trace: list[dict] = []
    prev_score: int | None = None
    for es in evaluation_scores:
        total = es.get("total")
        if total is None:
            continue
        entry = {
            "iteration_idx": es.get("retry", len(trace)),
            "tool_called": "evaluate_passages",
            "score_before": prev_score,
            "score_after": total,
            "score_delta": (total - prev_score) if prev_score is not None else None,
        }
        trace.append(entry)
        prev_score = total
    return trace
