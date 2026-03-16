"""Agentic RAG pipeline — Proposed Method.

Uses DSPy RLM (Recursive Language Model) to autonomously refine retrieval
quality through a REPL-based tool-use loop. The agent decides which tools
to call, in what order, and when to stop — replacing the fixed for-loop
with autonomous decision-making.

This is the primary pipeline for the paper's proposed method.
"""

from __future__ import annotations

import dspy
from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.pipeline._mixin import SelfCorrectiveMixin
from agentic_rag.pipeline.base import PipelineResult
from agentic_rag.retriever.indexer import Passage
from agentic_rag.signatures.rlm_refinement import RLMRefinementSignature
from agentic_rag.tools import create_tools


class AgenticRAGPipeline(SelfCorrectiveMixin):
    """Preprocess → RLM Agentic Refinement → Generate/Route.

    The RLM agent has access to 5 tools:
      - search_passages: hybrid retrieval (FAISS + BM25 + RRF)
      - list_document_sections: browse document structure/TOC
      - get_terminology: map user terms to document vocabulary
      - evaluate_passages: 4D quality assessment
      - get_passage_detail: read full passage content

    The agent autonomously combines these tools to iteratively improve
    retrieval quality until a quality threshold is met or it determines
    that further improvement is unlikely.
    """

    def run(
        self,
        question: str,
        conversation_history: str = "",
        system_prompt: str = "",
    ) -> PipelineResult:
        """Execute the Agentic RAG pipeline."""
        system_prompt = system_prompt or (
            "You are a helpful knowledge assistant. Answer based on the provided passages."
        )

        # STEP 1: Preprocessing
        search_query, search_keywords, hyde_query, _topic = self._preprocess(
            question, conversation_history
        )
        llm_calls = 1

        # STEP 2-3: RLM Agentic Refinement
        passages, eval_scores, action_history, final_action, rlm_calls = self._run_rlm_refinement(
            search_query, search_keywords, hyde_query
        )
        llm_calls += rlm_calls

        # STEP 4: Generation or Agent Routing
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

    def _run_rlm_refinement(
        self,
        search_query: str,
        search_keywords: list[str],
        hyde_query: str | None,
    ) -> tuple[list[Passage], list[dict], list[str], str, int]:
        """Execute RLM-based agentic retrieval refinement.

        The RLM agent autonomously decides which tools to use and in what
        order to optimize retrieval quality. It operates in a sandboxed
        Python REPL with access to the registered tools.

        Returns:
            (accumulated_passages, evaluation_scores, action_history,
             final_action, llm_calls)
        """
        rlm_cfg = settings.rlm

        # Create tools that close over pipeline components
        tools = create_tools(
            retriever=self.retriever,
            indexer=self.indexer,
            evaluator=self.evaluator,
            enabled_tools=rlm_cfg.enabled_tools,
        )

        sub_lm = dspy.LM(settings.model.evaluate_model)

        rlm = dspy.RLM(
            RLMRefinementSignature,
            max_iterations=rlm_cfg.max_iterations,
            max_llm_calls=rlm_cfg.max_llm_calls,
            max_output_chars=rlm_cfg.max_output_chars,
            verbose=rlm_cfg.verbose,
            tools=tools,
            sub_lm=sub_lm,
        )

        logger.info(
            f"[AgenticRAG:RLM] Starting agentic refinement: "
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

        trajectory_steps = len(getattr(result, "trajectory", []))
        llm_calls = trajectory_steps + total_search_calls

        logger.info(
            f"[AgenticRAG:RLM] Completed: action={final_action}, "
            f"passages={len(accumulated_passages)}, "
            f"search_calls={total_search_calls}, "
            f"trajectory_steps={trajectory_steps}"
        )

        return accumulated_passages, evaluation_scores, action_history, final_action, llm_calls
