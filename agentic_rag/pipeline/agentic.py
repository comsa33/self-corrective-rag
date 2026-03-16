"""Agentic RAG pipeline — Proposed Method.

Uses DSPy ReAct (Reasoning and Acting) to autonomously refine retrieval
quality through structured tool-use. The agent reasons about the current
situation, selects a tool, observes the result, and iterates until
quality thresholds are met — replacing the fixed for-loop with
autonomous decision-making.

This is the primary pipeline for the paper's proposed method.
"""

from __future__ import annotations

import json
import re

import dspy
from loguru import logger

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.pipeline._mixin import SelfCorrectiveMixin
from agentic_rag.pipeline.base import PipelineResult
from agentic_rag.retriever.indexer import Passage
from agentic_rag.signatures.agent import AgenticRefinementSignature
from agentic_rag.tools import create_tools


class AgenticRAGPipeline(SelfCorrectiveMixin):
    """Preprocess → ReAct Agentic Refinement → Generate/Route.

    The ReAct agent has access to 6 tools:
      - search_passages: hybrid retrieval (FAISS + BM25 + RRF)
      - decompose_query: multi-hop question decomposition
      - list_document_sections: browse document structure/TOC
      - get_terminology: map user terms to document vocabulary
      - evaluate_passages: 4D quality assessment
      - get_passage_detail: read full passage content

    The agent autonomously reasons about which tools to call, observes
    results, and iterates to improve retrieval quality until a quality
    threshold is met or it determines that further improvement is unlikely.
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

        # STEP 2-3: ReAct Agentic Refinement
        passages, eval_scores, action_history, final_action, react_calls = (
            self._run_react_refinement(search_query, search_keywords, hyde_query)
        )
        llm_calls += react_calls

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

    def _run_react_refinement(
        self,
        search_query: str,
        search_keywords: list[str],
        hyde_query: str | None,
    ) -> tuple[list[Passage], list[dict], list[str], str, int]:
        """Execute ReAct-based agentic retrieval refinement.

        The ReAct agent autonomously reasons about the situation and selects
        tools in a structured Thought → Action → Observation loop.

        Returns:
            (accumulated_passages, evaluation_scores, action_history,
             final_action, llm_calls)
        """
        agent_cfg = settings.agent

        # Create tools that close over pipeline components
        tools = create_tools(
            retriever=self.retriever,
            indexer=self.indexer,
            evaluator=self.evaluator,
            enabled_tools=agent_cfg.enabled_tools,
        )

        react = dspy.ReAct(
            AgenticRefinementSignature,
            tools=tools,
            max_iters=agent_cfg.max_iterations,
        )

        logger.info(
            f"[AgenticRAG:ReAct] Starting agentic refinement: "
            f"query='{search_query}', keywords={search_keywords[:5]}"
        )

        with dspy.context(lm=make_lm(settings.model.agent_model)):
            result = react(
                question=search_query,
                initial_query=hyde_query or search_query,
                initial_keywords=search_keywords,
                quality_threshold=settings.evaluation.quality_threshold,
                max_passages=settings.retrieval.max_passages,
            )

        # Parse structured trajectory
        trajectory = result.trajectory or {}
        action_history = parse_action_history(trajectory)
        evaluation_scores = parse_evaluation_scores(trajectory)
        search_calls = sum(1 for a in action_history if a == "search_passages")

        # Extract final outputs
        passage_ids = result.final_passages or []
        accumulated_passages = self.indexer.get_passages(passage_ids)
        final_action = result.final_action or "output"

        # LLM calls: one per ReAct iteration + one for final extraction
        llm_calls = (len(trajectory) // 4) + 1

        logger.info(
            f"[AgenticRAG:ReAct] Completed: action={final_action}, "
            f"passages={len(accumulated_passages)}, "
            f"search_calls={search_calls}, "
            f"trajectory_steps={len(action_history)}"
        )

        return accumulated_passages, evaluation_scores, action_history, final_action, llm_calls


# ---------------------------------------------------------------------------
# Trajectory parsing utilities
# ---------------------------------------------------------------------------


def parse_action_history(trajectory: dict) -> list[str]:
    """Extract ordered tool-call names from a ReAct trajectory.

    ReAct trajectory format:
        thought_0, tool_name_0, tool_args_0, observation_0,
        thought_1, tool_name_1, tool_args_1, observation_1, ...

    Returns:
        List of tool names in call order (e.g., ["search_passages", "evaluate_passages"]).
    """
    actions: list[str] = []
    idx = 0
    while f"tool_name_{idx}" in trajectory:
        tool_name = trajectory[f"tool_name_{idx}"]
        if tool_name != "finish":
            actions.append(tool_name)
        idx += 1
    return actions


def parse_evaluation_scores(trajectory: dict) -> list[dict]:
    """Extract 4D evaluation scores from evaluate_passages observations.

    Parses the JSON observation from each evaluate_passages tool call
    to capture the structured quality assessment results.

    Returns:
        List of evaluation score dicts with keys:
        {relevance, coverage, specificity, sufficiency, total, action, ...}
    """
    scores: list[dict] = []
    idx = 0
    while f"tool_name_{idx}" in trajectory:
        if trajectory[f"tool_name_{idx}"] == "evaluate_passages":
            observation = trajectory.get(f"observation_{idx}", "")
            parsed = _try_parse_json(observation)
            if parsed and "total" in parsed:
                scores.append(parsed)
        idx += 1
    return scores


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse JSON from a string, handling wrapped formats."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try extracting JSON from wrapped text (e.g., error prefix + JSON)
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
