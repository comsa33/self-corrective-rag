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

    The ReAct agent uses tools configured via settings.agent.enabled_tools
    (default: all 7 tools — search, decompose, structure, terminology,
    evaluate, inspect, calculate). For QA benchmarks, a subset is used.

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

        # STEP 1.5: Initial retrieval (baseline context, same as Naive)
        initial_results = self.retriever.search(query=search_query, top_k=settings.retrieval.top_k)
        initial_passages = self.indexer.get_passages([pid for pid, _ in initial_results])

        # STEP 2-3: ReAct Agentic Refinement
        agent_passages, eval_scores, action_history, final_action, react_calls = (
            self._run_react_refinement(search_query, search_keywords, hyde_query)
        )
        llm_calls += react_calls

        # STEP 3.5: Merge agent-refined + initial passages (agent first, deduped)
        # Use top_k as cap (not max_passages) to match Naive/CRAG passage counts
        seen = set()
        merged: list[Passage] = []
        cap = settings.retrieval.top_k
        for p in [*agent_passages, *initial_passages]:
            if p.id not in seen:
                seen.add(p.id)
                merged.append(p)
            if len(merged) >= cap:
                break
        passages = merged

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

        # Extract final outputs: agent-selected passages + trajectory supplement
        agent_ids = result.final_passages or []
        final_action = result.final_action or "output"

        # Always supplement agent-selected passages with trajectory-mined ones
        # Agent's picks come first (priority), then fill with search results
        max_p = settings.retrieval.max_passages
        trajectory_ids = _extract_passage_ids_from_trajectory(trajectory, max_p)
        # Merge: agent-selected first, then trajectory (deduped, up to max)
        seen = set()
        merged_ids: list[str] = []
        for pid in [*agent_ids, *trajectory_ids]:
            if pid not in seen:
                seen.add(pid)
                merged_ids.append(pid)
            if len(merged_ids) >= max_p:
                break
        accumulated_passages = self.indexer.get_passages(merged_ids)
        if not agent_ids and accumulated_passages:
            final_action = "output"

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


def _extract_passage_ids_from_trajectory(trajectory: dict, max_passages: int = 30) -> list[str]:
    """Extract unique passage IDs from search_passages observations in trajectory.

    When the agent fails to populate final_passages, this fallback mines
    passage IDs from all search tool observations, ordered by retrieval score.
    """
    seen: dict[str, float] = {}  # id → best score
    idx = 0
    while f"tool_name_{idx}" in trajectory:
        if trajectory[f"tool_name_{idx}"] == "search_passages":
            observation = trajectory.get(f"observation_{idx}", "")
            try:
                results = json.loads(observation)
                if isinstance(results, list):
                    for item in results:
                        pid = item.get("id", "")
                        score = item.get("score", 0.0)
                        if pid and (pid not in seen or score > seen[pid]):
                            seen[pid] = score
            except (json.JSONDecodeError, TypeError):
                pass
        idx += 1

    # Sort by score descending, take top max_passages
    sorted_ids = sorted(seen, key=lambda pid: seen[pid], reverse=True)
    return sorted_ids[:max_passages]


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
