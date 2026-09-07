"""Base pipeline interface for all RAG variants.

Every pipeline (Naive, CRAG, Self-Corrective) implements the same interface
so experiments can swap them transparently.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from agentic_rag.config.settings import settings
from agentic_rag.retriever.hybrid import HybridRetriever
from agentic_rag.retriever.indexer import DocumentIndexer, Passage


@dataclass
class PipelineResult:
    """Unified result container for all pipeline variants."""

    question: str
    answer: str
    footnotes: str = ""
    recommended_questions: list[str] = field(default_factory=list)

    # Retrieval metadata
    passages_used: list[Passage] = field(default_factory=list)
    total_passages_retrieved: int = 0

    # Self-corrective metadata
    retry_count: int = 0
    evaluation_scores: list[dict] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)
    agent_type: str | None = None  # clarification / domain_expert / fallback

    # Mediation analysis data (RQ2 extension)
    tool_score_trace: list[dict] = field(default_factory=list)
    # Each entry: {iteration_idx, tool_called, score_before, score_after, score_delta}
    question_difficulty: dict = field(default_factory=dict)
    # Keys: hop_count, entity_count, question_type

    # Cost tracking
    latency_seconds: float = 0.0
    llm_calls: int = 0
    total_tokens: int = 0

    @property
    def final_action(self) -> str:
        """The last action taken (output / route_to_agent)."""
        return self.action_history[-1] if self.action_history else "output"


class BasePipeline(ABC):
    """Abstract base class for RAG pipelines."""

    def __init__(
        self,
        retriever: HybridRetriever,
        indexer: DocumentIndexer,
    ):
        self.retriever = retriever
        self.indexer = indexer

    @abstractmethod
    def run(self, question: str, **kwargs) -> PipelineResult:
        """Execute the pipeline on a single question."""
        ...

    # ------------------------------------------------------------------
    # Context budget
    # ------------------------------------------------------------------
    @classmethod
    def passage_cap(cls) -> int | None:
        """Largest number of passages this pipeline hands to the generator.

        The accumulating pipelines (loop, agentic) always evict down to
        `max_passages`. Single-shot baselines override this: they are
        uncapped unless `max_passages_all_pipelines` is on. The run manifest
        records the answer per pipeline, and the verifier holds every result
        row to it, so the paper's "controlled for M" claim is checked rather
        than asserted.
        """
        return settings.retrieval.max_passages

    @classmethod
    def cap_passages(cls, passages: list[Passage]) -> list[Passage]:
        """Truncate ranked passages to this pipeline's cap, if it has one."""
        cap = cls.passage_cap()
        return passages if cap is None else passages[:cap]

    def run_timed(self, question: str, **kwargs) -> PipelineResult:
        """Run with automatic latency tracking."""
        start = time.perf_counter()
        result = self.run(question, **kwargs)
        result.latency_seconds = time.perf_counter() - start
        return result

    @staticmethod
    def format_passages(passages: list[Passage]) -> str:
        """Format passages into a context string for LLM input.

        Follows the original system's format:
        [id, title, content]
        """
        parts = []
        for p in passages:
            parts.append(f"[{p.id}, {p.title}, {p.content}]")
        return "\n\n".join(parts)
