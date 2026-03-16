"""Naive RAG pipeline — Baseline 1.

Simple retrieve-then-generate with no evaluation or correction.
Uses the same retriever and generator as the proposed method
for fair comparison.
"""

from __future__ import annotations

import dspy
from loguru import logger

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.pipeline.base import BasePipeline, PipelineResult
from agentic_rag.retriever.hybrid import HybridRetriever
from agentic_rag.retriever.indexer import DocumentIndexer
from agentic_rag.signatures.generate import QnAGenerateSignature


class NaiveRAGPipeline(BasePipeline):
    """Retrieve → Generate (no evaluation, no correction)."""

    def __init__(
        self,
        retriever: HybridRetriever,
        indexer: DocumentIndexer,
    ):
        super().__init__(retriever, indexer)

        # DSPy modules
        self.generator = dspy.ChainOfThought(QnAGenerateSignature)

    def run(
        self,
        question: str,
        system_prompt: str = "",
        top_k: int | None = None,
        query_method: str | None = None,
    ) -> PipelineResult:
        """Execute Naive RAG: retrieve → generate."""
        top_k = top_k or settings.retrieval.top_k
        system_prompt = system_prompt or (
            "You are a helpful knowledge assistant. Answer based on the provided passages."
        )

        # 1. Retrieve
        search_results = self.retriever.search(
            query=question,
            top_k=top_k,
            method=query_method,
        )
        passage_ids = [pid for pid, _score in search_results]
        passages = self.indexer.get_passages(passage_ids)

        logger.info(f"[NaiveRAG] Retrieved {len(passages)} passages")

        # 2. Generate
        context = self.format_passages(passages)

        with dspy.context(lm=make_lm(settings.model.generate_model)):
            gen_result = self.generator(
                question=question,
                passages=context,
                system_prompt=system_prompt,
            )

        return PipelineResult(
            question=question,
            answer=gen_result.answer,
            footnotes=gen_result.footnotes,
            recommended_questions=gen_result.recommended_questions,
            passages_used=passages,
            total_passages_retrieved=len(passages),
            retry_count=0,
            action_history=["output"],
            llm_calls=1,
        )
