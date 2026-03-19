"""Tests for pipeline components.

Unit tests for pipeline base classes, result structures,
and configuration-driven behavior (ablation flags).
Does NOT call LLM — tests structure and logic only.
"""

from __future__ import annotations

from agentic_rag.config.settings import settings
from agentic_rag.pipeline.base import PipelineResult


# ---------------------------------------------------------------
# Pipeline imports and class hierarchy
# ---------------------------------------------------------------
class TestPipelineImports:
    def test_import_agentic_pipeline(self):
        from agentic_rag.pipeline.agentic import AgenticRAGPipeline
        from agentic_rag.pipeline.base import BasePipeline

        assert issubclass(AgenticRAGPipeline, BasePipeline)

    def test_import_loop_pipeline(self):
        from agentic_rag.pipeline.base import BasePipeline
        from agentic_rag.pipeline.loop import LoopRAGPipeline

        assert issubclass(LoopRAGPipeline, BasePipeline)

    def test_import_all_from_package(self):
        from agentic_rag.pipeline import (
            AgenticRAGPipeline,
            CRAGReplicaPipeline,
            LoopRAGPipeline,
            NaiveRAGPipeline,
        )

        assert AgenticRAGPipeline is not None
        assert LoopRAGPipeline is not None
        assert NaiveRAGPipeline is not None
        assert CRAGReplicaPipeline is not None

    def test_tools_registry(self):
        from agentic_rag.tools import TOOL_REGISTRY

        assert set(TOOL_REGISTRY.keys()) == {
            "search",
            "structure",
            "terminology",
            "evaluate",
            "inspect",
            "decompose",
            "calculate",
        }


# ---------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------
class TestPipelineResult:
    def test_default_values(self):
        result = PipelineResult(question="test q", answer="test answer")
        assert result.answer == "test answer"
        assert result.question == "test q"
        assert result.retry_count == 0
        assert result.action_history == []
        assert result.evaluation_scores == []
        assert result.agent_type is None
        assert result.passages_used == []

    def test_fields(self):
        result = PipelineResult(
            question="hello?",
            answer="hello",
            retry_count=2,
            agent_type="clarification",
        )
        assert result.answer == "hello"
        assert result.retry_count == 2
        assert result.agent_type == "clarification"


# ---------------------------------------------------------------
# Settings / Ablation flags
# ---------------------------------------------------------------
class TestAblationFlags:
    def test_default_flags(self):
        """All features should be enabled by default."""
        assert settings.experiment.enable_iteration is True
        assert settings.experiment.enable_accumulation is True
        assert settings.experiment.enable_4d_evaluation is True
        assert settings.experiment.enable_refinement is True
        assert settings.experiment.enable_agent_routing is True
        assert settings.experiment.enable_dspy is True

    def test_quality_threshold(self):
        assert settings.evaluation.quality_threshold == 55

    def test_max_retry(self):
        assert settings.evaluation.max_retry_count == 3

    def test_max_passages(self):
        assert settings.retrieval.max_passages == 30


# ---------------------------------------------------------------
# Evaluation metrics (no LLM needed)
# ---------------------------------------------------------------
class TestMetrics:
    def test_exact_match(self):
        from agentic_rag.evaluation.metrics import exact_match

        assert exact_match("hello world", "hello world") == 1.0
        assert exact_match("Hello World", "hello world") == 1.0
        assert exact_match("hello", "world") == 0.0

    def test_f1_score(self):
        from agentic_rag.evaluation.metrics import token_f1

        # Perfect match
        assert token_f1("the quick brown fox", "the quick brown fox") == 1.0
        # Partial overlap
        f1 = token_f1("the quick brown fox", "the quick red fox")
        assert 0.5 < f1 < 1.0
        # No overlap
        assert token_f1("hello", "world") == 0.0

    def test_f1_empty(self):
        from agentic_rag.evaluation.metrics import token_f1

        assert token_f1("", "") == 1.0  # Both empty = match
        assert token_f1("hello", "") == 0.0


# ---------------------------------------------------------------
# Cost tracker
# ---------------------------------------------------------------
class TestCostTracker:
    def test_record_and_summary(self):
        from agentic_rag.evaluation.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(
            model="gpt-4o-mini",
            stage="test",
            input_tokens=100,
            output_tokens=50,
            latency_ms=500,
        )
        tracker.record(
            model="gpt-4o-mini",
            stage="test",
            input_tokens=200,
            output_tokens=100,
            latency_ms=800,
        )

        summary = tracker.summary()
        assert summary["total_calls"] == 2
        assert summary["total_tokens"] == 450  # 100+50 + 200+100

    def test_empty_tracker(self):
        from agentic_rag.evaluation.cost_tracker import CostTracker

        tracker = CostTracker()
        summary = tracker.summary()
        assert summary["total_calls"] == 0


# ---------------------------------------------------------------
# Training collector
# ---------------------------------------------------------------
class TestTrainingCollector:
    def test_add_and_retrieve(self):
        from agentic_rag.optimization.collector import TrainingCollector

        collector = TrainingCollector()
        collector.add("TestSig", {"q": "hello"}, {"a": "world"})
        assert collector.total_count == 1
        assert len(collector.get_examples("TestSig")) == 1

    def test_to_dspy_examples(self):
        from agentic_rag.optimization.collector import TrainingCollector

        collector = TrainingCollector()
        collector.add("TestSig", {"q": "hello"}, {"a": "world"})
        examples = collector.to_dspy_examples("TestSig")
        assert len(examples) == 1
        assert examples[0]["q"] == "hello"
        assert examples[0]["a"] == "world"

    def test_save_and_load(self, tmp_dir):
        from agentic_rag.optimization.collector import TrainingCollector

        collector = TrainingCollector()
        collector.add("Sig1", {"x": "1"}, {"y": "2"})
        collector.add("Sig2", {"x": "3"}, {"y": "4"})

        path = tmp_dir / "training.json"
        collector.save(path)

        loaded = TrainingCollector()
        loaded.load(path)
        assert loaded.total_count == 2
        assert loaded.summary() == {"Sig1": 1, "Sig2": 1}
