"""M=30 must be applicable to the single-shot baselines by a flag, off by default.

The March 2026 runs gave naive and CRAG-style all 50 retrieved passages
while the accumulating pipelines were capped at 30; nothing on disk said
so. The flag keeps that behaviour as the default (the uncontrolled
condition) and adds the controlled one, and each pipeline class reports
its own cap so the manifest can record it.
"""

from __future__ import annotations

import pytest

from agentic_rag.config.settings import settings
from agentic_rag.pipeline.agentic import AgenticRAGPipeline
from agentic_rag.pipeline.crag import CRAGReplicaPipeline
from agentic_rag.pipeline.loop import LoopRAGPipeline
from agentic_rag.pipeline.naive import NaiveRAGPipeline
from agentic_rag.retriever.indexer import Passage


@pytest.fixture
def flag(monkeypatch):
    def _set(on: bool) -> None:
        monkeypatch.setattr(settings.retrieval, "max_passages_all_pipelines", on)

    return _set


def _passages(n: int) -> list[Passage]:
    return [Passage(id=f"p{i}", title=f"t{i}", content="c") for i in range(n)]


def test_default_preserves_march_behaviour(flag):
    flag(False)
    assert NaiveRAGPipeline.passage_cap() is None
    assert CRAGReplicaPipeline.passage_cap() is None
    assert LoopRAGPipeline.passage_cap() == settings.retrieval.max_passages
    assert AgenticRAGPipeline.passage_cap() == settings.retrieval.max_passages


def test_flag_caps_every_pipeline_at_the_same_m(flag, monkeypatch):
    flag(True)
    monkeypatch.setattr(settings.retrieval, "max_passages", 30)
    caps = {
        cls.passage_cap()
        for cls in (NaiveRAGPipeline, CRAGReplicaPipeline, LoopRAGPipeline, AgenticRAGPipeline)
    }
    assert caps == {30}


def test_cap_passages_truncates_only_when_capped(flag, monkeypatch):
    monkeypatch.setattr(settings.retrieval, "max_passages", 3)
    flag(False)
    assert len(NaiveRAGPipeline.cap_passages(_passages(5))) == 5
    flag(True)
    kept = NaiveRAGPipeline.cap_passages(_passages(5))
    assert [p.id for p in kept] == ["p0", "p1", "p2"], "keeps the top-ranked passages"
    assert len(CRAGReplicaPipeline.cap_passages(_passages(2))) == 2


def test_manifest_records_cap_per_variant(monkeypatch):
    from agentic_rag.config.loader import load_experiment_config
    from experiments.run import _variant_passage_caps

    exp = load_experiment_config("configs/experiment/rq1.yaml")
    monkeypatch.setenv("RETRIEVAL_MAX_PASSAGES_ALL_PIPELINES", "true")
    monkeypatch.setattr(settings.retrieval, "max_passages_all_pipelines", True)
    caps = _variant_passage_caps(exp.variants)
    assert caps["Naive RAG"] == 30
    assert caps["CRAG Replica"] == 30
    assert caps["Agentic (ReAct)"] == 30

    monkeypatch.setattr(settings.retrieval, "max_passages_all_pipelines", False)
    caps = _variant_passage_caps(exp.variants)
    assert caps["Naive RAG"] is None
    assert caps["Loop Refinement"] == 30
