"""Budget-matched comparison: the loop must spend its budget, the agent must cap.

The paper's C1 question is whether the agentic pipeline's gain comes from
its control flow or from the extra calls it makes. That needs a loop that
makes exactly as many calls as the agent (stopping rule replaced by the
budget) and an agent held to the loop's cost (iterations capped), with the
budgets on record so the verifier can hold the rows to them.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agentic_rag.pipeline.loop as loop_mod
from agentic_rag.config.loader import load_experiment_config
from agentic_rag.config.settings import settings
from agentic_rag.pipeline.agentic import AgenticRAGPipeline
from agentic_rag.pipeline.loop import BUDGET_REFINE_ACTION, LOOP_FIXED_CALLS, LoopRAGPipeline
from agentic_rag.retriever.indexer import Passage


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("budget", "retries"), [(None, None), (10, 6), (4, 0), (2, 0), (7, 3)])
def test_loop_retry_budget(budget, retries):
    assert LoopRAGPipeline.loop_retry_budget(budget) == retries


@pytest.mark.parametrize(("budget", "iters"), [(None, 10), (4, 1), (5, 1), (10, 3), (30, 10)])
def test_agentic_effective_max_iters(monkeypatch, budget, iters):
    monkeypatch.setattr(settings.agent, "max_iterations", 10)
    monkeypatch.setattr(settings.experiment, "llm_call_budget", budget)
    assert AgenticRAGPipeline.effective_max_iters() == iters


# ---------------------------------------------------------------------------
# The loop really spends the budget (no LLM: evaluator/decomposer stubbed)
# ---------------------------------------------------------------------------
def _high_score_eval(**_):
    """An evaluation that would satisfy the score-stop rule on the first pass."""
    return SimpleNamespace(
        relevance_score=30,
        coverage_score=25,
        specificity_score=25,
        sufficiency_score=20,
        total_score=100,
        reasoning="fine",
        keywords_to_add=[],
        keywords_to_remove=[],
        suggested_query="",
    )


@pytest.fixture
def loop(monkeypatch):
    # Decomposer is built inside the loop; replace the factory with a stub.
    monkeypatch.setattr(
        loop_mod.dspy,
        "ChainOfThought",
        lambda *_a, **_k: lambda **_kw: SimpleNamespace(is_multi_hop=False, sub_questions=["q"]),
    )
    retriever = MagicMock()
    retriever.search.side_effect = lambda **kw: [
        (f"p{i}", 1.0 - i * 0.01) for i in range(3) if f"p{i}" not in (kw.get("exclude_ids") or ())
    ]
    indexer = MagicMock()
    indexer.get_passages.side_effect = lambda ids: [
        Passage(id=i, title=i, content="c") for i in ids
    ]
    monkeypatch.setattr(settings.experiment, "enable_iteration", True)
    monkeypatch.setattr(settings.experiment, "enable_4d_evaluation", True)
    monkeypatch.setattr(settings.experiment, "enable_dspy", True)
    monkeypatch.setattr(settings.evaluation, "max_retry_count", 3)
    pipeline = LoopRAGPipeline(retriever, indexer)
    pipeline.evaluator = MagicMock(side_effect=_high_score_eval)
    return pipeline


def test_score_stop_loop_stops_on_a_satisfied_evaluator(loop, monkeypatch):
    monkeypatch.setattr(settings.experiment, "llm_call_budget", None)
    _, _, actions, _, calls = loop._run_loop_refinement("q", ["k"], None)
    assert actions == ["output"]
    assert calls == 2  # decompose + one evaluation


@pytest.mark.parametrize("budget", [10, 6, 4])
def test_budget_loop_spends_exactly_the_budget(loop, monkeypatch, budget):
    monkeypatch.setattr(settings.experiment, "llm_call_budget", budget)
    _, scores, actions, final, calls = loop._run_loop_refinement("q", ["k"], None)

    # preprocess and generate happen outside; the loop accounts for the rest
    assert calls == budget - 2
    assert len(scores) == budget - LOOP_FIXED_CALLS + 1
    assert actions[:-1] == [BUDGET_REFINE_ACTION] * (budget - LOOP_FIXED_CALLS)
    assert actions[-1] == "output" and final == "output"
    assert loop.evaluator.call_count == budget - LOOP_FIXED_CALLS + 1


def test_budget_ignored_without_iteration(loop, monkeypatch):
    """Single-pass stays single-pass: a budget cannot turn iteration on."""
    monkeypatch.setattr(settings.experiment, "llm_call_budget", 10)
    monkeypatch.setattr(settings.experiment, "enable_iteration", False)
    _, _, actions, _, calls = loop._run_loop_refinement("q", ["k"], None)
    assert actions == ["output"] and calls == 2


# ---------------------------------------------------------------------------
# Config and manifest
# ---------------------------------------------------------------------------
def test_budget_matched_config_has_the_four_columns():
    exp = load_experiment_config("configs/experiment/budget_matched.yaml")
    names = [v.name for v in exp.variants]
    assert names == [
        "Loop (score-stop)",
        "Loop (budget=10)",
        "Agentic (ReAct)",
        "Agentic (budget=6)",
    ]
    budgets = [v.overrides.get("experiment", {}).get("llm_call_budget") for v in exp.variants]
    assert budgets == [None, 10, None, 6]


def test_manifest_controls_record_budget_and_pipeline_kind():
    from experiments.run import _manifest_controls

    exp = load_experiment_config("configs/experiment/budget_matched.yaml")
    controls = _manifest_controls(exp.variants)
    assert controls["llm_call_budget_by_pipeline"] == {
        "Loop (score-stop)": None,
        "Loop (budget=10)": 10,
        "Agentic (ReAct)": None,
        "Agentic (budget=6)": 6,
    }
    assert controls["pipeline_by_variant"]["Loop (budget=10)"] == "loop"
    assert controls["pipeline_by_variant"]["Agentic (budget=6)"] == "agentic"
    assert settings.experiment.llm_call_budget is None, "globals must be restored"


def test_variant_settings_do_not_leak_into_the_next_variant(monkeypatch):
    """A field base.yaml does not list must still reset between variants."""
    from agentic_rag.config.loader import apply_settings
    from experiments.run import _BASELINE_SETTINGS, _apply_variant

    exp = load_experiment_config("configs/experiment/budget_matched.yaml")
    by_name = {v.name: v for v in exp.variants}
    try:
        _apply_variant(by_name["Loop (budget=10)"])
        assert settings.experiment.llm_call_budget == 10
        _apply_variant(by_name["Agentic (ReAct)"])
        assert settings.experiment.llm_call_budget is None
    finally:
        apply_settings(_BASELINE_SETTINGS)
