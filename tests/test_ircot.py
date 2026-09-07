"""IRCoT re-implementation: the interleaving loop must behave as the paper says.

Retriever, indexer and both LLM modules are stubbed, so these check the
control flow only: termination on "answer is", the step bound, passage
deduplication across steps, the shared passage cap and the budget formula.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agentic_rag.config.loader import PIPELINE_REGISTRY, load_experiment_config
from agentic_rag.config.settings import settings
from agentic_rag.pipeline.ircot import PAPER_DEFAULTS, IRCoTPipeline
from agentic_rag.retriever.indexer import Passage


def _pipeline(monkeypatch, *, sentences: list[str], k: int = 2, max_steps: int = 4):
    """IRCoT over a fake corpus where query 'qN' style strings map to fresh ids.

    Each retrieval returns `k` passages named after the query so a repeated
    query yields the same ids (tests deduplication).
    """
    monkeypatch.setattr(settings.ircot, "per_step_k", k)
    monkeypatch.setattr(settings.ircot, "max_steps", max_steps)
    monkeypatch.setattr(settings.experiment, "llm_call_budget", None)

    retriever = MagicMock()

    def _search(query, top_k, exclude_ids):
        key = re.sub(r"\W+", "_", query)[:12]
        return [(f"{key}#{i}", 1.0) for i in range(top_k)]

    retriever.search.side_effect = _search
    indexer = MagicMock()
    indexer.get_passages.side_effect = lambda ids: [
        Passage(id=i, title=i, content="c") for i in ids
    ]

    p = IRCoTPipeline(retriever, indexer)
    it = iter(sentences)
    p.reasoner = MagicMock(side_effect=lambda **_: SimpleNamespace(next_sentence=next(it)))
    p.generator = MagicMock(
        return_value=SimpleNamespace(answer="42", footnotes="", recommended_questions=[])
    )
    return p


def test_stops_when_a_cot_sentence_contains_answer_is(monkeypatch):
    p = _pipeline(monkeypatch, sentences=["Alice was born in Paris.", "So the answer is: Paris."])
    r = p.run("Where was Alice born?")

    assert r.action_history == ["retrieve(q0)", "cot(1)", "retrieve(cot1)", "cot(2)", "output"]
    assert r.llm_calls == 3  # two reasoning steps + generation
    assert p.reasoner.call_count == 2
    assert r.answer == "42"
    assert r.evaluation_scores[-1]["terminated"] is True


def test_runs_to_max_steps_without_an_answer_sentence(monkeypatch):
    p = _pipeline(monkeypatch, sentences=["s1", "s2", "s3", "s4", "never used"], max_steps=4)
    r = p.run("q")

    cots = [a for a in r.action_history if a.startswith("cot(")]
    retrieves = [a for a in r.action_history if a.startswith("retrieve(")]
    assert cots == ["cot(1)", "cot(2)", "cot(3)", "cot(4)"]
    # the last sentence is not used for retrieval: nothing would read it
    assert retrieves == ["retrieve(q0)", "retrieve(cot1)", "retrieve(cot2)", "retrieve(cot3)"]
    assert r.llm_calls == 5 == settings.ircot.max_steps + 1
    assert r.action_history[-1] == "output"


def test_passages_are_deduplicated_across_steps(monkeypatch):
    # cot(1) repeats the question as its query -> same ids come back
    p = _pipeline(monkeypatch, sentences=["q", "So the answer is: x."], k=3)
    r = p.run("q")
    assert r.total_passages_retrieved == 3, "a repeated query must not add duplicates"
    assert len({x.id for x in r.passages_used}) == len(r.passages_used)


def test_collection_is_capped_at_m_but_retrieval_count_is_kept(monkeypatch):
    monkeypatch.setattr(settings.retrieval, "max_passages", 5)
    p = _pipeline(monkeypatch, sentences=["a", "b", "c", "So the answer is: d."], k=4)
    r = p.run("q")
    assert r.total_passages_retrieved == 16  # q0 + a + b + c, 4 each
    assert len(r.passages_used) == 5
    assert [x.id for x in r.passages_used][:4] == [f"q#{i}" for i in range(4)], "first collected"


def test_reasoner_sees_the_capped_context(monkeypatch):
    monkeypatch.setattr(settings.retrieval, "max_passages", 2)
    p = _pipeline(monkeypatch, sentences=["a", "So the answer is: d."], k=4)
    p.run("q")
    ctx = p.reasoner.call_args_list[-1].kwargs["passages"]
    assert ctx.count("[q#") == 2


@pytest.mark.parametrize(("budget", "steps"), [(None, 8), (2, 1), (1, 1), (5, 4), (20, 8)])
def test_budget_bounds_the_number_of_steps(monkeypatch, budget, steps):
    monkeypatch.setattr(settings.ircot, "max_steps", 8)
    monkeypatch.setattr(settings.experiment, "llm_call_budget", budget)
    assert IRCoTPipeline.effective_max_steps() == steps


def test_params_state_ours_next_to_the_papers(monkeypatch):
    monkeypatch.setattr(settings.experiment, "llm_call_budget", None)
    params = IRCoTPipeline.params()
    assert params["paper"] == PAPER_DEFAULTS
    assert params["paper"]["max_steps"] == 8 and params["paper"]["max_paragraphs"] == 15
    assert params["max_paragraphs"] == settings.retrieval.max_passages
    assert params["reader"].startswith("QnAGenerateSignature")


def test_ircot_is_registered_and_external_config_loads():
    assert PIPELINE_REGISTRY["ircot"].endswith("IRCoTPipeline")
    exp = load_experiment_config("configs/experiment/external.yaml")
    assert [v.name for v in exp.variants] == ["IRCoT"]
    assert exp.variants[0].overrides["ircot"] == {"max_steps": 8, "per_step_k": 8}


def test_manifest_controls_carry_ircot_params():
    from experiments.run import _manifest_controls

    exp = load_experiment_config("configs/experiment/external.yaml")
    controls = _manifest_controls(exp.variants)
    assert controls["pipeline_by_variant"] == {"IRCoT": "ircot"}
    assert controls["ircot_params_by_pipeline"]["IRCoT"]["max_steps"] == 8
    assert controls["max_passages_by_pipeline"]["IRCoT"] == settings.retrieval.max_passages
