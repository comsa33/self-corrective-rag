"""A run must be able to prove the conditions it ran under.

The March 2026 runs could not, so partial re-runs had to be discarded. The
manifest records the observed model snapshot, the parameters litellm really
forwards, package versions and the commit, and refuses to start when the
provider answers with a snapshot other than the declared one.
"""

from __future__ import annotations

import json

import dspy
import pytest

import experiments.manifest as manifest_mod
from agentic_rag.config.settings import ModelSettings, settings
from experiments.manifest import (
    MANIFEST_FILENAME,
    ManifestMismatchError,
    RunManifest,
    build_manifest,
    git_state,
    sent_params,
)

SNAPSHOT = "gpt-5-mini-2025-08-07"


def _pin(monkeypatch, value: str) -> None:
    monkeypatch.setattr(settings.model, "expected_response_models", value)


def _build(**overrides) -> RunManifest:
    kwargs = dict(
        run_id="20260907_000000_rq1_2wikimultihopqa_n2_gpt-5-mini",
        experiment="RQ1",
        dataset="2wikimultihopqa",
        n=2,
        variants=["Naive RAG", "Agentic (ReAct)"],
        config_path="configs/experiment/rq1.yaml",
        sample_size=2,
    )
    kwargs.update(overrides)
    return build_manifest(**kwargs)


# ---------------------------------------------------------------------------
# Declared snapshot
# ---------------------------------------------------------------------------
def test_expected_snapshot_is_parsed_per_model_string():
    ms = ModelSettings(
        LLM_EXPECTED_RESPONSE_MODELS=f"azure/gpt-5-mini={SNAPSHOT}, openai/gpt-5-mini={SNAPSHOT}"
    )
    assert ms.expected_response_model("azure/gpt-5-mini") == SNAPSHOT
    assert ms.expected_response_model("openai/gpt-5-mini") == SNAPSHOT
    assert ms.expected_response_model("gemini/gemini-3.1-flash-lite-preview") is None


def test_unpinned_when_nothing_declared():
    ms = ModelSettings(LLM_EXPECTED_RESPONSE_MODELS="")
    assert ms.expected_response_model("azure/gpt-5-mini") is None


# ---------------------------------------------------------------------------
# Parameters as sent
# ---------------------------------------------------------------------------
def test_gpt5_mini_drops_temperature_but_sends_reasoning_effort():
    """What the paper must state: temperature=0 never reached gpt-5-mini."""
    lm = dspy.LM("azure/gpt-5-mini", temperature=0.0, max_tokens=16000, reasoning_effort="low")
    sent = sent_params(lm.model, lm.kwargs)
    assert "temperature" not in sent
    assert sent["reasoning_effort"] == "low"
    assert sent["max_completion_tokens"] == 16000


def test_gemini_keeps_temperature():
    lm = dspy.LM("gemini/gemini-3.1-flash-lite-preview", temperature=0.0, max_tokens=4096)
    sent = sent_params(lm.model, lm.kwargs)
    assert sent["temperature"] == 0.0
    assert sent["max_output_tokens"] == 4096


def test_credentials_never_reach_the_manifest(monkeypatch):
    monkeypatch.setattr(settings.model, "generate_model", "ollama_cloud/gpt-oss:120b")
    rec = manifest_mod.model_slot_record("generate")
    dumped = rec.model_dump_json()
    assert "api_key" not in dumped
    assert "api_base" not in dumped
    assert rec.resolved == "openai/gpt-oss:120b"
    # Ollama's OpenAI-compatible route does not carry reasoning_effort.
    assert "reasoning_effort" not in rec.sent_params


# ---------------------------------------------------------------------------
# Build / persist
# ---------------------------------------------------------------------------
def test_manifest_records_environment_and_settings():
    m = _build()
    assert m.git["commit"] and len(m.git["commit"]) == 40
    assert m.packages["dspy"] and m.packages["litellm"]
    assert m.retrieval["max_passages"] == settings.retrieval.max_passages
    assert m.agent["enabled_tools"] == settings.agent.enabled_tools
    assert m.seed == settings.experiment.seed
    assert [r.slot for r in m.models] == ["preprocess", "evaluate", "generate", "agent"]
    assert m.preflight["status"] == "pending"


def test_manifest_roundtrip(tmp_path):
    m = _build()
    path = m.save(tmp_path)
    assert path.name == MANIFEST_FILENAME
    loaded = RunManifest.load(tmp_path)
    assert loaded == m
    assert json.loads(path.read_text())["run_id"] == m.run_id


def test_git_state_reports_this_repository():
    state = git_state()
    assert state["commit"]
    assert isinstance(state["dirty"], bool)
    assert all(not f.startswith(" ") for f in state["modified_files"])


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def _fake_observer(answer: str | None):
    def _observe(spec, run_id, meter):
        return answer

    return _observe


def test_preflight_aborts_on_snapshot_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.model, "generate_model", "azure/gpt-5-mini")
    _pin(monkeypatch, f"azure/gpt-5-mini={SNAPSHOT}")
    monkeypatch.setattr(manifest_mod, "_observe_response_model", _fake_observer("gpt-5-mini-2099"))
    m = _build()

    with pytest.raises(ManifestMismatchError, match="gpt-5-mini-2099"):
        m.run_preflight(tmp_path, meter=object())

    # The evidence is on disk even though the run never started.
    on_disk = RunManifest.load(tmp_path)
    assert on_disk.preflight["status"] == "mismatch"
    assert any("generate" in s for s in on_disk.preflight["mismatches"])


def test_preflight_passes_on_declared_snapshot(monkeypatch, tmp_path):
    for slot in ("preprocess", "evaluate", "generate", "agent"):
        monkeypatch.setattr(settings.model, f"{slot}_model", "azure/gpt-5-mini")
    _pin(monkeypatch, f"azure/gpt-5-mini={SNAPSHOT}")
    calls: list[str] = []

    def _observe(spec, run_id, meter):
        calls.append(spec)
        return SNAPSHOT

    monkeypatch.setattr(manifest_mod, "_observe_response_model", _observe)
    m = _build()
    m.run_preflight(tmp_path, meter=object())

    assert m.preflight["status"] == "ok"
    assert m.preflight["unpinned"] == []
    assert all(r.observed_response_model == SNAPSHOT for r in m.models)
    assert calls == ["azure/gpt-5-mini"], "one probe per distinct model, not per slot"


def test_preflight_records_but_does_not_abort_when_unpinned(monkeypatch, tmp_path):
    _pin(monkeypatch, "")
    monkeypatch.setattr(manifest_mod, "_observe_response_model", _fake_observer("whatever"))
    m = _build()
    m.run_preflight(tmp_path, meter=object())
    assert m.preflight["status"] == "ok"
    assert m.preflight["unpinned"] == sorted({r.spec for r in m.models})
