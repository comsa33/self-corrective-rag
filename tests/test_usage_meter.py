"""Measured per-question usage must land on the result row.

The March 2026 runs recorded no tokens and no cost per question, so the cost
column of the paper had to be re-measured afterwards. The meter attributes
litellm's own usage report to the question that caused it, including calls
whose success callback arrives after the pipeline has already returned.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import experiments.common as common
from agentic_rag.evaluation.cost_tracker import LiteLLMMeter
from agentic_rag.pipeline.base import PipelineResult
from experiments.common import run_pipeline_on_dataset


def _response(model="gpt-5-mini-2025-08-07", prompt=100, cached=40, completion=20, reasoning=5):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
    )


def _call(meter: LiteLLMMeter, call_id: str, cost: float | None = 0.001, **kw) -> None:
    kwargs = {"litellm_call_id": call_id, "model": "azure/gpt-5-mini", "response_cost": cost}
    meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)
    t0 = datetime.now()
    meter.log_success_event(kwargs, _response(**kw), t0, t0 + timedelta(milliseconds=250))


def test_window_aggregates_only_calls_after_mark():
    meter = LiteLLMMeter()
    _call(meter, "before")
    mark = meter.mark()
    _call(meter, "a", prompt=10, cached=0, completion=1, reasoning=0)
    _call(meter, "b", prompt=20, cached=5, completion=2, reasoning=1, cost=None)

    usage = meter.usage_since(mark)
    assert usage["metered_calls"] == 2
    assert usage["prompt_tokens"] == 30
    assert usage["cached_tokens"] == 5
    assert usage["completion_tokens"] == 3
    assert usage["reasoning_tokens"] == 1
    assert usage["total_tokens"] == 33
    assert usage["cost_usd"] == 0.001
    assert usage["calls_without_cost"] == 1
    assert usage["response_models"] == ["gpt-5-mini-2025-08-07"]


def test_drain_waits_for_late_success_callback():
    """litellm reports success from a thread pool after the call returned."""
    meter = LiteLLMMeter()
    kwargs = {"litellm_call_id": "late", "model": "azure/gpt-5-mini", "response_cost": 0.5}
    meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)

    def _late_report():
        time.sleep(0.2)
        t0 = datetime.now()
        meter.log_success_event(kwargs, _response(), t0, t0)

    threading.Thread(target=_late_report).start()
    assert meter.usage_since(0)["metered_calls"] == 0
    assert meter.drain(timeout=5.0)
    assert meter.usage_since(0)["metered_calls"] == 1
    assert meter.usage_since(0)["cost_usd"] == 0.5


def test_drain_gives_up_on_a_call_that_never_reports():
    meter = LiteLLMMeter()
    meter.log_pre_api_call("m", [], {"litellm_call_id": "lost"})
    assert meter.drain(timeout=0.05) is False


def test_failure_clears_pending():
    meter = LiteLLMMeter()
    kwargs = {"litellm_call_id": "boom"}
    meter.log_pre_api_call("m", [], kwargs)
    meter.log_failure_event(kwargs, None, None, None)
    assert meter.drain(timeout=0.05) is True
    assert meter.failures == 1


class _MeteredPipeline:
    """Pipeline stub whose every run triggers one metered call."""

    def __init__(self, meter: LiteLLMMeter) -> None:
        self.meter = meter
        self.n = 0

    def run(self, question: str) -> PipelineResult:
        self.n += 1
        _call(self.meter, f"c{self.n}", prompt=100 * self.n, cached=0, completion=10, reasoning=0)
        return PipelineResult(question=question, answer="x", llm_calls=1)


def test_result_rows_carry_their_own_usage(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    dataset = [{"id": "q1", "question": "one", "answer": "a"}, {"id": "q2", "question": "two"}]

    rows = run_pipeline_on_dataset(_MeteredPipeline(meter), dataset, "p")

    assert [r["prompt_tokens"] for r in rows] == [100, 200]
    assert all(r["metered_calls"] == 1 for r in rows)
    assert all(r["cost_usd"] == 0.001 for r in rows)
    assert rows[0]["response_models"] == ["gpt-5-mini-2025-08-07"]


def test_rows_without_a_meter_still_have_the_fields(monkeypatch):
    """Analysis code must be able to rely on the columns being present."""
    monkeypatch.setattr(common, "_meter", None)
    meter = LiteLLMMeter()
    rows = run_pipeline_on_dataset(_MeteredPipeline(meter), [{"id": "q", "question": "?"}], "p")
    assert rows[0]["metered_calls"] == 0
    assert rows[0]["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# A call that never reports in time must not be billed to the next question
# ---------------------------------------------------------------------------
def test_leaked_call_is_written_off_not_billed_to_next_window():
    meter = LiteLLMMeter()
    kwargs = {"litellm_call_id": "slow", "model": "azure/gpt-5-mini", "response_cost": 9.0}
    meter.log_pre_api_call("azure/gpt-5-mini", [], kwargs)
    assert meter.drain(timeout=0.05) is False
    assert meter.leaked_calls == 1

    # Next question starts; the slow report lands now.
    mark = meter.mark()
    t0 = datetime.now()
    meter.log_success_event(kwargs, _response(), t0, t0)
    _call(meter, "next", cost=0.001)

    usage = meter.usage_since(mark)
    assert usage["metered_calls"] == 1, "the late report must not appear in this window"
    assert usage["cost_usd"] == 0.001
    assert meter.drain(timeout=0.05) is True


class _SlowCallbackPipeline:
    """Every run starts one call whose success report never arrives in time."""

    def __init__(self, meter: LiteLLMMeter) -> None:
        self.meter = meter
        self.n = 0

    def run(self, question: str) -> PipelineResult:
        self.n += 1
        self.meter.log_pre_api_call("m", [], {"litellm_call_id": f"slow{self.n}"})
        _call(self.meter, f"fast{self.n}")
        return PipelineResult(question=question, answer="x", llm_calls=2)


def test_row_marks_incomplete_usage(monkeypatch):
    meter = LiteLLMMeter()
    monkeypatch.setattr(common, "_meter", meter)
    monkeypatch.setattr(common, "DRAIN_TIMEOUT_SECONDS", 0.05)
    dataset = [{"id": "q1", "question": "one"}, {"id": "q2", "question": "two"}]

    rows = run_pipeline_on_dataset(_SlowCallbackPipeline(meter), dataset, "p")

    assert [r["usage_complete"] for r in rows] == [False, False]
    assert [r["leaked_calls"] for r in rows] == [1, 1]
    assert [r["metered_calls"] for r in rows] == [1, 1], "only the reported call is counted"
    assert meter.leaked_calls == 2
