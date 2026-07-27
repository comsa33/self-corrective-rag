"""Checkpoint resume behavior for long, rate-limited experiment runs.

A run against a rate-limited provider will be interrupted. What matters is
that the rerun costs only the outstanding questions: completed items must
never be re-answered, and failed items must never be silently frozen into
the results as permanent gaps.
"""

from __future__ import annotations

import json

from agentic_rag.pipeline.base import PipelineResult
from experiments.common import run_pipeline_on_dataset


class FlakyPipeline:
    """Pipeline stub that fails for a chosen set of questions."""

    def __init__(self, fail_questions: set[str], fail_times: int = 999) -> None:
        self.fail_questions = fail_questions
        self.fail_times = fail_times
        self.calls: list[str] = []
        self._failures: dict[str, int] = {}

    def run(self, question: str) -> PipelineResult:
        self.calls.append(question)
        if question in self.fail_questions:
            seen = self._failures.get(question, 0)
            if seen < self.fail_times:
                self._failures[question] = seen + 1
                raise RuntimeError("rate limit exceeded")
        return PipelineResult(question=question, answer=f"answer to {question}")


def _dataset(n: int) -> list[dict]:
    return [{"id": f"q{i}", "question": f"question {i}", "answer": f"answer {i}"} for i in range(n)]


def test_completed_items_are_not_rerun(tmp_path):
    """A rerun must not spend a single call on already-answered questions."""
    dataset = _dataset(4)

    first = FlakyPipeline(fail_questions=set())
    run_pipeline_on_dataset(first, dataset, "p", checkpoint_dir=tmp_path)
    assert len(first.calls) == 4

    second = FlakyPipeline(fail_questions=set())
    results = run_pipeline_on_dataset(second, dataset, "p", checkpoint_dir=tmp_path)

    assert second.calls == [], "resumed run re-answered completed questions"
    assert len(results) == 4
    assert {r["id"] for r in results} == {"q0", "q1", "q2", "q3"}


def test_failed_items_are_retried_on_resume(tmp_path):
    """Items that failed must come back as work, not as a permanent gap."""
    dataset = _dataset(4)

    first = FlakyPipeline(fail_questions={"question 1", "question 2"})
    results = run_pipeline_on_dataset(
        first, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0
    )
    assert sum(1 for r in results if "error" in r) == 2

    # The provider recovers; the rerun should touch only the two failures.
    second = FlakyPipeline(fail_questions=set())
    results = run_pipeline_on_dataset(second, dataset, "p", checkpoint_dir=tmp_path)

    assert sorted(second.calls) == ["question 1", "question 2"]
    assert all("error" not in r for r in results)
    assert len(results) == 4
    assert {r["id"] for r in results} == {"q0", "q1", "q2", "q3"}


def test_transient_failure_recovers_within_one_run(tmp_path):
    """A single transient failure should be retried in place, not deferred."""
    dataset = _dataset(2)
    pipeline = FlakyPipeline(fail_questions={"question 0"}, fail_times=1)

    results = run_pipeline_on_dataset(
        pipeline, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=2, retry_backoff=0.0
    )

    assert all("error" not in r for r in results)
    assert pipeline.calls.count("question 0") == 2


def test_checkpoint_is_written_after_every_item(tmp_path):
    """An interrupt must cost at most the item in flight."""
    dataset = _dataset(3)
    pipeline = FlakyPipeline(fail_questions={"question 2"})

    run_pipeline_on_dataset(pipeline, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0)

    checkpoint = tmp_path / "p_checkpoint.jsonl"
    records = [json.loads(x) for x in checkpoint.read_text(encoding="utf-8").splitlines() if x]
    assert len(records) == 3
    assert not list(tmp_path.glob("*.tmp")), "temporary checkpoint file was left behind"


def test_resume_reports_only_successful_items_as_done(tmp_path):
    """Partial progress must be counted by successes, not by line count."""
    dataset = _dataset(5)

    first = FlakyPipeline(fail_questions={"question 3"})
    run_pipeline_on_dataset(first, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0)

    second = FlakyPipeline(fail_questions={"question 3"})
    run_pipeline_on_dataset(second, dataset, "p", checkpoint_dir=tmp_path, max_item_retries=0)

    # Only the still-failing question is retried, and it stays marked failed.
    assert second.calls == ["question 3"]
    checkpoint = tmp_path / "p_checkpoint.jsonl"
    records = [json.loads(x) for x in checkpoint.read_text(encoding="utf-8").splitlines() if x]
    assert sum(1 for r in records if "error" in r) == 1
    assert len(records) == 5
