"""The verifier must reject every defect that has invalidated a run before.

Each test plants one defect in an otherwise valid run directory and expects
exactly that check to FAIL; the clean directory must pass with no FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.manifest import REQUIRED_ROW_FIELDS, RunManifest
from experiments.verify import FAIL, WARN, verify_repeat_set, verify_run_dir

SNAPSHOT = "gpt-5-mini-2025-08-07"
RUN_ID = "run"
COMMIT = "0" * 40


def _row(i: int, **overrides) -> dict:
    row = {f: None for f in REQUIRED_ROW_FIELDS}
    row.update(
        run_id=RUN_ID,
        repeat_index=None,
        git_commit=COMMIT,
        llm_cache_disabled=True,
        id=f"q{i}",
        question=f"question {i}",
        reference="a",
        prediction="a",
        pipeline="naive_rag",
        passages_used=50,
        total_passages_retrieved=50,
        llm_calls=1,
        latency_seconds=3.2,
        metered_calls=1,
        usage_complete=True,
        cost_usd=0.001,
        response_models=[SNAPSHOT],
    )
    row.update(overrides)
    return row


def _manifest(**overrides) -> dict:
    slot = {
        "slot": "generate",
        "spec": "azure/gpt-5-mini",
        "resolved": "azure/gpt-5-mini",
        "provider": "azure",
        "expected_response_model": SNAPSHOT,
        "observed_response_model": SNAPSHOT,
    }
    m = {
        "run_id": RUN_ID,
        "created_at": "t",
        "finished_at": "t",
        "command": ["run.py"],
        "experiment": "RQ1",
        "dataset": "2wikimultihopqa",
        "n": 2,
        "variants": ["Naive RAG"],
        "seed": 42,
        "git": {"commit": COMMIT, "branch": "main", "dirty": False, "modified_files": []},
        "packages": {"dspy": "3.1.3"},
        "cache": {"llm_cache_disabled": True},
        "model_defaults": {},
        "retrieval": {},
        "evaluation": {},
        "agent": {},
        "models": [{**slot, "slot": s} for s in ("preprocess", "evaluate", "generate", "agent")],
        "max_passages_by_pipeline": {"Naive RAG": None},
        "preflight": {"status": "ok", "mismatches": [], "unpinned": []},
        "observed_response_models": [SNAPSHOT],
    }
    m.update(overrides)
    return m


def _write_run(tmp_path: Path, rows: list[dict], manifest: dict | None, summary: dict | None):
    if manifest is not None:
        RunManifest.model_validate(manifest).save(tmp_path)
    (tmp_path / "rq1_naive_rag.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    if summary is not None:
        (tmp_path / "rq1_naive_rag_summary.json").write_text(json.dumps(summary))
    return tmp_path


def _summary(variant: str = "Naive RAG") -> dict:
    return {
        "extra": {"variant": variant},
        "metrics": {"f1": 0.5},
        "settings": {
            "max_passages": 30,
            "passage_cap": None,
            "enabled_tools": ["search"],
            "models": {},
            "seed": 42,
            "top_k": 50,
        },
    }


def _failed(checks) -> set[str]:
    return {c.name.split(": ")[-1] for c in checks if c.status == FAIL}


@pytest.fixture
def clean_run(tmp_path):
    return _write_run(tmp_path, [_row(0), _row(1)], _manifest(), _summary())


def test_clean_run_passes(clean_run):
    checks = verify_run_dir(clean_run)
    assert not _failed(checks), [c for c in checks if c.status == FAIL]


def test_missing_manifest_fails(tmp_path):
    run = _write_run(tmp_path, [_row(0), _row(1)], None, _summary())
    assert "manifest" in _failed(verify_run_dir(run, n=2))


@pytest.mark.parametrize(
    ("rows", "expected_check"),
    [
        ([_row(0)], "n"),
        ([_row(0), _row(0)], "unique ids"),
        (
            [_row(0), {"id": "q1", "question": "x", "error": "rate limit", "pipeline": "p"}],
            "errors",
        ),
        ([_row(0), _row(1, latency_seconds=0.01)], "latency < 1s share"),
        ([_row(0), _row(1, metered_calls=0)], "usage metered"),
        ([_row(0), _row(1, usage_complete=False)], "usage complete"),
        ([_row(0), _row(1, response_models=["gpt-5-mini-2099"])], "row snapshots"),
    ],
)
def test_row_defects_fail(tmp_path, rows, expected_check):
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert expected_check in _failed(verify_run_dir(run))


def test_missing_required_field_fails(tmp_path):
    rows = [_row(0), _row(1)]
    del rows[1]["cost_usd"]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    checks = verify_run_dir(run)
    assert "required fields" in _failed(checks)
    assert any("cost_usd" in c.detail for c in checks if c.status == FAIL)


def test_snapshot_mismatch_in_manifest_fails(tmp_path):
    m = _manifest()
    m["models"][2]["observed_response_model"] = "gpt-5-mini-2099"
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    assert "snapshot generate" in _failed(verify_run_dir(run))


def test_stray_snapshot_during_run_fails(tmp_path):
    m = _manifest(observed_response_models=[SNAPSHOT, "gpt-5-mini-2099"])
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    assert "snapshot during run" in _failed(verify_run_dir(run))


def test_unpinned_snapshot_only_warns(tmp_path):
    m = _manifest()
    for slot in m["models"]:
        slot["expected_response_model"] = None
    m["preflight"]["unpinned"] = ["azure/gpt-5-mini"]
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    assert any(c.name == "snapshot pinned" and c.status == WARN for c in checks)


def test_cache_on_fails(tmp_path):
    run = _write_run(
        tmp_path, [_row(0), _row(1)], _manifest(cache={"llm_cache_disabled": False}), _summary()
    )
    assert "cache disabled" in _failed(verify_run_dir(run))


def test_dirty_tree_fails_unless_allowed(tmp_path):
    m = _manifest(
        git={"commit": "0" * 40, "branch": "main", "dirty": True, "modified_files": ["x"]}
    )
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    assert "git clean" in _failed(verify_run_dir(run))
    assert "git clean" not in _failed(verify_run_dir(run, allow_dirty=True))


def test_unfinished_or_mismatched_preflight_fails(tmp_path):
    m = _manifest(finished_at=None, preflight={"status": "mismatch", "mismatches": ["x"]})
    run = _write_run(tmp_path, [_row(0), _row(1)], m, _summary())
    failed = _failed(verify_run_dir(run))
    assert {"preflight", "run finished"} <= failed


def test_summary_without_settings_fails(tmp_path):
    run = _write_run(tmp_path, [_row(0), _row(1)], _manifest(), {"metrics": {"f1": 1}})
    assert "summary" in _failed(verify_run_dir(run))


def test_variant_count_must_match_manifest(tmp_path):
    run = _write_run(tmp_path, [_row(0), _row(1)], _manifest(variants=["A", "B"]), _summary())
    assert "variants complete" in _failed(verify_run_dir(run))


# ---------------------------------------------------------------------------
# Resumed runs: rows written by an earlier process
# ---------------------------------------------------------------------------
def test_resumed_rows_from_same_commit_only_warn(tmp_path):
    rows = [_row(0, run_id="earlier_run"), _row(1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    warned = [c for c in checks if c.status == WARN and c.name.endswith("resumed rows")]
    assert warned and "earlier_run" in warned[0].detail and "1 row(s)" in warned[0].detail


def test_resumed_rows_from_another_commit_fail(tmp_path):
    rows = [_row(0, run_id="earlier_run", git_commit="f" * 40), _row(1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "resumed rows commit" in _failed(verify_run_dir(run))


def test_rows_without_provenance_fail(tmp_path):
    rows = [_row(0, run_id=None, git_commit=None), _row(1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "provenance" in _failed(verify_run_dir(run))


def test_rows_made_with_cache_on_fail(tmp_path):
    rows = [_row(0, run_id="earlier_run", llm_cache_disabled=False), _row(1)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "row cache" in _failed(verify_run_dir(run))


def test_march_2026_paper_run_is_rejected():
    """The directories the paper was built from must not pass: that is the point."""
    paper = Path("data/results/paper")
    if not paper.exists():
        pytest.skip("paper results not on this machine")
    run = next(paper.glob("*_rq1_2wikimultihopqa_*"))
    assert "manifest" in _failed(verify_run_dir(run, n=200))


# ---------------------------------------------------------------------------
# Repeated runs checked as a set
# ---------------------------------------------------------------------------
def _repeat_dirs(tmp_path, ks, ns=None):
    dirs = []
    for i, k in enumerate(ks):
        n = ns[i] if ns else 2
        d = tmp_path / f"run_k{k}"
        d.mkdir()
        _write_run(
            d,
            [_row(0), _row(1)],
            _manifest(
                run_id=d.name,
                run_key=f"rq1|2wikimultihopqa|gpt-5-mini|n{n}|k{k}",
                repeat_index=k,
                n=n,
            ),
            _summary(),
        )
        dirs.append(d)
    return dirs


def _set_failed(checks) -> set[str]:
    return {c.name for c in checks if c.status == FAIL}


def test_complete_repeat_set_passes(tmp_path):
    assert not _set_failed(verify_repeat_set(_repeat_dirs(tmp_path, [1, 2, 3])))


def test_gap_in_repeat_indices_fails(tmp_path):
    assert "repeat index contiguous" in _set_failed(
        verify_repeat_set(_repeat_dirs(tmp_path, [1, 3]))
    )


def test_duplicate_repeat_index_fails(tmp_path):
    dirs = _repeat_dirs(tmp_path, [1, 2])
    # a second directory claiming k=2
    d = tmp_path / "run_k2_again"
    d.mkdir()
    _write_run(
        d,
        [_row(0), _row(1)],
        _manifest(run_id=d.name, run_key="rq1|2wikimultihopqa|gpt-5-mini|n2|k2", repeat_index=2),
        _summary(),
    )
    assert "repeat index unique" in _set_failed(verify_repeat_set([*dirs, d]))


def test_uneven_n_across_repeats_fails(tmp_path):
    assert "same n" in _set_failed(verify_repeat_set(_repeat_dirs(tmp_path, [1, 2], ns=[2, 3])))


def test_single_run_is_not_a_repeat_set(tmp_path):
    assert verify_repeat_set(_repeat_dirs(tmp_path, [1])) == []


# ---------------------------------------------------------------------------
# Passage cap (controlled-M scope)
# ---------------------------------------------------------------------------
def test_capped_variant_must_not_exceed_cap(tmp_path):
    m = _manifest(max_passages_by_pipeline={"Naive RAG": 30})
    rows = [_row(0, passages_used=30), _row(1, passages_used=31)]
    run = _write_run(tmp_path, rows, m, _summary())
    assert "passage cap" in _failed(verify_run_dir(run))


def test_capped_variant_within_cap_passes(tmp_path):
    m = _manifest(max_passages_by_pipeline={"Naive RAG": 30})
    rows = [_row(0, passages_used=30), _row(1, passages_used=30)]
    run = _write_run(tmp_path, rows, m, _summary())
    assert not _failed(verify_run_dir(run))


def test_uncapped_variant_must_use_everything_retrieved(tmp_path):
    rows = [_row(0), _row(1, passages_used=30, total_passages_retrieved=50)]
    run = _write_run(tmp_path, rows, _manifest(), _summary())
    assert "passage cap" in _failed(verify_run_dir(run))


def test_cap_never_reached_only_warns(tmp_path):
    m = _manifest(max_passages_by_pipeline={"Naive RAG": 30})
    rows = [_row(0, passages_used=12), _row(1, passages_used=9)]
    run = _write_run(tmp_path, rows, m, _summary())
    checks = verify_run_dir(run)
    assert not _failed(checks)
    assert any(c.name.endswith("passage cap reached") and c.status == WARN for c in checks)
