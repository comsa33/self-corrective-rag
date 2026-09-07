"""A run must know what it is a run of, and must not start twice by accident.

Checkpoints used to be keyed by (config, dataset, model) alone, so any new
run under the same settings silently resumed the previous run's rows. Now
the run id keys the checkpoints, continuing a run is an explicit --resume,
and a fresh run whose key already has a finished directory is refused.
"""

from __future__ import annotations

import json

import pytest

from experiments.manifest import (
    MANIFEST_FILENAME,
    DuplicateRunError,
    RunManifest,
    find_finished_runs,
    make_run_key,
    plan_run,
)

KEY = dict(stem="rq1", dataset="2wikimultihopqa", model_tag="gpt-5-mini", n=200)


def _finished_manifest(run_dir, *, run_key, repeat_index=None, finished=True, attempt=1):
    run_dir.mkdir(parents=True)
    data = {
        "run_id": run_dir.name,
        "run_key": run_key,
        "repeat_index": repeat_index,
        "attempt": attempt,
        "created_at": "t",
        "finished_at": "t" if finished else None,
        "command": [],
        "experiment": "RQ1",
        "dataset": "2wikimultihopqa",
        "n": 200,
        "variants": ["Naive RAG"],
        "seed": 42,
        "git": {},
        "packages": {},
        "cache": {},
        "model_defaults": {},
        "retrieval": {},
        "evaluation": {},
        "agent": {},
        "models": [],
    }
    (run_dir / MANIFEST_FILENAME).write_text(json.dumps(data))
    return run_dir


def test_run_key_separates_repeats():
    assert make_run_key("rq1", "d", "m", 2, None) == "rq1|d|m|n2|k"
    assert make_run_key("rq1", "d", "m", 2, 1) == "rq1|d|m|n2|k1"
    assert make_run_key("rq1", "d", "m", 2, 1) != make_run_key("rq1", "d", "m", 2, 2)


def test_fresh_runs_get_distinct_checkpoint_dirs(tmp_path):
    a = plan_run(tmp_path, **KEY, repeat_index=1)
    b = plan_run(tmp_path, **KEY, repeat_index=2)
    assert a.checkpoint_dir != b.checkpoint_dir
    assert a.checkpoint_dir == tmp_path / "checkpoints" / a.run_id
    assert a.run_id.endswith("_k1") and b.run_id.endswith("_k2")
    assert a.run_dir == tmp_path / a.run_id


def test_finished_run_with_same_key_blocks_a_fresh_start(tmp_path):
    key = make_run_key(**KEY, repeat_index=1)
    _finished_manifest(tmp_path / "old_k1", run_key=key, repeat_index=1)

    with pytest.raises(DuplicateRunError, match="old_k1"):
        plan_run(tmp_path, **KEY, repeat_index=1)

    # another repetition, or an explicit override, is fine
    plan_run(tmp_path, **KEY, repeat_index=2)
    plan_run(tmp_path, **KEY, repeat_index=1, force=True)


def test_unfinished_run_does_not_block(tmp_path):
    key = make_run_key(**KEY, repeat_index=1)
    _finished_manifest(tmp_path / "crashed_k1", run_key=key, repeat_index=1, finished=False)
    assert find_finished_runs(tmp_path, key) == []
    plan_run(tmp_path, **KEY, repeat_index=1)


def test_resume_keeps_run_id_and_archives_manifest(tmp_path):
    key = make_run_key(**KEY, repeat_index=1)
    _finished_manifest(tmp_path / "crashed_k1", run_key=key, repeat_index=1, finished=False)

    plan = plan_run(tmp_path, **KEY, repeat_index=1, resume="crashed_k1")

    assert plan.resumed and plan.run_id == "crashed_k1" and plan.attempt == 2
    assert plan.checkpoint_dir == tmp_path / "checkpoints" / "crashed_k1"
    assert (tmp_path / "crashed_k1" / "manifest.attempt1.json").exists()
    assert not (tmp_path / "crashed_k1" / MANIFEST_FILENAME).exists()


def test_resume_of_unknown_or_mismatched_run_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        plan_run(tmp_path, **KEY, resume="nope")

    other_key = make_run_key("rq2", "hotpotqa", "gpt-5-mini", 200, None)
    _finished_manifest(tmp_path / "rq2_run", run_key=other_key, finished=False)
    with pytest.raises(ValueError, match="describes"):
        plan_run(tmp_path, **KEY, resume="rq2_run")


def test_resume_of_finished_run_needs_force(tmp_path):
    key = make_run_key(**KEY, repeat_index=None)
    _finished_manifest(tmp_path / "done", run_key=key)
    with pytest.raises(DuplicateRunError, match="already finished"):
        plan_run(tmp_path, **KEY, resume="done")
    plan = plan_run(tmp_path, **KEY, resume="done", force=True)
    assert plan.attempt == 2


def test_manifest_roundtrips_run_identity(tmp_path):
    from experiments.manifest import build_manifest

    m = build_manifest(
        run_id="x",
        experiment="e",
        dataset="d",
        n=2,
        variants=["v"],
        run_key="k",
        repeat_index=3,
        attempt=2,
    )
    m.save(tmp_path)
    loaded = RunManifest.load(tmp_path)
    assert (loaded.run_key, loaded.repeat_index, loaded.attempt) == ("k", 3, 2)
