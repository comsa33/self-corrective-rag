"""Resuming a judging run must not lose or duplicate verdicts.

Both failures here were real. The checkpoint keyed on the question id alone,
but a results file can hold the same question once per pipeline — the judge
panel carries 200 rows over 184 ids — so a resume dropped 16 rows into the
lookup and rewrote the file without them. And a verdict from a truncated
completion was indistinguishable from a real one, so a rerun with more room had
nothing to re-score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

judge_module = pytest.importorskip("scripts.run_llm_judge", reason="requires scripts package")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_key_separates_same_id_across_pipelines():
    a = {"id": "q1", "pipeline": "naive_rag"}
    b = {"id": "q1", "pipeline": "agentic_(react)"}
    assert judge_module._key(a) != judge_module._key(b)


def test_key_survives_a_missing_pipeline_field():
    assert judge_module._key({"id": "q1"}) == ("", "q1")


@pytest.fixture
def judged_file(tmp_path: Path) -> Path:
    """A checkpoint holding one question under two pipelines, one of them bad."""
    path = tmp_path / "rq1_judged.jsonl"
    rows = [
        {"id": "q1", "pipeline": "naive_rag", "llm_judge": 1.0, "judge_status": "ok"},
        {"id": "q1", "pipeline": "agentic_(react)", "llm_judge": 0.0, "judge_status": "ok"},
        {"id": "q2", "pipeline": "naive_rag", "llm_judge": 0.0, "judge_status": "truncated"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_rewrite_keeps_both_rows_that_share_an_id(judged_file: Path):
    """Dropping the unusable verdict must not take the duplicate id with it."""
    kept, judged = [], {}
    for row in _rows(judged_file):
        if row.get("judge_status", "ok") != "ok":
            continue
        judged[judge_module._key(row)] = row
        kept.append(row)

    judged_file.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
    )

    after = _rows(judged_file)
    assert len(after) == 2, "a row sharing an id with another was dropped"
    assert {r["pipeline"] for r in after} == {"naive_rag", "agentic_(react)"}
    assert len(judged) == 2


def test_unusable_verdicts_are_not_treated_as_done(judged_file: Path):
    """The truncated verdict must be absent from the checkpoint so it is redone."""
    judged = {
        judge_module._key(r): r for r in _rows(judged_file) if r.get("judge_status", "ok") == "ok"
    }
    assert ("naive_rag", "q2") not in judged
    assert ("naive_rag", "q1") in judged


def test_verdicts_written_before_judge_status_existed_are_kept():
    """Older files carry no judge_status; those verdicts are still real."""
    legacy = {"id": "q1", "pipeline": "naive_rag", "llm_judge": 1.0}
    assert legacy.get("judge_status", "ok") == "ok"
