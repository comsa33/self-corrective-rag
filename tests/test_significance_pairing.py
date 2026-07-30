"""Paired significance tests must pair on question id, not list position.

A paired test is only valid if each pair holds the same question. As soon as
one pipeline drops an item — a provider failure, a filtered response — every
later position shifts, and pairing by position silently compares answers to
different questions.
"""

from __future__ import annotations

import json

from experiments.analysis.significance import SignificanceAnalyzer


def _write_run(directory, pipeline, rows):
    """Write one pipeline's results as the runner would."""
    path = directory / f"{pipeline}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for qid, prediction, reference in rows:
            f.write(
                json.dumps(
                    {
                        "id": qid,
                        "pipeline": pipeline,
                        "question": f"question {qid}",
                        "prediction": prediction,
                        "reference": reference,
                    }
                )
                + "\n"
            )


def test_pairing_survives_a_missing_item(tmp_path):
    """Dropping one item must not shift every later pair by one question."""
    # Baseline answers all six; the comparison pipeline lost q2 entirely.
    base_rows = [(f"q{i}", "hit" if i % 2 == 0 else "miss", "hit") for i in range(6)]
    comp_rows = [r for r in base_rows if r[0] != "q2"]

    _write_run(tmp_path, "agentic_(react)", base_rows)
    _write_run(tmp_path, "loop_refinement", comp_rows)

    analyzer = SignificanceAnalyzer.from_results_dir(tmp_path)
    result = analyzer.pairwise_tests(baseline="agentic_(react)", metric="f1")

    # Both pipelines answered identically on every shared question, so the
    # true delta is exactly zero. Position-based pairing would misalign q3..q5
    # against q2..q4 and invent a non-zero difference.
    assert result["loop_refinement"]["delta"] == 0.0


def test_pairing_is_order_independent(tmp_path):
    """A resumed run appends retried items out of order; that must not matter."""
    base_rows = [(f"q{i}", "hit" if i % 2 == 0 else "miss", "hit") for i in range(6)]
    # Same content, but q1 and q4 were retried and appended at the end.
    reordered = [r for r in base_rows if r[0] not in {"q1", "q4"}]
    reordered += [r for r in base_rows if r[0] in {"q1", "q4"}]

    _write_run(tmp_path, "agentic_(react)", base_rows)
    _write_run(tmp_path, "loop_refinement", reordered)

    analyzer = SignificanceAnalyzer.from_results_dir(tmp_path)
    result = analyzer.pairwise_tests(baseline="agentic_(react)", metric="f1")

    assert result["loop_refinement"]["delta"] == 0.0


def _write_legacy_run(directory, pipeline, rows):
    """Write results without an id field, as older result files were."""
    path = directory / f"{pipeline}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for prediction, reference, failed in rows:
            record = {"pipeline": pipeline, "prediction": prediction, "reference": reference}
            if failed:
                record["error"] = "provider failure"
            f.write(json.dumps(record) + "\n")


def test_legacy_files_fall_back_to_original_position(tmp_path):
    """Without ids, the fallback must be the position in the file as written.

    Numbering only the surviving records would renumber everything after a
    failure, reintroducing the misalignment in the one case the fallback exists
    to handle.
    """
    # Baseline answers all six. The comparison pipeline failed on index 2,
    # so its surviving records are indices 0,1,3,4,5 of the original file.
    base_rows = [("hit" if i % 2 == 0 else "miss", "hit", False) for i in range(6)]
    comp_rows = [(p, r, i == 2) for i, (p, r, _) in enumerate(base_rows)]

    _write_legacy_run(tmp_path, "agentic_(react)", base_rows)
    _write_legacy_run(tmp_path, "loop_refinement", comp_rows)

    analyzer = SignificanceAnalyzer.from_results_dir(tmp_path)
    result = analyzer.pairwise_tests(baseline="agentic_(react)", metric="f1")

    # Both pipelines answered identically on every question they both answered.
    assert result["loop_refinement"]["delta"] == 0.0


def test_identical_runs_report_no_difference(tmp_path):
    """The ordinary complete-data case still behaves as before."""
    rows_a = [(f"q{i}", "hit", "hit") for i in range(8)]
    rows_b = [(f"q{i}", "miss", "hit") for i in range(8)]

    _write_run(tmp_path, "agentic_(react)", rows_a)
    _write_run(tmp_path, "naive_rag", rows_b)

    analyzer = SignificanceAnalyzer.from_results_dir(tmp_path)
    result = analyzer.pairwise_tests(baseline="agentic_(react)", metric="f1")

    assert result["naive_rag"]["delta"] == 1.0
    assert result["naive_rag"]["boot_p"] < 0.05
