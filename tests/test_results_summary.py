"""A saved run must record the settings that decide its numbers.

Every field asserted here was missing at some point and cost real time:
max_passages hid that baselines carried 50 passages against the proposed
method's 30 in every published run; enabled_tools hid that RQ1 ran four tools
where RQ3/RQ4 ran all of them, which is why two tables disagreed; max_tokens
hid a limit that silently truncated a reasoning model's answers for a whole
dataset. A summary that omits them cannot tell one run from another.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

common = pytest.importorskip("experiments.common", reason="requires project deps")
from agentic_rag.config.settings import settings

REQUIRED = {
    "top_k",
    "max_passages",
    "query_method",
    "hybrid_weight",
    "enabled_tools",
    "agent_max_iterations",
    "max_tokens",
    "temperature",
    "models",
    "quality_threshold",
    "max_retry",
    "seed",
}


@pytest.fixture
def summary(tmp_path: Path) -> dict:
    rows = [
        {"id": "q1", "prediction": "Nolan", "reference": "Christopher Nolan", "question": "Who?"}
    ]
    common.save_results(rows, "probe", run_dir=tmp_path)
    return json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))


def test_summary_records_every_setting_that_changes_results(summary: dict):
    assert set(summary["settings"]) >= REQUIRED, REQUIRED - set(summary["settings"])


def test_summary_records_all_four_model_slots(summary: dict):
    assert set(summary["settings"]["models"]) == {"preprocess", "evaluate", "generate", "agent"}


def test_summary_reflects_the_values_actually_in_force(tmp_path: Path):
    """A sweep varies these per run, so the file must follow, not hardcode."""
    original = (settings.retrieval.top_k, settings.retrieval.max_passages)
    try:
        settings.retrieval.top_k = 7
        settings.retrieval.max_passages = 7
        rows = [{"id": "q1", "prediction": "a", "reference": "a", "question": "q"}]
        common.save_results(rows, "probe", run_dir=tmp_path)
        recorded = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))[
            "settings"
        ]
        assert recorded["top_k"] == 7
        assert recorded["max_passages"] == 7
    finally:
        settings.retrieval.top_k, settings.retrieval.max_passages = original


def test_a_passed_snapshot_wins_over_the_current_globals(tmp_path: Path):
    """The sweep depends on this and it silently did not hold.

    Variants run one after another, then every result is saved at the end, so
    reading the globals at save time stamped the last variant's settings onto
    all of them — six variants across two conditions all claimed top_k=50. The
    caller therefore snapshots while its variant is running and passes that.
    """
    snapshot = dict(common.settings_snapshot(), top_k=5, max_passages=5)
    rows = [{"id": "q1", "prediction": "a", "reference": "a", "question": "q"}]
    common.save_results(rows, "probe", run_dir=tmp_path, settings_used=snapshot)

    recorded = json.loads((tmp_path / "probe_summary.json").read_text(encoding="utf-8"))["settings"]
    assert (recorded["top_k"], recorded["max_passages"]) == (5, 5)
    assert settings.retrieval.top_k != 5, "globals should be untouched by the snapshot"
