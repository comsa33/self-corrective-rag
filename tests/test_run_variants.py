"""Variant selection must reject unknown names on every execution path.

Silently dropping a misspelled variant is the worst failure mode for an
experiment runner: fewer variants run than the author asked for, and the run
still exits successfully, so the gap is only noticed when the results are
assembled — if at all.
"""

from __future__ import annotations

import pytest

from experiments.run import run_ablation, run_experiment

RQ1_CONFIG = "configs/experiment/rq1.yaml"


def test_config_run_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        run_experiment(RQ1_CONFIG, variant_names=["Nonexistent Pipeline"])


def test_config_run_reports_available_variants():
    """The error must name the valid options, since these come from YAML."""
    with pytest.raises(ValueError, match="Agentic \\(ReAct\\)"):
        run_experiment(RQ1_CONFIG, variant_names=["Agentic ReAct"])  # missing parentheses


def test_ablation_run_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown variant"):
        run_ablation(variant_names=["Nonexistent Variant"])


def test_partially_valid_selection_is_still_rejected():
    """One good name must not mask a typo in another."""
    with pytest.raises(ValueError, match="Nonexistent Pipeline"):
        run_experiment(RQ1_CONFIG, variant_names=["Naive RAG", "Nonexistent Pipeline"])
