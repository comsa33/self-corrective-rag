"""Tests for YAML config loader.

Unit tests for config loading, merging, and experiment config parsing.
"""

from __future__ import annotations

from agentic_rag.config.loader import (
    ExperimentConfig,
    VariantConfig,
    _deep_merge,
    load_ablation_configs,
    load_config,
    load_experiment_config,
)
from agentic_rag.config.settings import PROJECT_ROOT

CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------
class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"model": {"temperature": 0.0, "max_tokens": 4096}}
        override = {"model": {"temperature": 0.7}}
        result = _deep_merge(base, override)
        assert result["model"]["temperature"] == 0.7
        assert result["model"]["max_tokens"] == 4096

    def test_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1


# ---------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------
class TestLoadConfig:
    def test_load_base_yaml(self):
        cfg = load_config(CONFIGS_DIR / "base.yaml")
        assert cfg["model"]["generate_model"] == "gpt-4o"
        assert cfg["evaluation"]["quality_threshold"] == 55
        assert cfg["experiment"]["seed"] == 42

    def test_load_pipeline_config_merges_base(self):
        cfg = load_config(CONFIGS_DIR / "pipeline" / "agentic.yaml")
        # Should have base defaults merged
        assert cfg["model"]["generate_model"] == "gpt-4o"
        # Should have pipeline-specific override
        assert cfg["experiment"]["enable_rlm_refinement"] is True

    def test_overrides_applied(self):
        cfg = load_config(
            CONFIGS_DIR / "base.yaml",
            overrides={"experiment": {"seed": 123}},
        )
        assert cfg["experiment"]["seed"] == 123


# ---------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------
class TestLoadExperimentConfig:
    def test_load_rq1(self):
        exp = load_experiment_config(CONFIGS_DIR / "experiment" / "rq1.yaml")
        assert isinstance(exp, ExperimentConfig)
        assert "RQ1" in exp.name
        assert len(exp.variants) == 5

    def test_variant_names(self):
        exp = load_experiment_config(CONFIGS_DIR / "experiment" / "rq1.yaml")
        names = [v.name for v in exp.variants]
        assert "Naive RAG" in names
        assert "Agentic (RLM)" in names

    def test_variant_pipeline_types(self):
        exp = load_experiment_config(CONFIGS_DIR / "experiment" / "rq1.yaml")
        pipelines = {v.name: v.pipeline for v in exp.variants}
        assert pipelines["Naive RAG"] == "naive"
        assert pipelines["CRAG Replica"] == "crag"
        assert pipelines["Agentic (RLM)"] == "agentic"

    def test_variant_overrides(self):
        exp = load_experiment_config(CONFIGS_DIR / "experiment" / "rq1.yaml")
        single_pass = next(v for v in exp.variants if v.name == "Single-Pass")
        assert single_pass.overrides["experiment"]["enable_iteration"] is False

    def test_rq3_rlm_tool_ablation(self):
        """RQ3 has a variant with explicit enabled_tools list."""
        exp = load_experiment_config(CONFIGS_DIR / "experiment" / "rq3.yaml")
        wo_eval = next(v for v in exp.variants if v.name == "RLM w/o Eval Tool")
        assert "evaluate" not in wo_eval.overrides["rlm"]["enabled_tools"]
        assert "search" in wo_eval.overrides["rlm"]["enabled_tools"]

    def test_rq4_structure_aware_tools(self):
        """RQ4 tests structure-aware tool ablation."""
        exp = load_experiment_config(CONFIGS_DIR / "experiment" / "rq4.yaml")
        assert len(exp.variants) == 4
        wo_section = next(v for v in exp.variants if v.name == "w/o Section Index")
        assert "structure" not in wo_section.overrides["rlm"]["enabled_tools"]

    def test_rq5_has_train_size(self):
        exp = load_experiment_config(CONFIGS_DIR / "experiment" / "rq5.yaml")
        assert exp.train_size == 50
        assert exp.val_size == 20

    def test_rq5_optimization_field(self):
        exp = load_experiment_config(CONFIGS_DIR / "experiment" / "rq5.yaml")
        bootstrap = next(v for v in exp.variants if v.name == "DSPy + Bootstrap")
        assert bootstrap.optimization == "bootstrap"

    def test_load_all_experiment_configs(self):
        """All experiment YAML files parse without error."""
        for yaml_path in sorted((CONFIGS_DIR / "experiment").glob("*.yaml")):
            exp = load_experiment_config(yaml_path)
            assert len(exp.variants) > 0, f"{yaml_path.name} has no variants"


# ---------------------------------------------------------------
# Variant config
# ---------------------------------------------------------------
class TestVariantConfig:
    def test_pipeline_class_path(self):
        v = VariantConfig(name="test", pipeline="naive")
        assert "NaiveRAGPipeline" in v.pipeline_class_path

    def test_import_pipeline_class(self):
        v = VariantConfig(name="test", pipeline="loop")
        cls = v.import_pipeline_class()
        assert cls.__name__ == "LoopRAGPipeline"

    def test_all_pipelines_importable(self):
        for key in ("naive", "crag", "loop", "agentic"):
            v = VariantConfig(name="test", pipeline=key)
            cls = v.import_pipeline_class()
            assert cls is not None


# ---------------------------------------------------------------
# Ablation configs
# ---------------------------------------------------------------
class TestLoadAblationConfigs:
    def test_loads_all_ablation_yamls(self):
        configs = load_ablation_configs()
        assert len(configs) >= 8

    def test_ablation_names(self):
        configs = load_ablation_configs()
        names = [c.name for c in configs]
        assert "Full (RLM + All Tools)" in names
        assert "w/o RLM (For-Loop)" in names
        assert "w/o Evaluate Tool" in names

    def test_ablation_tool_level(self):
        """Tool-level ablation configs correctly exclude specific tools."""
        configs = load_ablation_configs()
        wo_eval = next(c for c in configs if c.name == "w/o Evaluate Tool")
        assert "evaluate" not in wo_eval.overrides["rlm"]["enabled_tools"]
        assert "search" in wo_eval.overrides["rlm"]["enabled_tools"]

    def test_ablation_wo_rlm_uses_loop_pipeline(self):
        configs = load_ablation_configs()
        wo_rlm = next(c for c in configs if c.name == "w/o RLM (For-Loop)")
        assert wo_rlm.pipeline == "loop"
