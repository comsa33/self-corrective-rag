"""YAML configuration loader with layered merge and CLI override support.

Loads configs from configs/ directory, merges with base.yaml defaults,
and applies CLI argument overrides. Produces Settings objects or
experiment variant descriptors for the unified runner.

Usage:
    from agentic_rag.config.loader import load_config, load_experiment_config

    # Load a single pipeline config
    cfg = load_config("configs/pipeline/agentic.yaml")

    # Load an experiment config (returns name, description, variants)
    exp = load_experiment_config("configs/experiment/rq1.yaml")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from agentic_rag.config.settings import (
    PROJECT_ROOT,
    EvaluationSettings,
    ExperimentSettings,
    ModelSettings,
    RetrievalSettings,
    RLMSettings,
    Settings,
    settings,
)

CONFIGS_DIR = PROJECT_ROOT / "configs"

# Pipeline class name → module path mapping
PIPELINE_REGISTRY: dict[str, str] = {
    "naive": "agentic_rag.pipeline.naive.NaiveRAGPipeline",
    "crag": "agentic_rag.pipeline.crag.CRAGReplicaPipeline",
    "loop": "agentic_rag.pipeline.loop.LoopRAGPipeline",
    "agentic": "agentic_rag.pipeline.agentic.AgenticRAGPipeline",
}


# ---------------------------------------------------------------------------
# YAML loading utilities
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a YAML config merged with base.yaml defaults.

    Args:
        config_path: Path to the YAML config file.
        overrides: Additional overrides (e.g., from CLI args).

    Returns:
        Merged configuration dict.
    """
    base_path = CONFIGS_DIR / "base.yaml"
    base = _load_yaml(base_path) if base_path.exists() else {}

    config = _load_yaml(Path(config_path))
    merged = _deep_merge(base, config)

    if overrides:
        merged = _deep_merge(merged, overrides)

    return merged


def apply_settings(config: dict[str, Any]) -> Settings:
    """Apply a merged config dict to the global settings singleton.

    Mutates the global `settings` object in-place and returns it.
    This preserves the singleton pattern used by pipeline code.
    """
    section_map = {
        "model": (settings.model, ModelSettings),
        "retrieval": (settings.retrieval, RetrievalSettings),
        "evaluation": (settings.evaluation, EvaluationSettings),
        "experiment": (settings.experiment, ExperimentSettings),
        "rlm": (settings.rlm, RLMSettings),
    }

    for section_key, (section_obj, _section_cls) in section_map.items():
        section_data = config.get(section_key, {})
        for field_name, value in section_data.items():
            if hasattr(section_obj, field_name):
                setattr(section_obj, field_name, value)

    return settings


# ---------------------------------------------------------------------------
# Experiment config loading
# ---------------------------------------------------------------------------
class VariantConfig:
    """Describes one experiment variant (pipeline + settings overrides)."""

    def __init__(
        self,
        name: str,
        pipeline: str,
        overrides: dict[str, Any] | None = None,
        optimization: str | None = None,
    ) -> None:
        self.name = name
        self.pipeline = pipeline
        self.overrides = overrides or {}
        self.optimization = optimization

    @property
    def pipeline_class_path(self) -> str:
        """Return the fully qualified class path for this pipeline."""
        return PIPELINE_REGISTRY[self.pipeline]

    def import_pipeline_class(self):
        """Dynamically import and return the pipeline class."""
        class_path = self.pipeline_class_path
        module_path, class_name = class_path.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def __repr__(self) -> str:
        return f"VariantConfig(name={self.name!r}, pipeline={self.pipeline!r})"


class ExperimentConfig:
    """Describes a full experiment with multiple variants."""

    def __init__(
        self,
        name: str,
        description: str,
        variants: list[VariantConfig],
        train_size: int | None = None,
        val_size: int | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.variants = variants
        self.train_size = train_size
        self.val_size = val_size

    def __repr__(self) -> str:
        return f"ExperimentConfig(name={self.name!r}, variants={len(self.variants)})"


def load_experiment_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load an experiment YAML config and return an ExperimentConfig.

    Args:
        config_path: Path to experiment YAML (e.g., configs/experiment/rq1.yaml).
        overrides: Global overrides applied to all variants.

    Returns:
        ExperimentConfig with parsed variants.
    """
    raw = _load_yaml(Path(config_path))

    variants = []
    for v in raw.get("variants", []):
        # Build per-variant overrides from experiment/model/retrieval sections
        variant_overrides = {}
        for section in ("experiment", "model", "retrieval", "evaluation", "rlm"):
            if section in v:
                variant_overrides[section] = v[section]

        # Merge global overrides into each variant
        if overrides:
            variant_overrides = _deep_merge(variant_overrides, overrides)

        variants.append(
            VariantConfig(
                name=v["name"],
                pipeline=v.get("pipeline", "loop"),
                overrides=variant_overrides,
                optimization=v.get("optimization"),
            )
        )

    return ExperimentConfig(
        name=raw.get("name", "Experiment"),
        description=raw.get("description", ""),
        variants=variants,
        train_size=raw.get("train_size"),
        val_size=raw.get("val_size"),
    )


def load_ablation_configs(
    ablation_dir: str | Path | None = None,
) -> list[VariantConfig]:
    """Load all ablation YAML configs from a directory.

    Args:
        ablation_dir: Path to ablation configs directory.
            Defaults to configs/ablation/.

    Returns:
        List of VariantConfig objects, one per ablation file.
    """
    ablation_dir = Path(ablation_dir) if ablation_dir else CONFIGS_DIR / "ablation"
    configs = []

    for yaml_path in sorted(ablation_dir.glob("*.yaml")):
        raw = _load_yaml(yaml_path)

        variant_overrides = {}
        for section in ("experiment", "model", "retrieval", "evaluation", "rlm"):
            if section in raw:
                variant_overrides[section] = raw[section]

        configs.append(
            VariantConfig(
                name=raw.get("name", yaml_path.stem),
                pipeline=raw.get("pipeline", "loop"),
                overrides=variant_overrides,
            )
        )

    return configs
