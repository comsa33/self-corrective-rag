"""Unified config-driven experiment runner.

Runs experiments defined by YAML config files. Supports experiment configs
(multiple variants), ablation configs (directory of variants), and
single pipeline configs.

Usage:
    # Run a single experiment
    uv run python experiments/run.py --config configs/experiment/rq1.yaml --dataset popqa --sample 100

    # Run ablation study
    uv run python experiments/run.py --ablation --dataset popqa --sample 100

    # Run all experiments + ablation
    uv run python experiments/run.py --all --dataset popqa --sample 20

    # Run specific ablation variants
    uv run python experiments/run.py --ablation --variants "Full System" "w/o Iteration"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console

from agentic_rag.config.loader import (
    VariantConfig,
    apply_settings,
    load_ablation_configs,
    load_config,
    load_experiment_config,
)
from agentic_rag.config.settings import settings
from experiments.common import (
    load_dataset,
    load_retriever,
    print_comparison_table,
    run_pipeline_on_dataset,
    save_results,
    setup_experiment,
)

console = Console()

EXPERIMENT_CONFIGS = [
    "configs/experiment/rq1.yaml",
    "configs/experiment/rq2.yaml",
    "configs/experiment/rq3.yaml",
    "configs/experiment/rq4.yaml",
    "configs/experiment/rq5.yaml",
]


# ---------------------------------------------------------------------------
# Variant execution
# ---------------------------------------------------------------------------
def _run_variant(
    variant: VariantConfig,
    dataset: list[dict],
    retriever,
    indexer,
) -> list[dict]:
    """Run a single variant: apply settings, create pipeline, execute."""
    # Apply variant-specific settings
    base_cfg = load_config("configs/base.yaml")
    merged = base_cfg.copy()
    for section, overrides in variant.overrides.items():
        if section in merged:
            merged[section] = {**merged.get(section, {}), **overrides}
        else:
            merged[section] = overrides
    apply_settings(merged)

    # Import and create pipeline
    pipeline_cls = variant.import_pipeline_class()
    pipeline = pipeline_cls(retriever, indexer)

    slug = variant.name.lower().replace(" ", "_").replace("/", "_")
    return run_pipeline_on_dataset(pipeline, dataset, slug)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    config_path: str,
    dataset_name: str = "popqa",
    sample_size: int | None = None,
) -> dict[str, list[dict]]:
    """Run an experiment defined by a YAML config file."""
    exp = load_experiment_config(config_path)
    logger.info(f"Running experiment: {exp.name}")
    logger.info(f"  Description: {exp.description}")
    logger.info(f"  Variants: {len(exp.variants)}")

    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever()

    # Handle train/val split for optimization experiments (e.g., RQ4)
    test_data = dataset
    if exp.train_size is not None:
        train_end = exp.train_size
        val_end = train_end + (exp.val_size or 0)
        test_data = dataset[val_end:]
        if not test_data:
            test_data = dataset[train_end:val_end] or dataset
        logger.info(
            f"  Data split: train={train_end}, val={exp.val_size or 0}, test={len(test_data)}"
        )

    all_results: dict[str, list[dict]] = {}
    for variant in exp.variants:
        logger.info(f"  Running variant: {variant.name}")
        results = _run_variant(variant, test_data, retriever, indexer)
        all_results[variant.name] = results

    # Report & save
    print_comparison_table(all_results, title=f"{exp.name} ({dataset_name})")
    for name, results in all_results.items():
        slug = name.lower().replace(" ", "_").replace("/", "_")
        save_results(
            results,
            f"{Path(config_path).stem}_{slug}",
            {"experiment": exp.name, "dataset": dataset_name, "variant": name},
        )

    return all_results


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
def run_ablation(
    dataset_name: str = "popqa",
    sample_size: int | None = None,
    variant_names: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Run ablation study from configs/ablation/ directory."""
    variants = load_ablation_configs()

    if variant_names:
        variants = [v for v in variants if v.name in variant_names]

    logger.info(f"Running ablation study: {len(variants)} variants")

    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever()

    all_results: dict[str, list[dict]] = {}
    for variant in variants:
        logger.info(f"  Running ablation variant: {variant.name}")
        results = _run_variant(variant, dataset, retriever, indexer)
        all_results[variant.name] = results

    print_comparison_table(all_results, title=f"Ablation Study ({dataset_name})")
    for name, results in all_results.items():
        slug = name.lower().replace(" ", "_").replace("/", "_")
        save_results(
            results,
            f"ablation_{slug}",
            {"experiment": "ablation", "dataset": dataset_name, "variant": name},
        )

    return all_results


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
def run_all(
    dataset_name: str = "popqa",
    sample_size: int | None = None,
    skip: list[str] | None = None,
) -> None:
    """Run all experiments + ablation study."""
    skip = skip or []

    for config_path in EXPERIMENT_CONFIGS:
        exp_name = Path(config_path).stem  # e.g., "rq1"
        if exp_name in skip:
            logger.info(f"Skipping {exp_name}")
            continue
        try:
            run_experiment(config_path, dataset_name, sample_size)
        except Exception as e:
            logger.error(f"{exp_name} failed: {e}")

    if "ablation" not in skip:
        try:
            run_ablation(dataset_name, sample_size)
        except Exception as e:
            logger.error(f"Ablation failed: {e}")

    console.print("\n[bold green]All experiments complete![/]")
    console.print(f"Results saved to: {settings.results_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified config-driven experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  uv run python experiments/run.py --config configs/experiment/rq1.yaml --sample 100
  uv run python experiments/run.py --ablation --sample 50
  uv run python experiments/run.py --all --dataset popqa --sample 20
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study from configs/ablation/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all experiments + ablation",
    )
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Experiments to skip when using --all (e.g., rq4 rq5 ablation)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Specific ablation variant names to run",
    )

    args = parser.parse_args()

    if args.run_all:
        run_all(args.dataset, args.sample, args.skip)
    elif args.ablation:
        run_ablation(args.dataset, args.sample, args.variants)
    elif args.config:
        run_experiment(args.config, args.dataset, args.sample)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
