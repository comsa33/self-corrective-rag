"""Ablation Study: Systematically disable each contribution to measure
its individual impact on overall pipeline performance.

Variants (from CLAUDE_CODE_CONTEXT.md §4.5):
  1. Full System          — All contributions enabled
  2. w/o Iteration       — C1: remove retry loop (single-pass)
  3. w/o Accumulation    — C1: no passage accumulation (reset each retry)
  4. 1D Evaluation       — C2: single relevance score instead of 4D
  5. w/o Refinement      — C3: no keyword changes on retry
  6. w/o Agent           — C4: no agent routing (force generate after max retry)
  7. Manual Prompt       — C5: disable DSPy optimization

Usage:
  uv run python experiments/run_ablation.py --dataset popqa --sample 100
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config.settings import settings
from experiments.common import (
    load_dataset,
    load_retriever,
    print_comparison_table,
    run_pipeline_on_dataset,
    save_results,
    setup_experiment,
)
from src.pipeline.self_corrective import SelfCorrectiveRAGPipeline


@dataclass
class AblationVariant:
    """Configuration for one ablation variant."""

    name: str
    contribution: str  # C1-C5
    description: str
    enable_iteration: bool = True
    enable_accumulation: bool = True
    enable_4d_evaluation: bool = True
    enable_refinement: bool = True
    enable_agent_routing: bool = True
    enable_dspy: bool = True


ABLATION_VARIANTS = [
    AblationVariant(
        name="Full System",
        contribution="All",
        description="All contributions enabled (proposed method)",
    ),
    AblationVariant(
        name="w/o Iteration",
        contribution="C1",
        description="Remove iterative retry loop (single-pass only)",
        enable_iteration=False,
    ),
    AblationVariant(
        name="w/o Accumulation",
        contribution="C1",
        description="No passage accumulation (reset passages each retry)",
        enable_accumulation=False,
    ),
    AblationVariant(
        name="1D Evaluation",
        contribution="C2",
        description="Single relevance score instead of 4D assessment",
        enable_4d_evaluation=False,
    ),
    AblationVariant(
        name="w/o Refinement",
        contribution="C3",
        description="No keyword refinement on retry (same query re-used)",
        enable_refinement=False,
    ),
    AblationVariant(
        name="w/o Agent",
        contribution="C4",
        description="No agent routing (force generate after max retry)",
        enable_agent_routing=False,
    ),
    AblationVariant(
        name="Manual Prompt",
        contribution="C5",
        description="Disable DSPy (use manual prompts)",
        enable_dspy=False,
    ),
]


def run_ablation(
    dataset_name: str = "popqa",
    sample_size: int | None = None,
    variants: list[str] | None = None,
):
    """Execute ablation study."""
    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever()

    # Filter variants if specified
    active_variants = ABLATION_VARIANTS
    if variants:
        active_variants = [
            v for v in ABLATION_VARIANTS if v.name in variants or v.contribution in variants
        ]

    all_results = {}

    for variant in active_variants:
        logger.info(f"Running ablation: {variant.name} ({variant.description})")

        # Apply settings
        settings.experiment.enable_iteration = variant.enable_iteration
        settings.experiment.enable_accumulation = variant.enable_accumulation
        settings.experiment.enable_4d_evaluation = variant.enable_4d_evaluation
        settings.experiment.enable_refinement = variant.enable_refinement
        settings.experiment.enable_agent_routing = variant.enable_agent_routing
        settings.experiment.enable_dspy = variant.enable_dspy

        pipeline = SelfCorrectiveRAGPipeline(retriever, indexer)

        results = run_pipeline_on_dataset(pipeline, dataset, variant.name.lower().replace(" ", "_"))
        all_results[variant.name] = results

    # Reset settings to defaults
    settings.experiment.enable_iteration = True
    settings.experiment.enable_accumulation = True
    settings.experiment.enable_4d_evaluation = True
    settings.experiment.enable_refinement = True
    settings.experiment.enable_agent_routing = True
    settings.experiment.enable_dspy = True

    # --- Results ---
    print_comparison_table(all_results, title=f"Ablation Study ({dataset_name})")

    # Print contribution impact summary
    _print_ablation_impact(all_results)

    for name, results in all_results.items():
        save_results(
            results,
            f"ablation_{name.lower().replace(' ', '_')}",
            {
                "experiment": "ablation",
                "dataset": dataset_name,
                "variant": name,
            },
        )

    return all_results


def _print_ablation_impact(all_results: dict[str, list[dict]]) -> None:
    """Compute and print the impact of removing each contribution."""
    from src.evaluation.metrics import evaluate_batch

    logger.info("\n=== Contribution Impact Summary ===")

    # Get full system metrics as baseline
    full_results = all_results.get("Full System", [])
    if not full_results:
        return

    full_valid = [r for r in full_results if "error" not in r]
    full_preds = [r["prediction"] for r in full_valid]
    full_refs = [r["reference"] for r in full_valid]
    full_metrics = evaluate_batch(full_preds, full_refs, compute_bert_score=False)

    for variant in ABLATION_VARIANTS:
        if variant.name == "Full System":
            continue

        results = all_results.get(variant.name, [])
        if not results:
            continue

        valid = [r for r in results if "error" not in r]
        preds = [r["prediction"] for r in valid]
        refs = [r["reference"] for r in valid]
        metrics = evaluate_batch(preds, refs, compute_bert_score=False)

        # Compute deltas
        em_delta = metrics["exact_match"] - full_metrics["exact_match"]
        f1_delta = metrics["f1"] - full_metrics["f1"]

        logger.info(
            f"  {variant.name:25s} ({variant.contribution}): "
            f"EM {em_delta:+.3f}, F1 {f1_delta:+.3f} "
            f"{'↓ quality loss' if f1_delta < 0 else '↑ improvement'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Specific variants or contributions to run (e.g., 'C1' 'w/o Agent')",
    )
    args = parser.parse_args()
    run_ablation(args.dataset, args.sample, args.variants)
