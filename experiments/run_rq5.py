"""RQ5: Does RLM-based agentic refinement improve retrieval quality
compared to fixed-loop refinement, and at what cost trade-off?

Compares:
  - Proposed Full (for-loop refinement, C1-C5)
  - Proposed + RLM (agentic refinement, C1-C6)
  - Proposed w/o Iteration (single-pass, reference baseline)

Key metrics:
  - Quality: EM, F1, ROUGE-L
  - Efficiency: search calls, LLM calls, total cost (USD), latency

Expected outcome: RLM >= for-loop on quality, with higher but bounded cost.

Usage:
  uv run python experiments/run_rq5.py --dataset popqa --sample 100
"""

from __future__ import annotations

import argparse
import sys
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


def run_rq5(dataset_name: str = "popqa", sample_size: int | None = None):
    """Execute RQ5 experiment: for-loop vs RLM refinement."""
    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever()

    all_results = {}

    # --- Variant 1: Proposed w/o Iteration (single-pass reference) ---
    logger.info("Running Proposed w/o Iteration (reference baseline)...")
    settings.experiment.enable_iteration = False
    settings.experiment.enable_rlm_refinement = False
    wo_iter = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["Single-Pass"] = run_pipeline_on_dataset(wo_iter, dataset, "rq5_single_pass")

    # --- Variant 2: Proposed Full (for-loop) ---
    logger.info("Running Proposed Full (for-loop refinement)...")
    settings.experiment.enable_iteration = True
    settings.experiment.enable_rlm_refinement = False
    full_loop = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["For-Loop Refinement"] = run_pipeline_on_dataset(full_loop, dataset, "rq5_for_loop")

    # --- Variant 3: Proposed + RLM (agentic refinement) ---
    logger.info("Running Proposed + RLM (agentic refinement)...")
    settings.experiment.enable_iteration = True  # RLM handles iteration internally
    settings.experiment.enable_rlm_refinement = True
    full_rlm = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["RLM Refinement"] = run_pipeline_on_dataset(full_rlm, dataset, "rq5_rlm")

    # --- Reset settings ---
    settings.experiment.enable_iteration = True
    settings.experiment.enable_rlm_refinement = False

    # --- Results ---
    print_comparison_table(
        all_results,
        title=f"RQ5: For-Loop vs RLM Refinement ({dataset_name})",
    )

    # --- Cost comparison ---
    _print_cost_comparison(all_results)

    for name, results in all_results.items():
        save_results(
            results,
            f"rq5_{name.lower().replace(' ', '_').replace('-', '_')}",
            {
                "rq": "RQ5",
                "dataset": dataset_name,
                "variant": name,
            },
        )

    return all_results


def _print_cost_comparison(all_results: dict[str, list[dict]]) -> None:
    """Print efficiency comparison between variants."""
    logger.info("\n=== RQ5: Efficiency Comparison ===")

    for name, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        if not valid:
            continue

        avg_retries = sum(r.get("retry_count", 0) for r in valid) / len(valid)
        avg_llm_calls = sum(r.get("llm_calls", 0) for r in valid) / len(valid)
        avg_latency = sum(r.get("latency", 0) for r in valid) / len(valid)

        logger.info(
            f"  {name:25s}: "
            f"avg_retries={avg_retries:.1f}, "
            f"avg_llm_calls={avg_llm_calls:.1f}, "
            f"avg_latency={avg_latency:.2f}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ5: For-loop vs RLM refinement")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_rq5(args.dataset, args.sample)
