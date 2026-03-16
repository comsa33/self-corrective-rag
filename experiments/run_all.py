"""Run all experiments in sequence.

Master script that executes RQ1-RQ4 + ablation study across
all configured datasets. Generates a unified report.

Usage:
  # Full experiment (all datasets, all questions)
  uv run python experiments/run_all.py

  # Quick validation run
  uv run python experiments/run_all.py --sample 20

  # Specific dataset
  uv run python experiments/run_all.py --dataset popqa --sample 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console

from agentic_rag.config.settings import settings
from experiments.run_ablation import run_ablation
from experiments.run_rq1 import run_rq1
from experiments.run_rq2 import run_rq2
from experiments.run_rq3 import run_rq3
from experiments.run_rq4 import run_rq4
from experiments.run_rq5 import run_rq5

console = Console()


def run_all(
    datasets: list[str] | None = None,
    sample_size: int | None = None,
    skip: list[str] | None = None,
):
    """Run all experiments."""
    datasets = datasets or settings.experiment.datasets
    skip = skip or []
    total_start = time.time()

    logger.info("Starting full experiment suite")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Sample size: {sample_size or 'full'}")
    logger.info(f"  Skip: {skip or 'none'}")

    all_summaries = {}

    for dataset_name in datasets:
        logger.info(f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}")
        dataset_results = {}

        # RQ1: Iterative loop
        if "rq1" not in skip:
            logger.info("\n--- RQ1: Iterative Loop Effect ---")
            try:
                dataset_results["rq1"] = run_rq1(dataset_name, sample_size)
            except Exception as e:
                logger.error(f"RQ1 failed: {e}")

        # RQ2: 4D evaluation
        if "rq2" not in skip:
            logger.info("\n--- RQ2: 4D Evaluation Effect ---")
            try:
                dataset_results["rq2"] = run_rq2(dataset_name, sample_size)
            except Exception as e:
                logger.error(f"RQ2 failed: {e}")

        # RQ3: Query refinement
        if "rq3" not in skip:
            logger.info("\n--- RQ3: Query Refinement Effect ---")
            try:
                dataset_results["rq3"] = run_rq3(dataset_name, sample_size)
            except Exception as e:
                logger.error(f"RQ3 failed: {e}")

        # RQ4: DSPy optimization
        if "rq4" not in skip:
            logger.info("\n--- RQ4: DSPy Optimization Effect ---")
            try:
                dataset_results["rq4"] = run_rq4(dataset_name, sample_size)
            except Exception as e:
                logger.error(f"RQ4 failed: {e}")

        # RQ5: RLM refinement
        if "rq5" not in skip:
            logger.info("\n--- RQ5: RLM Refinement Effect ---")
            try:
                dataset_results["rq5"] = run_rq5(dataset_name, sample_size)
            except Exception as e:
                logger.error(f"RQ5 failed: {e}")

        # Ablation study
        if "ablation" not in skip:
            logger.info("\n--- Ablation Study ---")
            try:
                dataset_results["ablation"] = run_ablation(dataset_name, sample_size)
            except Exception as e:
                logger.error(f"Ablation failed: {e}")

        all_summaries[dataset_name] = dataset_results

    total_time = time.time() - total_start

    # Save master summary
    summary_path = settings.results_dir / "experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_time_seconds": total_time,
                "datasets": datasets,
                "sample_size": sample_size,
                "settings": {
                    "seed": settings.experiment.seed,
                    "quality_threshold": settings.evaluation.quality_threshold,
                    "max_retry": settings.evaluation.max_retry_count,
                    "top_k": settings.retrieval.top_k,
                    "models": {
                        "preprocess": settings.model.preprocess_model,
                        "evaluate": settings.model.evaluate_model,
                        "generate": settings.model.generate_model,
                        "agent": settings.model.agent_model,
                        "embedding": settings.model.embedding_model,
                    },
                },
            },
            f,
            indent=2,
        )

    console.print("\n[bold green]All experiments complete![/]")
    console.print(f"Total time: {total_time / 60:.1f} minutes")
    console.print(f"Results saved to: {settings.results_dir}")

    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument(
        "--dataset", nargs="*", default=None, help="Datasets to run (default: all configured)"
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Experiments to skip: rq1, rq2, rq3, rq4, rq5, ablation",
    )
    args = parser.parse_args()
    run_all(args.dataset, args.sample, args.skip)
