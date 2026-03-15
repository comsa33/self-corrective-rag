"""RQ3: Does targeted query refinement improve re-retrieval efficiency
compared to blind re-search or web search fallback?

Compares:
  - CRAG Replica (web search fallback on failure)
  - Proposed w/o Refinement (iterate with same query, no keyword changes)
  - Proposed Full (targeted refinement: keywords_to_add/remove + suggested_query)

Additional analysis:
  - Retrieval improvement per retry (unique new passages found)
  - Keyword change effectiveness (how often added keywords appear in results)
  - Convergence speed (retries needed to reach threshold)

Usage:
  uv run python experiments/run_rq3.py --dataset popqa --sample 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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
from src.pipeline.crag_replica import CRAGReplicaPipeline
from src.pipeline.self_corrective import SelfCorrectiveRAGPipeline


def run_rq3(dataset_name: str = "popqa", sample_size: int | None = None):
    """Execute RQ3 experiment."""
    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever()

    all_results = {}

    # --- Variant 1: CRAG (web search fallback) ---
    logger.info("Running CRAG Replica (web search fallback)...")
    crag = CRAGReplicaPipeline(retriever, indexer)
    all_results["CRAG (Web Fallback)"] = run_pipeline_on_dataset(crag, dataset, "crag_web_fallback")

    # --- Variant 2: Proposed w/o Refinement ---
    logger.info("Running Proposed w/o Refinement...")
    settings.experiment.enable_iteration = True
    settings.experiment.enable_accumulation = True
    settings.experiment.enable_4d_evaluation = True
    settings.experiment.enable_refinement = False  # key ablation
    wo_refine = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["Proposed w/o Refinement"] = run_pipeline_on_dataset(
        wo_refine, dataset, "proposed_wo_refinement"
    )

    # --- Variant 3: Proposed Full (with refinement) ---
    logger.info("Running Proposed Full (with refinement)...")
    settings.experiment.enable_refinement = True
    full = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["Proposed Full"] = run_pipeline_on_dataset(full, dataset, "proposed_full")

    # --- Results ---
    print_comparison_table(all_results, title=f"RQ3: Query Refinement Effect ({dataset_name})")
    _analyze_refinement_efficiency(all_results)

    for name, results in all_results.items():
        save_results(
            results,
            f"rq3_{name.lower().replace(' ', '_')}",
            {
                "rq": "RQ3",
                "dataset": dataset_name,
                "variant": name,
            },
        )

    return all_results


def _analyze_refinement_efficiency(all_results: dict[str, list[dict]]) -> None:
    """Analyze retry efficiency across variants."""
    logger.info("\n=== Refinement Efficiency Analysis ===")

    for name, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        if not valid:
            continue

        retries = [r.get("retry_count", 0) for r in valid]
        passages_retrieved = [r.get("total_passages_retrieved", 0) for r in valid]

        # Count how many needed retries
        needed_retry = sum(1 for r in retries if r > 0)
        reached_max = sum(1 for r in retries if r >= settings.evaluation.max_retry_count)
        routed_to_agent = sum(1 for r in valid if r.get("agent_type") is not None)

        # Score progression across retries
        score_progressions = []
        for r in valid:
            scores = [ev.get("total", 0) for ev in r.get("evaluation_scores", []) if "total" in ev]
            if len(scores) > 1:
                score_progressions.append(scores[-1] - scores[0])

        logger.info(f"\n--- {name} ---")
        logger.info(f"  Avg retries: {np.mean(retries):.2f}")
        logger.info(
            f"  Needed retry: {needed_retry}/{len(valid)} ({needed_retry / len(valid) * 100:.1f}%)"
        )
        logger.info(
            f"  Reached max retry: {reached_max}/{len(valid)} ({reached_max / len(valid) * 100:.1f}%)"
        )
        logger.info(f"  Routed to agent: {routed_to_agent}/{len(valid)}")
        logger.info(f"  Avg passages retrieved: {np.mean(passages_retrieved):.1f}")

        if score_progressions:
            logger.info(
                f"  Score improvement (retry items): "
                f"mean={np.mean(score_progressions):.1f}, "
                f"median={np.median(score_progressions):.1f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ3: Query refinement effect")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_rq3(args.dataset, args.sample)
