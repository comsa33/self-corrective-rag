"""RQ1: Does the iterative self-corrective loop with passage accumulation
improve answer quality compared to single-pass retrieval?

Compares:
  - Naive RAG (no evaluation, no correction)
  - CRAG Replica (1-pass correction)
  - Proposed w/o Iteration (C1 disabled: single-pass with 4D eval)
  - Proposed w/o Accumulation (C1 partial: iterate but reset passages)
  - Proposed Full (iterative loop + passage accumulation)

Expected outcome: Full > w/o Accumulation > w/o Iteration > CRAG > Naive

Usage:
  uv run python experiments/run_rq1.py --dataset popqa --sample 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.pipeline.crag import CRAGReplicaPipeline
from agentic_rag.pipeline.naive import NaiveRAGPipeline
from agentic_rag.pipeline.self_corrective import SelfCorrectiveRAGPipeline
from experiments.common import (
    load_dataset,
    load_retriever,
    print_comparison_table,
    run_pipeline_on_dataset,
    save_results,
    setup_experiment,
)


def run_rq1(dataset_name: str = "popqa", sample_size: int | None = None):
    """Execute RQ1 experiment."""
    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever()

    all_results = {}

    # --- Variant 1: Naive RAG ---
    logger.info("Running Naive RAG...")
    naive = NaiveRAGPipeline(retriever, indexer)
    all_results["Naive RAG"] = run_pipeline_on_dataset(naive, dataset, "naive_rag")

    # --- Variant 2: CRAG Replica ---
    logger.info("Running CRAG Replica...")
    crag = CRAGReplicaPipeline(retriever, indexer)
    all_results["CRAG Replica"] = run_pipeline_on_dataset(crag, dataset, "crag_replica")

    # --- Variant 3: Proposed w/o Iteration (single-pass, 4D eval) ---
    logger.info("Running Proposed w/o Iteration...")
    settings.experiment.enable_iteration = False
    settings.experiment.enable_accumulation = True
    wo_iter = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["Proposed w/o Iteration"] = run_pipeline_on_dataset(
        wo_iter, dataset, "proposed_wo_iteration"
    )

    # --- Variant 4: Proposed w/o Accumulation (iterate, reset passages) ---
    logger.info("Running Proposed w/o Accumulation...")
    settings.experiment.enable_iteration = True
    settings.experiment.enable_accumulation = False
    wo_accum = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["Proposed w/o Accumulation"] = run_pipeline_on_dataset(
        wo_accum, dataset, "proposed_wo_accumulation"
    )

    # --- Variant 5: Proposed Full ---
    logger.info("Running Proposed Full...")
    settings.experiment.enable_iteration = True
    settings.experiment.enable_accumulation = True
    full = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["Proposed Full"] = run_pipeline_on_dataset(full, dataset, "proposed_full")

    # --- Results ---
    print_comparison_table(all_results, title=f"RQ1: Iterative Loop Effect ({dataset_name})")

    for name, results in all_results.items():
        save_results(
            results,
            f"rq1_{name.lower().replace(' ', '_')}",
            {
                "rq": "RQ1",
                "dataset": dataset_name,
                "variant": name,
            },
        )

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ1: Iterative loop effect")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_rq1(args.dataset, args.sample)
