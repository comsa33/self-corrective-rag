"""RQ2: Does 4-dimensional quality assessment improve corrective action
appropriateness compared to single-dimension relevance evaluation?

Compares:
  - CRAG Replica (binary: correct/incorrect/ambiguous)
  - Proposed 1D Evaluation (single relevance score, threshold-based)
  - Proposed 4D Evaluation (Relevance + Coverage + Specificity + Sufficiency)

Additional analysis:
  - Action accuracy: how often each evaluation method triggers the correct action
  - Score distribution: how 4D scores distribute across question types
  - False positive/negative rates for "refine" decisions

Usage:
  uv run python experiments/run_rq2.py --dataset popqa --sample 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.pipeline.crag import CRAGReplicaPipeline
from agentic_rag.pipeline.self_corrective import SelfCorrectiveRAGPipeline
from experiments.common import (
    load_dataset,
    load_retriever,
    print_comparison_table,
    run_pipeline_on_dataset,
    save_results,
    setup_experiment,
)


# ---------------------------------------------------------------------------
# 1D Evaluation Signature (ablation variant)
# ---------------------------------------------------------------------------
class SingleDimEvaluationSignature(dspy.Signature):
    """Single-dimension relevance evaluation (ablation for C2).

    Only evaluates overall relevance on a 0-100 scale,
    without the 4D breakdown.
    """

    question: str = dspy.InputField()
    passages: str = dspy.InputField()
    retry_count: int = dspy.InputField()
    max_retry: int = dspy.InputField()

    relevance_score: int = dspy.OutputField(desc="Overall relevance score (0-100).")
    total_score: int = dspy.OutputField(
        desc="Same as relevance_score (for interface compatibility)."
    )
    action: str = dspy.OutputField(desc='"output" | "refine" | "route_to_agent".')
    keywords_to_add: list[str] = dspy.OutputField(desc="Keywords to add.")
    keywords_to_remove: list[str] = dspy.OutputField(desc="Keywords to remove.")
    suggested_query: str = dspy.OutputField(desc="Improved query.")
    reasoning: str = dspy.OutputField(desc="Evaluation rationale.")

    # Dummy fields for interface compatibility
    coverage_score: int = dspy.OutputField(desc="Not used in 1D mode.", default=0)
    specificity_score: int = dspy.OutputField(desc="Not used in 1D mode.", default=0)
    sufficiency_score: int = dspy.OutputField(desc="Not used in 1D mode.", default=0)


def run_rq2(dataset_name: str = "popqa", sample_size: int | None = None):
    """Execute RQ2 experiment."""
    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever()

    all_results = {}

    # --- Variant 1: CRAG binary evaluation ---
    logger.info("Running CRAG Replica (binary evaluation)...")
    crag = CRAGReplicaPipeline(retriever, indexer)
    all_results["CRAG (Binary)"] = run_pipeline_on_dataset(crag, dataset, "crag_binary")

    # --- Variant 2: Proposed with 1D evaluation ---
    logger.info("Running Proposed with 1D Evaluation...")
    settings.experiment.enable_4d_evaluation = True  # use eval, but swap signature
    proposed_1d = SelfCorrectiveRAGPipeline(retriever, indexer)
    # Monkey-patch the evaluator to use 1D signature
    proposed_1d.evaluator = dspy.Predict(SingleDimEvaluationSignature)
    all_results["Proposed (1D)"] = run_pipeline_on_dataset(proposed_1d, dataset, "proposed_1d")

    # --- Variant 3: Proposed with 4D evaluation (full) ---
    logger.info("Running Proposed with 4D Evaluation...")
    settings.experiment.enable_4d_evaluation = True
    proposed_4d = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["Proposed (4D)"] = run_pipeline_on_dataset(proposed_4d, dataset, "proposed_4d")

    # --- Results ---
    print_comparison_table(all_results, title=f"RQ2: 4D vs 1D Evaluation ({dataset_name})")

    # Additional analysis: score distributions
    _analyze_score_distributions(all_results)

    for name, results in all_results.items():
        save_results(
            results,
            f"rq2_{name.lower().replace(' ', '_')}",
            {
                "rq": "RQ2",
                "dataset": dataset_name,
                "variant": name,
            },
        )

    return all_results


def _analyze_score_distributions(all_results: dict[str, list[dict]]) -> None:
    """Analyze and print 4D score distributions."""
    for name, results in all_results.items():
        scores = []
        for r in results:
            for ev in r.get("evaluation_scores", []):
                if "total" in ev:
                    scores.append(ev)

        if not scores:
            continue

        logger.info(f"\n--- {name} Score Distribution ---")
        for dim in ["relevance", "coverage", "specificity", "sufficiency", "total"]:
            vals = [s.get(dim, 0) for s in scores if dim in s]
            if vals:
                logger.info(
                    f"  {dim:15s}: mean={np.mean(vals):.1f}, "
                    f"std={np.std(vals):.1f}, "
                    f"min={min(vals)}, max={max(vals)}"
                )

        actions = [s.get("action", "") for s in scores]
        for action in ["output", "refine", "route_to_agent"]:
            count = actions.count(action)
            logger.info(f"  action={action}: {count} ({count / len(actions) * 100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2: 4D evaluation effect")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run_rq2(args.dataset, args.sample)
