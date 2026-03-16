"""RQ4: Does DSPy automatic optimization improve pipeline performance
and maintainability compared to manual prompt engineering?

Compares:
  - Manual Prompt baseline (hand-crafted prompts, no DSPy optimization)
  - DSPy Unoptimized (DSPy Signatures, no optimization)
  - DSPy + BootstrapFewShot (auto-generated few-shot demos)
  - DSPy + MIPROv2 (auto-optimized instructions + demos)

Additional analysis:
  - Token efficiency (prompt tokens per call)
  - Optimization cost (API calls for optimization)
  - Prompt stability (variance across runs)

Usage:
  uv run python experiments/run_rq4.py --dataset popqa --sample 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from agentic_rag.config.settings import settings
from agentic_rag.optimization.bootstrap import optimize_bootstrap
from agentic_rag.optimization.collector import TrainingCollector
from agentic_rag.optimization.mipro import optimize_mipro
from agentic_rag.pipeline.self_corrective import SelfCorrectiveRAGPipeline
from experiments.common import (
    load_dataset,
    load_retriever,
    print_comparison_table,
    run_pipeline_on_dataset,
    save_results,
    setup_experiment,
)


def run_rq4(
    dataset_name: str = "popqa",
    sample_size: int | None = None,
    train_size: int = 50,
    val_size: int = 20,
):
    """Execute RQ4 experiment."""
    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever()

    # Split dataset: train / val / test
    train_data = dataset[:train_size]
    val_data = dataset[train_size : train_size + val_size]
    test_data = dataset[train_size + val_size :]

    if not test_data:
        logger.warning("No test data left after split. Using val_data as test.")
        test_data = val_data

    logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    all_results = {}

    # --- Variant 1: Manual Prompts (no DSPy) ---
    logger.info("Running Manual Prompt baseline...")
    settings.experiment.enable_dspy = False
    manual_pipeline = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["Manual Prompt"] = run_pipeline_on_dataset(
        manual_pipeline, test_data, "manual_prompt"
    )
    settings.experiment.enable_dspy = True

    # --- Variant 2: DSPy Unoptimized ---
    logger.info("Running DSPy Unoptimized...")
    unopt_pipeline = SelfCorrectiveRAGPipeline(retriever, indexer)
    all_results["DSPy Unoptimized"] = run_pipeline_on_dataset(
        unopt_pipeline, test_data, "dspy_unoptimized"
    )

    # --- Collect training data for optimization ---
    logger.info("Collecting training data...")
    collector = _collect_training_data(retriever, indexer, train_data)

    # --- Variant 3: DSPy + BootstrapFewShot ---
    logger.info("Running DSPy + BootstrapFewShot...")
    bootstrap_pipeline = _optimize_pipeline_bootstrap(retriever, indexer, collector)
    if bootstrap_pipeline:
        all_results["DSPy + Bootstrap"] = run_pipeline_on_dataset(
            bootstrap_pipeline, test_data, "dspy_bootstrap"
        )

    # --- Variant 4: DSPy + MIPROv2 ---
    logger.info("Running DSPy + MIPROv2...")
    mipro_pipeline = _optimize_pipeline_mipro(retriever, indexer, collector)
    if mipro_pipeline:
        all_results["DSPy + MIPROv2"] = run_pipeline_on_dataset(
            mipro_pipeline, test_data, "dspy_mipro"
        )

    # --- Results ---
    print_comparison_table(all_results, title=f"RQ4: DSPy Optimization Effect ({dataset_name})")

    for name, results in all_results.items():
        save_results(
            results,
            f"rq4_{name.lower().replace(' ', '_')}",
            {
                "rq": "RQ4",
                "dataset": dataset_name,
                "variant": name,
                "train_size": len(train_data),
            },
        )

    return all_results


def _collect_training_data(
    retriever,
    indexer,
    train_data: list[dict],
) -> TrainingCollector:
    """Run pipeline on training data to collect examples for optimization."""
    collector = TrainingCollector()
    pipeline = SelfCorrectiveRAGPipeline(retriever, indexer)

    for item in train_data:
        try:
            result = pipeline.run(item["question"])

            # Collect generator examples
            if result.answer:
                collector.add(
                    "QnAGenerateSignature",
                    inputs={
                        "question": item["question"],
                        "passages": pipeline.format_passages(result.passages_used),
                        "system_prompt": "You are a helpful knowledge assistant.",
                    },
                    outputs={
                        "answer": result.answer,
                        "footnotes": result.footnotes,
                        "recommended_questions": result.recommended_questions,
                    },
                    reference=item.get("answer", ""),
                )

            # Collect evaluation examples
            for ev in result.evaluation_scores:
                if "total" in ev:
                    collector.add(
                        "EvaluationSignature",
                        inputs={
                            "question": item["question"],
                            "passages": pipeline.format_passages(result.passages_used),
                            "retry_count": ev.get("retry", 0),
                            "max_retry": settings.evaluation.max_retry_count,
                        },
                        outputs={
                            "relevance_score": ev.get("relevance", 0),
                            "coverage_score": ev.get("coverage", 0),
                            "specificity_score": ev.get("specificity", 0),
                            "sufficiency_score": ev.get("sufficiency", 0),
                            "total_score": ev.get("total", 0),
                            "action": ev.get("action", "output"),
                            "keywords_to_add": ev.get("keywords_to_add", []),
                            "keywords_to_remove": ev.get("keywords_to_remove", []),
                            "suggested_query": ev.get("suggested_query", ""),
                            "reasoning": ev.get("reasoning", ""),
                        },
                    )
        except Exception as e:
            logger.warning(f"Training collection error: {e}")

    logger.info(f"Collected training data: {collector.summary()}")
    collector.save(settings.results_dir / "rq4_training_data.json")
    return collector


def _optimize_pipeline_bootstrap(
    retriever,
    indexer,
    collector: TrainingCollector,
) -> SelfCorrectiveRAGPipeline | None:
    """Create a pipeline with BootstrapFewShot-optimized modules."""
    try:
        pipeline = SelfCorrectiveRAGPipeline(retriever, indexer)

        # Optimize generator
        gen_trainset = collector.to_dspy_examples("QnAGenerateSignature")
        if gen_trainset:
            pipeline.generator = optimize_bootstrap(pipeline.generator, gen_trainset, max_demos=4)

        return pipeline
    except Exception as e:
        logger.error(f"BootstrapFewShot optimization failed: {e}")
        return None


def _optimize_pipeline_mipro(
    retriever,
    indexer,
    collector: TrainingCollector,
) -> SelfCorrectiveRAGPipeline | None:
    """Create a pipeline with MIPROv2-optimized modules."""
    try:
        pipeline = SelfCorrectiveRAGPipeline(retriever, indexer)

        gen_trainset = collector.to_dspy_examples("QnAGenerateSignature")
        if gen_trainset:
            pipeline.generator = optimize_mipro(pipeline.generator, gen_trainset, num_candidates=7)

        return pipeline
    except Exception as e:
        logger.error(f"MIPROv2 optimization failed: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ4: DSPy optimization effect")
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--val-size", type=int, default=20)
    args = parser.parse_args()
    run_rq4(args.dataset, args.sample, args.train_size, args.val_size)
