"""BootstrapFewShot optimizer integration.

Automatically generates few-shot demonstrations for DSPy modules
by running the pipeline on training examples and selecting
the best-performing ones as in-context demos.
"""

from __future__ import annotations

from pathlib import Path

import dspy
from loguru import logger

from src.optimization.collector import TrainingCollector


def create_trainset(
    collector: TrainingCollector,
    signature_name: str,
    max_examples: int = 50,
) -> list[dspy.Example]:
    """Convert collected training data to DSPy trainset format."""
    examples = collector.to_dspy_examples(signature_name)
    if len(examples) > max_examples:
        examples = examples[:max_examples]
    logger.info(f"Created trainset for {signature_name}: {len(examples)} examples")
    return examples


def optimize_bootstrap(
    module: dspy.Module,
    trainset: list[dspy.Example],
    metric_fn=None,
    max_demos: int = 4,
    max_rounds: int = 1,
) -> dspy.Module:
    """Optimize a DSPy module using BootstrapFewShot.

    Args:
        module: The DSPy module to optimize (e.g., ChainOfThought(Signature)).
        trainset: Training examples as dspy.Example list.
        metric_fn: Evaluation metric function(example, prediction, trace) -> float.
                   If None, uses a default "non-empty output" metric.
        max_demos: Maximum number of few-shot demonstrations to bootstrap.
        max_rounds: Number of bootstrap rounds.

    Returns:
        Optimized DSPy module with bootstrapped demonstrations.
    """
    if metric_fn is None:
        metric_fn = _default_metric

    optimizer = dspy.BootstrapFewShot(
        metric=metric_fn,
        max_bootstrapped_demos=max_demos,
        max_rounds=max_rounds,
    )

    logger.info(
        f"Running BootstrapFewShot: "
        f"trainset={len(trainset)}, max_demos={max_demos}, rounds={max_rounds}"
    )
    optimized = optimizer.compile(module, trainset=trainset)
    logger.info("BootstrapFewShot optimization complete")
    return optimized


def save_optimized(module: dspy.Module, path: Path) -> None:
    """Save optimized module state (demos) to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    module.save(str(path))
    logger.info(f"Saved optimized module to {path}")


def load_optimized(module: dspy.Module, path: Path) -> dspy.Module:
    """Load optimized module state from disk."""
    module.load(str(path))
    logger.info(f"Loaded optimized module from {path}")
    return module


def _default_metric(example, prediction, trace=None) -> float:
    """Default metric: check that all output fields are non-empty."""
    for key in example:
        if key in example.inputs():
            continue
        pred_val = getattr(prediction, key, None)
        if pred_val is None or (isinstance(pred_val, str) and not pred_val.strip()):
            return 0.0
    return 1.0
