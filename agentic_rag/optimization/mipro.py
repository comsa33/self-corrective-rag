"""MIPROv2 optimizer integration.

MIPROv2 (Multi-prompt Instruction Proposal Optimizer) automatically
generates and optimizes both instructions and few-shot demonstrations.
More powerful than BootstrapFewShot but requires more API calls.
"""

from __future__ import annotations

import dspy
from loguru import logger


def optimize_mipro(
    module: dspy.Module,
    trainset: list[dspy.Example],
    metric_fn=None,
    num_candidates: int = 10,
    max_demos: int = 4,
    eval_kwargs: dict | None = None,
) -> dspy.Module:
    """Optimize a DSPy module using MIPROv2.

    MIPROv2 proposes multiple instruction variants and few-shot
    demonstration sets, then evaluates them to find the best combination.

    Args:
        module: The DSPy module to optimize.
        trainset: Training examples as dspy.Example list.
        metric_fn: Evaluation metric function.
        num_candidates: Number of instruction candidates to propose.
        max_demos: Maximum few-shot demonstrations per candidate.
        eval_kwargs: Additional kwargs for the evaluation step.

    Returns:
        Optimized DSPy module with best instructions and demos.
    """
    if metric_fn is None:
        metric_fn = _default_metric

    optimizer = dspy.MIPROv2(
        metric=metric_fn,
        auto="light",  # light | medium | heavy — sets num_candidates/trials automatically
    )

    logger.info(f"Running MIPROv2: trainset={len(trainset)}, auto=light, max_demos={max_demos}")

    eval_kwargs = eval_kwargs or {}
    optimized = optimizer.compile(
        module,
        trainset=trainset,
        **eval_kwargs,
    )

    logger.info("MIPROv2 optimization complete")
    return optimized


def compare_optimizers(
    module: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    metric_fn,
) -> dict[str, float]:
    """Compare BootstrapFewShot vs MIPROv2 on a validation set.

    Used in RQ4 experiments to evaluate DSPy optimization impact.

    Returns:
        Dict with scores for each optimizer variant.
    """
    from agentic_rag.optimization.bootstrap import optimize_bootstrap

    results = {}

    # Baseline: unoptimized
    baseline_scores = _evaluate_module(module, valset, metric_fn)
    results["unoptimized"] = baseline_scores
    logger.info(f"Baseline (unoptimized): {baseline_scores:.4f}")

    # BootstrapFewShot
    bootstrap_module = optimize_bootstrap(module, trainset, metric_fn, max_demos=4)
    bootstrap_scores = _evaluate_module(bootstrap_module, valset, metric_fn)
    results["bootstrap_fewshot"] = bootstrap_scores
    logger.info(f"BootstrapFewShot: {bootstrap_scores:.4f}")

    # MIPROv2
    mipro_module = optimize_mipro(module, trainset, metric_fn, num_candidates=10)
    mipro_scores = _evaluate_module(mipro_module, valset, metric_fn)
    results["miprov2"] = mipro_scores
    logger.info(f"MIPROv2: {mipro_scores:.4f}")

    return results


def _evaluate_module(
    module: dspy.Module,
    valset: list[dspy.Example],
    metric_fn,
) -> float:
    """Evaluate a module on a validation set."""
    scores = []
    for example in valset:
        try:
            inputs = {k: example[k] for k in example.inputs()}
            prediction = module(**inputs)
            score = metric_fn(example, prediction)
            scores.append(score)
        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def _default_metric(example, prediction, trace=None) -> float:
    """Default metric: check non-empty outputs."""
    for key in example:
        if key in example.inputs():
            continue
        pred_val = getattr(prediction, key, None)
        if pred_val is None or (isinstance(pred_val, str) and not pred_val.strip()):
            return 0.0
    return 1.0
