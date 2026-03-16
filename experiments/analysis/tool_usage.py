"""Per-tool effectiveness analysis.

Measures how each tool contributes to retrieval quality improvement,
including per-tool success rates, quality score deltas, and
tool-specific usage patterns across question types.

Usage:
    from experiments.analysis.tool_usage import ToolUsageAnalyzer

    analyzer = ToolUsageAnalyzer.from_results("data/results/")
    analyzer.print_tool_effectiveness()
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger


class ToolUsageAnalyzer:
    """Analyze per-tool effectiveness from ablation experiment results."""

    def __init__(self, variant_results: dict[str, list[dict]]) -> None:
        """Initialize with results keyed by variant name."""
        self.variant_results = variant_results

    @classmethod
    def from_results_dir(cls, results_dir: str | Path) -> ToolUsageAnalyzer:
        """Load results from multiple result directories."""
        results_dir = Path(results_dir)
        variant_results: dict[str, list[dict]] = {}

        for subdir in sorted(results_dir.iterdir()):
            if not subdir.is_dir():
                continue
            results = []
            for jsonl_path in subdir.glob("*.jsonl"):
                with open(jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        results.append(json.loads(line.strip()))
            if results:
                variant_results[subdir.name] = results

        return cls(variant_results)

    def compute_metrics(self) -> dict[str, dict[str, float]]:
        """Compute per-variant metrics for comparison."""
        from agentic_rag.evaluation.metrics import evaluate_batch

        metrics = {}
        for name, results in self.variant_results.items():
            valid = [r for r in results if "error" not in r]
            if not valid:
                continue
            preds = [r.get("prediction", "") for r in valid]
            refs = [r.get("reference", "") for r in valid]
            batch_metrics = evaluate_batch(preds, refs, compute_bert_score=False)
            batch_metrics["avg_latency"] = float(
                np.mean([r.get("latency_seconds", 0) for r in valid])
            )
            batch_metrics["avg_llm_calls"] = float(np.mean([r.get("llm_calls", 0) for r in valid]))
            metrics[name] = batch_metrics
        return metrics

    def compute_tool_impact(self, baseline_name: str = "full") -> dict[str, dict[str, float]]:
        """Compute quality delta when each tool is removed.

        Args:
            baseline_name: Name of the full-system variant for comparison.

        Returns:
            Dict of {variant_name: {metric: delta}} relative to baseline.
        """
        all_metrics = self.compute_metrics()
        baseline = all_metrics.get(baseline_name)
        if not baseline:
            logger.warning(f"Baseline '{baseline_name}' not found in results")
            return {}

        impact = {}
        for name, metrics in all_metrics.items():
            if name == baseline_name:
                continue
            impact[name] = {
                metric: metrics[metric] - baseline[metric]
                for metric in ("exact_match", "f1", "rouge_l")
                if metric in metrics and metric in baseline
            }
        return impact

    def print_tool_effectiveness(self) -> None:
        """Print tool effectiveness comparison table."""
        metrics = self.compute_metrics()
        if not metrics:
            logger.warning("No metrics to display")
            return

        logger.info("\n=== Tool Effectiveness (Per-Variant Metrics) ===")
        for name, m in sorted(metrics.items()):
            logger.info(
                f"  {name:30s}: EM={m.get('exact_match', 0):.3f} "
                f"F1={m.get('f1', 0):.3f} "
                f"ROUGE-L={m.get('rouge_l', 0):.3f} "
                f"Latency={m.get('avg_latency', 0):.1f}s "
                f"LLM={m.get('avg_llm_calls', 0):.1f}"
            )

    def group_by_question_type(self) -> dict[str, dict[str, list[dict]]]:
        """Group results by question type (from action_history or metadata)."""
        grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for name, results in self.variant_results.items():
            for r in results:
                qtype = r.get("agent_type", "standard") or "standard"
                grouped[qtype][name].append(r)
        return dict(grouped)
