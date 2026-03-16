"""Quality score progression analysis across refinement iterations.

Tracks how quality scores (4D: relevance, coverage, specificity, sufficiency)
change across iterations for both for-loop and ReAct agentic refinement approaches.

Usage:
    from experiments.analysis.score_progression import ScoreProgressionAnalyzer

    analyzer = ScoreProgressionAnalyzer.from_results("data/results/rq1_*/")
    analyzer.print_progression_summary()
    analyzer.plot_score_curves("figures/score_progression.pdf")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger


class ScoreProgressionAnalyzer:
    """Analyze quality score progression across iterations."""

    def __init__(self, results: list[dict]) -> None:
        self.results = [r for r in results if "error" not in r]

    @classmethod
    def from_results(cls, results_dir: str | Path) -> ScoreProgressionAnalyzer:
        """Load results from a directory of JSONL files."""
        results_dir = Path(results_dir)
        results = []
        for jsonl_path in results_dir.glob("*.jsonl"):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
        return cls(results)

    def extract_score_sequences(self) -> list[list[dict]]:
        """Extract per-question evaluation score sequences.

        Returns:
            List of score sequences, each being a list of score dicts
            with keys: relevance, coverage, specificity, sufficiency, total.
        """
        sequences = []
        for r in self.results:
            scores = r.get("evaluation_scores", [])
            if scores and isinstance(scores, list) and len(scores) > 0:
                sequences.append(scores)
        return sequences

    def compute_improvement_stats(self) -> dict[str, float]:
        """Compute statistics about score improvement across iterations."""
        sequences = self.extract_score_sequences()
        if not sequences:
            return {}

        improvements = []
        for seq in sequences:
            totals = [s.get("total", 0) for s in seq if "total" in s]
            if len(totals) >= 2:
                improvements.append(totals[-1] - totals[0])

        if not improvements:
            return {}

        return {
            "mean_improvement": float(np.mean(improvements)),
            "median_improvement": float(np.median(improvements)),
            "std_improvement": float(np.std(improvements)),
            "pct_improved": float(np.mean([1 if imp > 0 else 0 for imp in improvements])),
            "max_improvement": float(max(improvements)),
            "min_improvement": float(min(improvements)),
            "n_multi_iteration": len(improvements),
        }

    def per_dimension_progression(self) -> dict[str, list[float]]:
        """Compute average score per dimension at each iteration index.

        Returns:
            Dict of {dimension: [avg_score_at_iter_0, avg_score_at_iter_1, ...]}.
        """
        sequences = self.extract_score_sequences()
        dimensions = ["relevance", "coverage", "specificity", "sufficiency", "total"]
        max_len = max((len(seq) for seq in sequences), default=0)

        progression: dict[str, list[float]] = {d: [] for d in dimensions}
        for idx in range(max_len):
            for dim in dimensions:
                vals = [
                    seq[idx].get(dim, 0) for seq in sequences if idx < len(seq) and dim in seq[idx]
                ]
                progression[dim].append(float(np.mean(vals)) if vals else 0.0)

        return progression

    def print_progression_summary(self) -> None:
        """Print score progression summary."""
        stats = self.compute_improvement_stats()
        if not stats:
            logger.info("No multi-iteration results to analyze")
            return

        logger.info("\n=== Score Progression Summary ===")
        logger.info(f"  Multi-iteration items: {stats['n_multi_iteration']}")
        logger.info(f"  Mean improvement: {stats['mean_improvement']:.1f}")
        logger.info(f"  Median improvement: {stats['median_improvement']:.1f}")
        logger.info(f"  % improved: {stats['pct_improved'] * 100:.1f}%")
        logger.info(f"  Range: [{stats['min_improvement']:.1f}, {stats['max_improvement']:.1f}]")

        progression = self.per_dimension_progression()
        logger.info("\n--- Per-Dimension Average by Iteration ---")
        for dim, values in progression.items():
            if values:
                formatted = " → ".join(f"{v:.1f}" for v in values[:5])
                logger.info(f"  {dim:15s}: {formatted}")
