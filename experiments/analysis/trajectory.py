"""RLM trajectory analysis — tool sequence patterns and agent behavior.

Analyzes RLM agent trajectories from experiment results to understand
how the agent uses its tools, what sequences emerge, and how search
strategies evolve across refinement iterations.

Usage:
    from experiments.analysis.trajectory import TrajectoryAnalyzer

    analyzer = TrajectoryAnalyzer.from_results("data/results/rq2_agentic_full_tools/")
    analyzer.print_summary()
    analyzer.plot_tool_sequences("figures/rq2_tool_sequences.pdf")
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger


class TrajectoryAnalyzer:
    """Analyze RLM agent tool-call trajectories."""

    def __init__(self, results: list[dict]) -> None:
        self.results = [r for r in results if "error" not in r]
        self._trajectories: list[list[str]] = []
        self._parse_trajectories()

    @classmethod
    def from_results(cls, results_dir: str | Path) -> TrajectoryAnalyzer:
        """Load results from a directory of JSONL files."""
        results_dir = Path(results_dir)
        results = []
        for jsonl_path in results_dir.glob("*.jsonl"):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
        return cls(results)

    def _parse_trajectories(self) -> None:
        """Extract tool-call sequences from action_history."""
        for r in self.results:
            history = r.get("action_history", [])
            if isinstance(history, list):
                # action_history contains tool call names or action strings
                self._trajectories.append(history)

    @property
    def tool_call_counts(self) -> Counter:
        """Count total occurrences of each tool across all trajectories."""
        counts: Counter = Counter()
        for traj in self._trajectories:
            counts.update(traj)
        return counts

    @property
    def avg_trajectory_length(self) -> float:
        """Average number of tool calls per question."""
        if not self._trajectories:
            return 0.0
        return np.mean([len(t) for t in self._trajectories])

    @property
    def tool_bigrams(self) -> Counter:
        """Count tool-call bigrams (sequential pairs) across trajectories."""
        bigrams: Counter = Counter()
        for traj in self._trajectories:
            for i in range(len(traj) - 1):
                bigrams[(traj[i], traj[i + 1])] += 1
        return bigrams

    def print_summary(self) -> None:
        """Print trajectory analysis summary."""
        logger.info(f"Total trajectories: {len(self._trajectories)}")
        logger.info(f"Avg trajectory length: {self.avg_trajectory_length:.1f}")

        logger.info("\n--- Tool Call Frequency ---")
        for tool, count in self.tool_call_counts.most_common():
            logger.info(f"  {tool}: {count}")

        logger.info("\n--- Top Tool Bigrams ---")
        for (t1, t2), count in self.tool_bigrams.most_common(10):
            logger.info(f"  {t1} → {t2}: {count}")

    def to_dataframe(self):
        """Convert trajectory data to a pandas DataFrame for further analysis."""
        import pandas as pd

        rows = []
        for i, (r, traj) in enumerate(zip(self.results, self._trajectories, strict=False)):
            rows.append(
                {
                    "id": r.get("id", str(i)),
                    "question": r.get("question", ""),
                    "trajectory_length": len(traj),
                    "tool_calls": traj,
                    "unique_tools": len(set(traj)),
                    "retry_count": r.get("retry_count", 0),
                    "llm_calls": r.get("llm_calls", 0),
                }
            )
        return pd.DataFrame(rows)
