"""ReAct trajectory analysis — tool sequence patterns and agent behavior.

Analyzes ReAct agent trajectories from experiment results to understand
how the agent uses its tools, what sequences emerge, and how search
strategies evolve across refinement iterations.

The ReAct trajectory format is a dict with structured keys:
    thought_0, tool_name_0, tool_args_0, observation_0,
    thought_1, tool_name_1, tool_args_1, observation_1, ...

Usage:
    from experiments.analysis.trajectory import TrajectoryAnalyzer

    analyzer = TrajectoryAnalyzer.from_results("data/results/rq2_agentic_full_tools/")
    analyzer.print_summary()
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger


class TrajectoryAnalyzer:
    """Analyze ReAct agent tool-call trajectories."""

    def __init__(self, results: list[dict]) -> None:
        self.results = [r for r in results if "error" not in r]
        self._trajectories: list[list[str]] = []
        self._thoughts: list[list[str]] = []
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
        """Extract tool-call sequences from action_history.

        action_history is a list of tool names produced by
        agentic.parse_action_history() (excludes 'finish').
        """
        for r in self.results:
            history = r.get("action_history", [])
            if isinstance(history, list):
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
        return float(np.mean([len(t) for t in self._trajectories]))

    @property
    def tool_bigrams(self) -> Counter:
        """Count tool-call bigrams (sequential pairs) across trajectories."""
        bigrams: Counter = Counter()
        for traj in self._trajectories:
            for i in range(len(traj) - 1):
                bigrams[(traj[i], traj[i + 1])] += 1
        return bigrams

    @property
    def tool_trigrams(self) -> Counter:
        """Count tool-call trigrams across trajectories."""
        trigrams: Counter = Counter()
        for traj in self._trajectories:
            for i in range(len(traj) - 2):
                trigrams[(traj[i], traj[i + 1], traj[i + 2])] += 1
        return trigrams

    def first_tool_distribution(self) -> Counter:
        """Distribution of which tool is called first in each trajectory."""
        counts: Counter = Counter()
        for traj in self._trajectories:
            if traj:
                counts[traj[0]] += 1
        return counts

    def trajectory_length_distribution(self) -> dict[str, float]:
        """Statistics on trajectory lengths."""
        lengths = [len(t) for t in self._trajectories]
        if not lengths:
            return {}
        return {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "std": float(np.std(lengths)),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
        }

    def pattern_distribution(self) -> dict[str, float]:
        """Frequency of the three dominant behavioral patterns.

        Classifies each trajectory into one of the patterns reported in the
        paper, using the number of decomposition and evaluation calls:

          - ``decompose_search_evaluate``: decomposition followed by a single
            evaluation pass (canonical multi-hop strategy).
          - ``search_evaluate``: no decomposition, single evaluation pass
            (single-hop questions).
          - ``decompose_search_evaluate_refine``: two or more evaluation calls,
            i.e. the self-corrective loop actually re-entered refinement.
          - ``other``: no evaluation call recorded.

        Returns proportions in [0, 1] plus the mean evaluation-call count,
        which quantifies how often the refinement loop is exercised at all.
        """
        counts: Counter = Counter()
        eval_calls: list[int] = []

        for traj in self._trajectories:
            n_eval = sum(1 for t in traj if "evaluate" in t)
            n_decompose = sum(1 for t in traj if "decompose" in t)
            eval_calls.append(n_eval)

            if n_eval >= 2:
                counts["decompose_search_evaluate_refine"] += 1
            elif n_eval == 1 and n_decompose >= 1:
                counts["decompose_search_evaluate"] += 1
            elif n_eval == 1:
                counts["search_evaluate"] += 1
            else:
                counts["other"] += 1

        total = len(self._trajectories)
        if not total:
            return {}

        dist = {
            name: counts[name] / total
            for name in (
                "decompose_search_evaluate",
                "search_evaluate",
                "decompose_search_evaluate_refine",
                "other",
            )
        }
        dist["mean_evaluate_calls"] = float(np.mean(eval_calls))
        dist["n"] = total
        return dist

    def print_summary(self) -> None:
        """Print trajectory analysis summary."""
        logger.info(f"Total trajectories: {len(self._trajectories)}")
        logger.info(f"Avg trajectory length: {self.avg_trajectory_length:.1f}")

        patterns = self.pattern_distribution()
        if patterns:
            logger.info("\n--- Behavioral Pattern Distribution ---")
            for name in (
                "decompose_search_evaluate",
                "search_evaluate",
                "decompose_search_evaluate_refine",
                "other",
            ):
                logger.info(f"  {name}: {100 * patterns[name]:.1f}%")
            logger.info(f"  mean evaluate calls/question: {patterns['mean_evaluate_calls']:.2f}")

        dist = self.trajectory_length_distribution()
        if dist:
            logger.info(
                f"Length distribution: "
                f"median={dist['median']:.0f}, "
                f"min={dist['min']}, max={dist['max']}"
            )

        logger.info("\n--- Tool Call Frequency ---")
        for tool, count in self.tool_call_counts.most_common():
            logger.info(f"  {tool}: {count}")

        logger.info("\n--- First Tool Distribution ---")
        for tool, count in self.first_tool_distribution().most_common():
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
                    "first_tool": traj[0] if traj else None,
                    "has_decompose": "decompose_query" in traj,
                    "has_evaluate": "evaluate_passages" in traj,
                    "search_count": traj.count("search_passages"),
                    "llm_calls": r.get("llm_calls", 0),
                }
            )
        return pd.DataFrame(rows)
