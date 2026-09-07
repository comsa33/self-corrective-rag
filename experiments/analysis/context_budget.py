"""What each pipeline was actually allowed to consume, per run.

The context-budget threat (CHANGES B9, C1) is that the arms of a
comparison see different amounts of the corpus. This module reports, for
every (model, dataset, pipeline), the passages placed in front of the
generator (``passages_used``), the passages retrieved in total, the
number of LLM calls, and wall-clock latency -- the quantities a matched-
budget protocol has to equalise. Latency from the March trees is cache-
contaminated (see ``latency_analysis``); it is reported here only so the
contamination is visible, not as a measurement.

Usage:
    uv run python experiments/analysis/context_budget.py --csv out.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from loguru import logger

from experiments.analysis.runs import DATASETS, load_default_trees, load_runs


class ContextBudgetAnalyzer:
    """Passages, LLM calls and latency per pipeline run."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records

    @classmethod
    def from_default_trees(cls, prefix: str = "rq1") -> ContextBudgetAnalyzer:
        return cls(load_default_trees(prefix=prefix))

    @classmethod
    def from_results_dir(
        cls, results_dir: str | Path, prefix: str = "rq1", model: str | None = None
    ) -> ContextBudgetAnalyzer:
        return cls(load_runs(results_dir, prefix=prefix, model=model))

    def compute(self) -> list[dict]:
        out = []
        for rec in self.records:
            rows = rec["rows"]
            if not rows:
                continue
            used = np.array([r.get("passages_used", 0) for r in rows], dtype=float)
            retrieved = np.array([r.get("total_passages_retrieved", 0) for r in rows], dtype=float)
            calls = np.array([r.get("llm_calls", 0) for r in rows], dtype=float)
            latency = np.array([r.get("latency_seconds", 0.0) for r in rows], dtype=float)
            out.append(
                {
                    "model": rec["model"],
                    "dataset": rec["dataset"],
                    "pipeline": rec["pipeline"],
                    "n": len(rows),
                    "passages_used_mean": round(float(used.mean()), 2),
                    "passages_used_min": int(used.min()),
                    "passages_used_max": int(used.max()),
                    "passages_retrieved_mean": round(float(retrieved.mean()), 2),
                    "llm_calls_mean": round(float(calls.mean()), 2),
                    "llm_calls_median": float(np.median(calls)),
                    "llm_calls_max": int(calls.max()),
                    "latency_mean_s": round(float(latency.mean()), 2),
                    "latency_sub1s_share": round(float(np.mean(latency < 1.0)), 3),
                }
            )
        return out

    def print_table(self) -> None:
        rows = self.compute()
        for model in sorted({r["model"] for r in rows}):
            logger.info(f"\n=== Context budget: {model} ===")
            logger.info(
                f"{'dataset':<17}{'pipeline':<20}{'used':>6}{'(min-max)':>10}{'retr':>7}"
                f"{'calls':>7}{'max':>5}{'lat(s)':>8}{'<1s':>6}"
            )
            for dataset in DATASETS:
                for r in [x for x in rows if x["model"] == model and x["dataset"] == dataset]:
                    logger.info(
                        f"{r['dataset']:<17}{r['pipeline']:<20}{r['passages_used_mean']:>6.1f}"
                        f"  {r['passages_used_min']:>3}-{r['passages_used_max']:<4}"
                        f"{r['passages_retrieved_mean']:>7.1f}{r['llm_calls_mean']:>7.1f}"
                        f"{r['llm_calls_max']:>5}{r['latency_mean_s']:>8.1f}"
                        f"{100 * r['latency_sub1s_share']:>5.0f}%"
                    )

    def to_csv(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.compute()
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"Context budget table written to {out_path}")
        return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Context budget per pipeline run")
    parser.add_argument("--results-dir", default=None, help="One tree; default is all three")
    parser.add_argument("--model", default=None, help="Model label for a flat-layout tree")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if args.results_dir:
        analyzer = ContextBudgetAnalyzer.from_results_dir(args.results_dir, model=args.model)
    else:
        analyzer = ContextBudgetAnalyzer.from_default_trees()
    analyzer.print_table()
    if args.csv:
        analyzer.to_csv(args.csv)
