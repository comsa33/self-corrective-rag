"""Per-question latency analysis with LLM response-cache detection.

DSPy caches LM responses on disk by default. At ``temperature = 0`` a rerun
over previously seen questions returns cached completions in milliseconds,
which silently deflates measured latency without changing any answer. Any
latency reported from a run that reused a warm cache is therefore invalid
even though its accuracy metrics remain correct.

This module reports latency both as measured and with suspected cache hits
excluded, and flags runs whose cache-hit share makes the raw mean unusable.

Usage:
    from experiments.analysis.latency_analysis import LatencyAnalyzer

    analyzer = LatencyAnalyzer.from_results_dir("data/results/paper")
    analyzer.print_latency_table()

Note:
    To measure latency cleanly, disable the cache before the run:
    ``dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)``
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from loguru import logger

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "financebench")

# A pipeline stage that issues at least one LLM call cannot plausibly finish
# faster than this; anything below is treated as a cache hit.
CACHE_HIT_THRESHOLD_SECONDS = 1.0

# Above this share of cache hits, the raw mean is reported but marked invalid.
CACHE_CONTAMINATION_LIMIT = 0.10


class LatencyAnalyzer:
    """Latency statistics per pipeline, with cache-contamination diagnostics."""

    def __init__(self, records: list[dict]) -> None:
        """Initialize with flat records: model, dataset, pipeline, latencies."""
        self.records = records

    @classmethod
    def from_results_dir(cls, results_dir: str | Path, prefix: str = "rq1") -> LatencyAnalyzer:
        """Load per-item latencies from every ``{prefix}_*.jsonl`` under subdirectories."""
        results_dir = Path(results_dir)
        records: list[dict] = []

        for subdir in sorted(results_dir.iterdir()):
            if not subdir.is_dir() or f"_{prefix}_" not in subdir.name:
                continue
            dataset = next((d for d in DATASETS if d in subdir.name), None)
            if dataset is None:
                continue
            model = "gpt-5-mini" if "gpt-5-mini" in subdir.name else "gemini-flash-lite"

            for jsonl_path in sorted(subdir.glob(f"{prefix}_*.jsonl")):
                if jsonl_path.name.endswith("_judged.jsonl"):
                    continue
                rows = [
                    json.loads(line)
                    for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                valid = [r for r in rows if "error" not in r]
                records.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "pipeline": jsonl_path.stem[len(prefix) + 1 :],
                        "run": subdir.name,
                        "latencies": [
                            r["latency_seconds"] for r in valid if "latency_seconds" in r
                        ],
                        "llm_calls": [r["llm_calls"] for r in valid if r.get("llm_calls")],
                        "n_missing_latency": sum(1 for r in valid if "latency_seconds" not in r),
                        "n_error": len(rows) - len(valid),
                    }
                )

        logger.info(f"Loaded {len(records)} pipeline runs from {results_dir}")
        return cls(records)

    def compute(self) -> list[dict]:
        """Compute raw and cache-corrected latency statistics per run."""
        out: list[dict] = []
        for rec in self.records:
            lat = rec["latencies"]
            if not lat:
                continue
            uncached = [x for x in lat if x >= CACHE_HIT_THRESHOLD_SECONDS]
            hit_rate = 1.0 - len(uncached) / len(lat)

            out.append(
                {
                    "model": rec["model"],
                    "dataset": rec["dataset"],
                    "pipeline": rec["pipeline"],
                    "n": len(lat),
                    "mean_raw": round(float(np.mean(lat)), 2),
                    "median_raw": round(float(np.median(lat)), 2),
                    "mean_uncached": round(float(np.mean(uncached)), 2) if uncached else None,
                    "cache_hits": len(lat) - len(uncached),
                    "cache_hit_rate": round(hit_rate, 3),
                    "cache_contaminated": hit_rate > CACHE_CONTAMINATION_LIMIT,
                    "mean_llm_calls": round(float(np.mean(rec["llm_calls"])), 1)
                    if rec["llm_calls"]
                    else None,
                    "n_error": rec["n_error"],
                    "n_missing_latency": rec["n_missing_latency"],
                }
            )
        return out

    def print_latency_table(self) -> None:
        """Print latency per model and dataset, flagging cache-contaminated runs."""
        rows = self.compute()
        for model in sorted({r["model"] for r in rows}):
            logger.info(f"\n=== Latency: {model} ===")
            logger.info(
                f"{'dataset':<17}{'pipeline':<20}{'mean_raw':>10}"
                f"{'mean_uncached':>15}{'cacheHit':>10}{'llm_calls':>11}  flag"
            )
            for dataset in DATASETS:
                for r in [x for x in rows if x["model"] == model and x["dataset"] == dataset]:
                    unc = f"{r['mean_uncached']:.1f}" if r["mean_uncached"] is not None else "-"
                    calls = f"{r['mean_llm_calls']:.1f}" if r["mean_llm_calls"] else "-"
                    flag = "CACHE-CONTAMINATED" if r["cache_contaminated"] else ""
                    logger.info(
                        f"{r['dataset']:<17}{r['pipeline']:<20}{r['mean_raw']:>10.1f}"
                        f"{unc:>15}{100 * r['cache_hit_rate']:>9.0f}%{calls:>11}  {flag}"
                    )

    def to_csv(self, out_path: str | Path) -> Path:
        """Write latency statistics to CSV."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.compute()
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Latency statistics written to {out_path}")
        return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Latency analysis with cache detection")
    parser.add_argument("--results-dir", default="data/results/paper")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    analyzer = LatencyAnalyzer.from_results_dir(args.results_dir)
    analyzer.print_latency_table()
    if args.csv:
        analyzer.to_csv(args.csv)
