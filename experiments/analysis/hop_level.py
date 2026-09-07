"""F1 by question hop count, per pipeline, across the three model trees.

Backs the hop axis of the decision boundary (CHANGES D4): for every
(model, dataset, hop_count) cell it reports each pipeline's mean F1 and,
for the main comparison, the paired difference of the agentic pipeline
against a baseline with a paired-bootstrap p-value and 95% CI. Questions
are paired by ``id``, so the difference is computed on the same questions.

``hop_count`` is taken from the ``question_difficulty`` field stamped on
every result row by ``experiments.common``; HotpotQA is all 2-hop and
FinanceBench all 1-hop, so only 2Wiki (2/4) and MuSiQue (2/3/4) split.

Usage:
    uv run python experiments/analysis/hop_level.py --csv out.csv
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from experiments.analysis.runs import DATASETS, best_f1, load_default_trees, load_runs

AGENTIC = "agentic_(react)"


def _paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    """Two-sided paired bootstrap of mean(a - b): (p, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    observed = float(diff.mean())
    # p-value: share of bootstrap means on the far side of zero, two-sided
    p = 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    return min(p, 1.0), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), observed


class HopLevelAnalyzer:
    """Hop-stratified F1 and paired deltas per model and dataset."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records

    @classmethod
    def from_default_trees(cls, prefix: str = "rq1") -> HopLevelAnalyzer:
        return cls(load_default_trees(prefix=prefix))

    @classmethod
    def from_results_dir(
        cls, results_dir: str | Path, prefix: str = "rq1", model: str | None = None
    ) -> HopLevelAnalyzer:
        return cls(load_runs(results_dir, prefix=prefix, model=model))

    # ------------------------------------------------------------------
    def _f1_by_id(self, rec: dict) -> dict[str, tuple[int, float]]:
        """id -> (hop_count, best F1) for one run."""
        out = {}
        for r in rec["rows"]:
            hop = r.get("question_difficulty", {}).get("hop_count")
            if hop is None:
                continue
            out[r["id"]] = (int(hop), best_f1(str(r.get("prediction", "")), r))
        return out

    def per_cell(self) -> list[dict]:
        """Mean F1 per (model, dataset, hop_count, pipeline)."""
        out = []
        for rec in self.records:
            by_hop: dict[int, list[float]] = defaultdict(list)
            for hop, f1 in self._f1_by_id(rec).values():
                by_hop[hop].append(f1)
            for hop in sorted(by_hop):
                scores = np.array(by_hop[hop])
                out.append(
                    {
                        "model": rec["model"],
                        "dataset": rec["dataset"],
                        "hop_count": hop,
                        "pipeline": rec["pipeline"],
                        "n": len(scores),
                        "f1_mean": round(float(scores.mean()), 4),
                        "f1_std": round(float(scores.std()), 4),
                    }
                )
        return out

    def paired_deltas(self, baseline: str = "loop_refinement", n_boot: int = 10000) -> list[dict]:
        """AGENTIC minus ``baseline`` per (model, dataset, hop_count), paired by id."""
        runs = {(r["model"], r["dataset"], r["pipeline"]): r for r in self.records}
        out = []
        for (model, dataset, pipeline), rec in sorted(runs.items()):
            if pipeline != AGENTIC:
                continue
            base = runs.get((model, dataset, baseline))
            if base is None:
                continue
            a_map, b_map = self._f1_by_id(rec), self._f1_by_id(base)
            by_hop: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for qid, (hop, fa) in a_map.items():
                if qid in b_map:
                    by_hop[hop].append((fa, b_map[qid][1]))
            for hop in sorted(by_hop):
                a = np.array([x for x, _ in by_hop[hop]])
                b = np.array([y for _, y in by_hop[hop]])
                p, lo, hi, delta = _paired_bootstrap(a, b, n_boot=n_boot)
                out.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "hop_count": hop,
                        "baseline": baseline,
                        "n": len(a),
                        "f1_agentic": round(float(a.mean()), 4),
                        "f1_baseline": round(float(b.mean()), 4),
                        "delta": round(delta, 4),
                        "ci_lower": round(lo, 4),
                        "ci_upper": round(hi, 4),
                        "p_boot": round(p, 4),
                    }
                )
        return out

    # ------------------------------------------------------------------
    def print_report(self, baseline: str = "loop_refinement") -> None:
        cells = self.per_cell()
        for model in sorted({c["model"] for c in cells}):
            logger.info(f"\n=== Hop-level F1: {model} ===")
            logger.info(f"{'dataset':<17}{'hop':>4}{'pipeline':<22}{'n':>5}{'F1':>8}")
            for dataset in DATASETS:
                for c in [x for x in cells if x["model"] == model and x["dataset"] == dataset]:
                    logger.info(
                        f"{c['dataset']:<17}{c['hop_count']:>4}  {c['pipeline']:<20}"
                        f"{c['n']:>5}{c['f1_mean']:>8.3f}"
                    )
        logger.info(f"\n=== {AGENTIC} - {baseline}, paired by question ===")
        header = f"{'model':<18}{'dataset':<17}{'hop':>4}{'n':>5}"
        logger.info(header + f"{'Δ F1':>8}{'95% CI':>18}{'p':>8}")
        for d in self.paired_deltas(baseline=baseline):
            logger.info(
                f"{d['model']:<18}{d['dataset']:<17}{d['hop_count']:>4}{d['n']:>5}"
                f"{d['delta']:>+8.3f}  [{d['ci_lower']:+.3f}, {d['ci_upper']:+.3f}]{d['p_boot']:>8.3f}"
            )

    def to_csv(self, out_path: str | Path, baseline: str = "loop_refinement") -> Path:
        """Write per-cell F1 and the paired deltas as two CSVs (``<stem>_deltas.csv``)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cells = self.per_cell()
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
            w.writeheader()
            w.writerows(cells)
        deltas = self.paired_deltas(baseline=baseline)
        delta_path = out_path.with_name(out_path.stem + "_deltas.csv")
        with open(delta_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
            w.writeheader()
            w.writerows(deltas)
        logger.info(f"Hop-level tables written to {out_path} and {delta_path}")
        return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hop-level F1 analysis across model trees")
    parser.add_argument("--results-dir", default=None, help="One tree; default is all three")
    parser.add_argument("--model", default=None, help="Model label for a flat-layout tree")
    parser.add_argument("--baseline", default="loop_refinement")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    if args.results_dir:
        analyzer = HopLevelAnalyzer.from_results_dir(args.results_dir, model=args.model)
    else:
        analyzer = HopLevelAnalyzer.from_default_trees()
    analyzer.print_report(baseline=args.baseline)
    if args.csv:
        analyzer.to_csv(args.csv, baseline=args.baseline)
