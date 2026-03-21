"""Statistical significance testing for pipeline comparisons.

Provides paired significance tests, effect sizes, and confidence intervals
for comparing RAG pipeline variants on EM, F1, and ROUGE-L metrics.

Usage:
    from experiments.analysis.significance import SignificanceAnalyzer

    analyzer = SignificanceAnalyzer.from_results_dir("data/results/rq1_hotpotqa_n50/")
    analyzer.print_pairwise_tests(baseline="agentic_(react)")
    analyzer.print_confidence_intervals()
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger


class SignificanceAnalyzer:
    """Statistical significance analysis across pipeline variants."""

    def __init__(self, pipeline_scores: dict[str, dict[str, list[float]]]) -> None:
        """Initialize with per-pipeline, per-metric score lists.

        Args:
            pipeline_scores: {pipeline_name: {metric_name: [score_per_item]}}
        """
        self.scores = pipeline_scores

    @classmethod
    def from_results_dir(cls, results_dir: str | Path) -> SignificanceAnalyzer:
        """Load all JSONL result files and compute per-item scores."""
        from agentic_rag.evaluation.metrics import exact_match, token_f1

        results_dir = Path(results_dir)
        pipeline_scores: dict[str, dict[str, list[float]]] = {}

        for jsonl_path in sorted(results_dir.glob("*.jsonl")):
            items = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    items.append(json.loads(line.strip()))

            valid = [r for r in items if "error" not in r]
            if not valid:
                continue

            pipeline = valid[0].get("pipeline", jsonl_path.stem)
            em_scores = []
            f1_scores = []

            for r in valid:
                pred = r.get("prediction", "")
                ref = r.get("reference", "")
                em_scores.append(exact_match(pred, ref))
                f1_scores.append(token_f1(pred, ref))

            pipeline_scores[pipeline] = {
                "em": em_scores,
                "f1": f1_scores,
            }

        return cls(pipeline_scores)

    # ------------------------------------------------------------------
    # Confidence Intervals (Bootstrap)
    # ------------------------------------------------------------------
    def bootstrap_ci(
        self,
        metric: str = "f1",
        n_boot: int = 10000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict[str, dict]:
        """Compute bootstrap confidence intervals for each pipeline.

        Returns:
            {pipeline: {mean, ci_lower, ci_upper, std}}
        """
        rng = np.random.default_rng(seed)
        alpha = 1 - ci
        results = {}

        for pipeline, metrics in self.scores.items():
            scores = np.array(metrics.get(metric, []))
            if len(scores) == 0:
                continue

            boot_means = np.zeros(n_boot)
            for b in range(n_boot):
                sample = rng.choice(scores, size=len(scores), replace=True)
                boot_means[b] = np.mean(sample)

            results[pipeline] = {
                "mean": float(np.mean(scores)),
                "ci_lower": float(np.percentile(boot_means, 100 * alpha / 2)),
                "ci_upper": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
                "std": float(np.std(scores)),
                "n": len(scores),
            }

        return results

    # ------------------------------------------------------------------
    # Pairwise Significance Tests
    # ------------------------------------------------------------------
    def pairwise_tests(
        self,
        baseline: str = "agentic_(react)",
        metric: str = "f1",
        n_boot: int = 10000,
        seed: int = 42,
    ) -> dict[str, dict]:
        """Run pairwise significance tests: baseline vs each other pipeline.

        Uses:
        - Paired bootstrap test (non-parametric, no normality assumption)
        - Wilcoxon signed-rank test (non-parametric paired test)
        - McNemar's test (for binary EM metric)
        - Cohen's d effect size

        Returns:
            {pipeline: {delta, boot_p, wilcoxon_p, mcnemar_p, cohens_d, significant}}
        """
        from scipy import stats

        if baseline not in self.scores:
            logger.warning(f"Baseline '{baseline}' not found in results")
            return {}

        base_scores = np.array(self.scores[baseline].get(metric, []))
        base_em = np.array(self.scores[baseline].get("em", []))
        rng = np.random.default_rng(seed)
        results = {}

        n_comparisons = len(self.scores) - 1
        bonferroni_alpha = 0.05 / max(n_comparisons, 1)

        for pipeline, metrics in self.scores.items():
            if pipeline == baseline:
                continue

            comp_scores = np.array(metrics.get(metric, []))
            comp_em = np.array(metrics.get("em", []))
            n = min(len(base_scores), len(comp_scores))

            if n < 5:
                continue

            b = base_scores[:n]
            c = comp_scores[:n]
            diff = b - c

            # --- Cohen's d effect size ---
            pooled_std = np.sqrt((np.std(b, ddof=1) ** 2 + np.std(c, ddof=1) ** 2) / 2)
            cohens_d = float(np.mean(diff) / pooled_std) if pooled_std > 1e-10 else 0.0

            # --- Paired bootstrap test ---
            # H0: no difference. Count how often bootstrap delta <= 0
            observed_delta = float(np.mean(diff))
            boot_deltas = np.zeros(n_boot)
            for i in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot_deltas[i] = np.mean(b[idx]) - np.mean(c[idx])
            # Two-sided p-value: proportion of bootstrap samples on wrong side of 0
            if observed_delta >= 0:
                boot_p = float(np.mean(boot_deltas <= 0)) * 2
            else:
                boot_p = float(np.mean(boot_deltas >= 0)) * 2
            boot_p = min(boot_p, 1.0)

            # --- Wilcoxon signed-rank test ---
            try:
                _, wilcoxon_p = stats.wilcoxon(b, c, alternative="two-sided")
                wilcoxon_p = float(wilcoxon_p)
            except ValueError:
                # All differences are zero
                wilcoxon_p = 1.0

            # --- McNemar's test (for EM, binary) ---
            mcnemar_p = None
            if len(base_em) >= n and len(comp_em) >= n:
                be = base_em[:n]
                ce = comp_em[:n]
                # b=1,c=0 (baseline correct, comp wrong) and b=0,c=1
                b10 = int(np.sum((be == 1) & (ce == 0)))
                b01 = int(np.sum((be == 0) & (ce == 1)))
                if b10 + b01 > 0:
                    # McNemar's chi-squared with continuity correction
                    chi2 = (abs(b10 - b01) - 1) ** 2 / (b10 + b01)
                    mcnemar_p = float(1 - stats.chi2.cdf(chi2, df=1))
                else:
                    mcnemar_p = 1.0

            results[pipeline] = {
                "delta": observed_delta,
                "boot_p": boot_p,
                "wilcoxon_p": wilcoxon_p,
                "mcnemar_p": mcnemar_p,
                "cohens_d": cohens_d,
                "bonferroni_alpha": bonferroni_alpha,
                "significant_boot": boot_p < bonferroni_alpha,
                "significant_wilcoxon": wilcoxon_p < bonferroni_alpha,
                "n": n,
            }

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def print_confidence_intervals(self, metric: str = "f1", ci: float = 0.95) -> None:
        """Print bootstrap confidence intervals for each pipeline."""
        cis = self.bootstrap_ci(metric=metric, ci=ci)
        logger.info(f"\n=== {ci:.0%} Confidence Intervals ({metric.upper()}) ===")
        for pipeline, r in sorted(cis.items(), key=lambda x: x[1]["mean"], reverse=True):
            logger.info(
                f"  {pipeline:25s}: {r['mean']:.3f} "
                f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}] "
                f"(n={r['n']})"
            )

    def print_pairwise_tests(
        self,
        baseline: str = "agentic_(react)",
        metric: str = "f1",
    ) -> None:
        """Print pairwise significance test results."""
        results = self.pairwise_tests(baseline=baseline, metric=metric)
        logger.info(f"\n=== Pairwise Tests: {baseline} vs others ({metric.upper()}) ===")
        logger.info(
            f"  Bonferroni-corrected alpha: {results[next(iter(results))]['bonferroni_alpha']:.4f}"
        )

        for pipeline, r in sorted(results.items(), key=lambda x: x[1]["delta"], reverse=True):
            sig_markers = []
            if r["significant_boot"]:
                sig_markers.append("boot*")
            if r["significant_wilcoxon"]:
                sig_markers.append("wilcox*")
            sig_str = ", ".join(sig_markers) if sig_markers else "n.s."

            mcnemar_str = f", McNemar p={r['mcnemar_p']:.4f}" if r["mcnemar_p"] is not None else ""

            logger.info(
                f"  vs {pipeline:25s}: "
                f"delta={r['delta']:+.3f}, "
                f"d={r['cohens_d']:.3f}, "
                f"boot_p={r['boot_p']:.4f}, "
                f"wilcox_p={r['wilcoxon_p']:.4f}"
                f"{mcnemar_str} "
                f"[{sig_str}]"
            )

    def print_full_report(self, baseline: str = "agentic_(react)") -> None:
        """Print comprehensive statistical report for paper."""
        for metric in ["em", "f1"]:
            self.print_confidence_intervals(metric=metric)
            self.print_pairwise_tests(baseline=baseline, metric=metric)
