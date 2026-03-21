"""Mediation effect analysis for RQ2 extension.

Implements Baron & Kenny (1986) causal mediation analysis and bootstrap
confidence intervals (Hayes PROCESS style) to test whether tool diversity
and quality score improvement mediate the relationship between pipeline
type and answer quality.

Path model:
    Pipeline Type (IV) → Tool Diversity / Score Improvement (M) → EM/F1 (DV)

Usage:
    from experiments.analysis.mediation import MediationAnalyzer

    analyzer = MediationAnalyzer.from_results_dir("data/results/rq1_hotpotqa_n50/")
    analyzer.print_summary()
    analyzer.run_baron_kenny()
    analyzer.run_bootstrap_mediation(n_boot=5000)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class MediationAnalyzer:
    """Mediation effect analysis across pipeline variants."""

    def __init__(self, results: list[dict]) -> None:
        self.results = [r for r in results if "error" not in r]
        self.df = self._build_dataframe()

    @classmethod
    def from_results_dir(cls, results_dir: str | Path) -> MediationAnalyzer:
        """Load all JSONL result files from a directory."""
        results_dir = Path(results_dir)
        results = []
        for jsonl_path in sorted(results_dir.glob("*.jsonl")):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
        return cls(results)

    def _build_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame with mediation variables."""
        from agentic_rag.evaluation.metrics import exact_match, token_f1

        rows = []
        for r in self.results:
            pred = r.get("prediction", "")
            ref = r.get("reference", "")
            action_history = r.get("action_history", [])
            eval_scores = r.get("evaluation_scores", [])
            tool_score_trace = r.get("tool_score_trace", [])
            difficulty = r.get("question_difficulty", {})
            pipeline = r.get("pipeline", "unknown")

            # --- Independent Variable: Pipeline type ---
            # Binary encoding (agentic=1, others=0) avoids equal-interval
            # assumption problems with ordinal encoding in OLS regression.
            pipeline_code = _encode_pipeline(pipeline)
            is_agentic = 1 if "agentic" in pipeline.lower() else 0

            # --- Mediator 1: Tool diversity (Shannon entropy) ---
            tool_diversity = _shannon_entropy(action_history)

            # --- Mediator 2: Score improvement (total delta across iterations) ---
            score_improvement = _compute_score_improvement(eval_scores)

            # --- Mediator 3: Iteration depth ---
            iteration_depth = len(action_history)

            # --- Dependent Variables ---
            em = exact_match(pred, ref)
            f1 = token_f1(pred, ref)

            # --- Moderating Variables (question difficulty) ---
            hop_count = difficulty.get("hop_count", 1)
            entity_count = difficulty.get("entity_count", 0)
            question_type = difficulty.get("question_type", "factoid")

            rows.append(
                {
                    "id": r.get("id", ""),
                    "pipeline": pipeline,
                    "pipeline_code": pipeline_code,
                    "is_agentic": is_agentic,
                    "tool_diversity": tool_diversity,
                    "score_improvement": score_improvement,
                    "iteration_depth": iteration_depth,
                    "em": em,
                    "f1": f1,
                    "hop_count": hop_count,
                    "entity_count": entity_count,
                    "question_type": question_type,
                    "latency": r.get("latency_seconds", 0),
                    "llm_calls": r.get("llm_calls", 0),
                    "tool_score_trace": tool_score_trace,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Baron & Kenny (1986) 4-Step Mediation Test
    # ------------------------------------------------------------------
    def run_baron_kenny(
        self,
        iv: str = "is_agentic",
        mediators: list[str] | None = None,
        dv: str = "f1",
    ) -> dict[str, dict]:
        """Run Baron & Kenny 4-step mediation analysis.

        Steps:
            1. IV → DV (total effect, path c)
            2. IV → M (path a)
            3. M → DV controlling for IV (path b)
            4. IV → DV controlling for M (direct effect, path c')

        Mediation exists if: a is significant, b is significant,
        and c' < c (partial) or c' ≈ 0 (full mediation).
        """
        import statsmodels.api as sm

        if mediators is None:
            mediators = ["tool_diversity", "score_improvement", "iteration_depth"]

        results = {}
        df = self.df.dropna(subset=[iv, dv])

        if len(df) < 10:
            logger.warning(f"Insufficient data for mediation analysis (n={len(df)})")
            return results

        # Step 1: Total effect (c path): IV → DV
        X = sm.add_constant(df[[iv]])
        model_c = sm.OLS(df[dv], X).fit()
        c_coef = model_c.params[iv]
        c_pval = model_c.pvalues[iv]

        for mediator in mediators:
            if mediator not in df.columns:
                continue

            df_m = df.dropna(subset=[mediator])
            if len(df_m) < 10:
                continue

            # Step 2: IV → M (a path)
            X_a = sm.add_constant(df_m[[iv]])
            model_a = sm.OLS(df_m[mediator], X_a).fit()
            a_coef = model_a.params[iv]
            a_pval = model_a.pvalues[iv]

            # Step 3 & 4: IV + M → DV (b and c' paths)
            X_bc = sm.add_constant(df_m[[iv, mediator]])
            model_bc = sm.OLS(df_m[dv], X_bc).fit()
            b_coef = model_bc.params[mediator]
            b_pval = model_bc.pvalues[mediator]
            c_prime = model_bc.params[iv]
            c_prime_pval = model_bc.pvalues[iv]

            # Indirect effect = a * b
            indirect = a_coef * b_coef

            # Proportion mediated
            proportion = indirect / c_coef if abs(c_coef) > 1e-10 else 0.0

            # Sobel test
            sobel_z, sobel_p = _sobel_test(
                a_coef,
                b_coef,
                model_a.bse.get(iv, 0),
                model_bc.bse.get(mediator, 0),
            )

            results[mediator] = {
                "a_coef": a_coef,
                "a_pval": a_pval,
                "b_coef": b_coef,
                "b_pval": b_pval,
                "c_coef": c_coef,
                "c_pval": c_pval,
                "c_prime": c_prime,
                "c_prime_pval": c_prime_pval,
                "indirect_effect": indirect,
                "proportion_mediated": proportion,
                "sobel_z": sobel_z,
                "sobel_p": sobel_p,
                "n": len(df_m),
                "mediation_type": _classify_mediation(c_coef, c_prime, a_pval, b_pval),
            }

        return results

    # ------------------------------------------------------------------
    # Bootstrap Mediation (Hayes PROCESS style)
    # ------------------------------------------------------------------
    def run_bootstrap_mediation(
        self,
        iv: str = "is_agentic",
        mediators: list[str] | None = None,
        dv: str = "f1",
        n_boot: int = 5000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict[str, dict]:
        """Run bootstrap mediation analysis (Hayes PROCESS Model 4).

        Bootstrap the indirect effect (a*b) to construct confidence intervals
        without assuming normality of the indirect effect distribution.
        """
        import statsmodels.api as sm

        if mediators is None:
            mediators = ["tool_diversity", "score_improvement", "iteration_depth"]

        rng = np.random.default_rng(seed)
        results = {}
        df = self.df.dropna(subset=[iv, dv])

        for mediator in mediators:
            df_m = df.dropna(subset=[mediator])
            n = len(df_m)
            if n < 10:
                continue

            # Point estimates
            X_a = sm.add_constant(df_m[[iv]])
            a_coef = sm.OLS(df_m[mediator], X_a).fit().params[iv]

            X_bc = sm.add_constant(df_m[[iv, mediator]])
            model_bc = sm.OLS(df_m[dv], X_bc).fit()
            b_coef = model_bc.params[mediator]
            indirect_point = a_coef * b_coef

            # Bootstrap
            boot_indirect = np.zeros(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot_df = df_m.iloc[idx]

                try:
                    X_a_b = sm.add_constant(boot_df[[iv]])
                    a_b = sm.OLS(boot_df[mediator], X_a_b).fit().params[iv]

                    X_bc_b = sm.add_constant(boot_df[[iv, mediator]])
                    b_b = sm.OLS(boot_df[dv], X_bc_b).fit().params[mediator]

                    boot_indirect[b] = a_b * b_b
                except Exception:
                    boot_indirect[b] = np.nan

            boot_indirect = boot_indirect[~np.isnan(boot_indirect)]

            alpha = 1 - ci
            ci_lower = float(np.percentile(boot_indirect, 100 * alpha / 2))
            ci_upper = float(np.percentile(boot_indirect, 100 * (1 - alpha / 2)))

            # If CI doesn't include 0, the indirect effect is significant
            significant = not (ci_lower <= 0 <= ci_upper)

            results[mediator] = {
                "indirect_effect": indirect_point,
                "boot_ci_lower": ci_lower,
                "boot_ci_upper": ci_upper,
                "boot_mean": float(np.mean(boot_indirect)),
                "boot_se": float(np.std(boot_indirect)),
                "significant": significant,
                "n_boot": len(boot_indirect),
                "n": n,
                "ci_level": ci,
            }

        return results

    # ------------------------------------------------------------------
    # Summary & Reporting
    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print descriptive statistics for mediation variables by pipeline."""
        logger.info("\n=== Mediation Analysis: Descriptive Statistics ===")

        for pipeline in sorted(self.df["pipeline"].unique()):
            sub = self.df[self.df["pipeline"] == pipeline]
            logger.info(
                f"\n  [{pipeline}] n={len(sub)}"
                f"\n    tool_diversity:   mean={sub['tool_diversity'].mean():.3f}, "
                f"std={sub['tool_diversity'].std():.3f}"
                f"\n    score_improvement: mean={sub['score_improvement'].mean():.1f}, "
                f"std={sub['score_improvement'].std():.1f}"
                f"\n    iteration_depth:   mean={sub['iteration_depth'].mean():.1f}, "
                f"std={sub['iteration_depth'].std():.1f}"
                f"\n    em:                mean={sub['em'].mean():.3f}"
                f"\n    f1:                mean={sub['f1'].mean():.3f}"
            )

    def print_baron_kenny(self, bk_results: dict[str, dict]) -> None:
        """Pretty-print Baron & Kenny results."""
        logger.info("\n=== Baron & Kenny Mediation Analysis ===")
        for mediator, r in bk_results.items():
            sig_a = "*" if r["a_pval"] < 0.05 else ""
            sig_b = "*" if r["b_pval"] < 0.05 else ""
            sig_c = "*" if r["c_pval"] < 0.05 else ""
            logger.info(
                f"\n  Mediator: {mediator} (n={r['n']})"
                f"\n    Path a (IV→M):  β={r['a_coef']:.4f}, p={r['a_pval']:.4f}{sig_a}"
                f"\n    Path b (M→DV):  β={r['b_coef']:.4f}, p={r['b_pval']:.4f}{sig_b}"
                f"\n    Path c (total): β={r['c_coef']:.4f}, p={r['c_pval']:.4f}{sig_c}"
                f"\n    Path c' (direct): β={r['c_prime']:.4f}, p={r['c_prime_pval']:.4f}"
                f"\n    Indirect effect: {r['indirect_effect']:.4f}"
                f"\n    Proportion mediated: {r['proportion_mediated']:.1%}"
                f"\n    Sobel test: z={r['sobel_z']:.3f}, p={r['sobel_p']:.4f}"
                f"\n    → {r['mediation_type']}"
            )

    def print_bootstrap(self, boot_results: dict[str, dict]) -> None:
        """Pretty-print bootstrap mediation results."""
        logger.info("\n=== Bootstrap Mediation Analysis ===")
        for mediator, r in boot_results.items():
            sig = "SIGNIFICANT" if r["significant"] else "not significant"
            logger.info(
                f"\n  Mediator: {mediator} (n={r['n']}, boots={r['n_boot']})"
                f"\n    Indirect effect: {r['indirect_effect']:.4f}"
                f"\n    {r['ci_level']:.0%} CI: [{r['boot_ci_lower']:.4f}, {r['boot_ci_upper']:.4f}]"
                f"\n    Boot SE: {r['boot_se']:.4f}"
                f"\n    → {sig}"
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the analysis DataFrame for external use."""
        return self.df.copy()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _encode_pipeline(pipeline_name: str) -> int:
    """Encode pipeline type as ordinal (increasing sophistication)."""
    encoding = {
        "naive_rag": 0,
        "crag_replica": 1,
        "single-pass": 2,
        "loop_refinement": 3,
        "agentic_(react)": 4,
    }
    return encoding.get(pipeline_name, 0)


def _shannon_entropy(action_history: list[str]) -> float:
    """Compute Shannon entropy of tool usage distribution.

    Higher entropy = more diverse tool usage.
    Returns 0 for empty or single-tool histories.
    """
    if not action_history:
        return 0.0
    counts = Counter(action_history)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_score_improvement(evaluation_scores: list[dict]) -> float:
    """Compute total quality score improvement across iterations.

    Returns the delta between first and last evaluation total score.
    Returns 0 if fewer than 2 evaluations exist (no iteration = no improvement).
    """
    totals = [s.get("total", 0) for s in evaluation_scores if "total" in s]
    if len(totals) < 2:
        return 0.0
    return float(totals[-1] - totals[0])


def _sobel_test(a: float, b: float, se_a: float, se_b: float) -> tuple[float, float]:
    """Sobel test for significance of the indirect effect (a*b).

    Returns (z_statistic, p_value).
    """
    from scipy import stats

    se_indirect = math.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    if se_indirect < 1e-10:
        return 0.0, 1.0
    z = (a * b) / se_indirect
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _classify_mediation(c: float, c_prime: float, a_pval: float, b_pval: float) -> str:
    """Classify mediation type based on Baron & Kenny criteria."""
    a_sig = a_pval < 0.05
    b_sig = b_pval < 0.05

    if not a_sig or not b_sig:
        return "No mediation (a or b path not significant)"

    if abs(c) < 1e-10:
        return "No mediation (no total effect)"

    reduction = 1 - (c_prime / c) if abs(c) > 1e-10 else 0
    if abs(c_prime) < 1e-10 or reduction > 0.8:
        return "Full mediation"
    elif reduction > 0.2:
        return "Partial mediation"
    else:
        return "No mediation (c' not reduced)"
