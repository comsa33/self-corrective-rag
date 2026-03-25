"""Run Bootstrap CI and pairwise significance tests on all results.

This is fast (no LLM calls) — runs on existing EM/F1 scores.
Optionally includes LLM-as-Judge scores if _judged.jsonl files exist.

Usage:
    # All RQ1 results
    uv run python scripts/run_bootstrap_ci.py --pattern rq1

    # Specific directory
    uv run python scripts/run_bootstrap_ci.py data/results/20260324_171259_rq1_hotpotqa_n200_gemini-3.1-flash-lite/

    # All results
    uv run python scripts/run_bootstrap_ci.py --all

    # Save to file
    uv run python scripts/run_bootstrap_ci.py --all --output data/results/bootstrap_ci_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from experiments.analysis.significance import SignificanceAnalyzer


def analyze_directory(results_dir: Path) -> dict:
    """Run full statistical analysis on a results directory."""
    analyzer = SignificanceAnalyzer.from_results_dir(results_dir)

    if not analyzer.scores:
        return {"dir": results_dir.name, "error": "No valid results found"}

    report = {
        "dir": results_dir.name,
        "pipelines": {},
        "pairwise_f1": {},
        "pairwise_em": {},
    }

    # Bootstrap CI for each metric
    for metric in ["f1", "em"]:
        cis = analyzer.bootstrap_ci(metric=metric)
        for pipeline, ci in cis.items():
            if pipeline not in report["pipelines"]:
                report["pipelines"][pipeline] = {}
            report["pipelines"][pipeline][f"{metric}_mean"] = ci["mean"]
            report["pipelines"][pipeline][f"{metric}_ci_lower"] = ci["ci_lower"]
            report["pipelines"][pipeline][f"{metric}_ci_upper"] = ci["ci_upper"]
            report["pipelines"][pipeline][f"{metric}_std"] = ci["std"]
            report["pipelines"][pipeline]["n"] = ci["n"]

    # Pairwise tests (Agentic vs others)
    baseline = "agentic_(react)"
    if baseline not in analyzer.scores:
        # Try to find the best match
        for key in analyzer.scores:
            if "agentic" in key.lower():
                baseline = key
                break

    if baseline in analyzer.scores:
        for metric in ["f1", "em"]:
            pairwise = analyzer.pairwise_tests(baseline=baseline, metric=metric)
            report[f"pairwise_{metric}"] = {
                k: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in pairwise.items()
            }

    return report


def print_report(report: dict) -> None:
    """Pretty-print a single directory report."""
    print(f"\n{'=' * 70}")
    print(f"  {report['dir']}")
    print(f"{'=' * 70}")

    if "error" in report:
        print(f"  ERROR: {report['error']}")
        return

    # CI table
    print(
        f"\n  {'Pipeline':<30s} {'F1 Mean':>8s} {'95% CI':>18s} {'EM Mean':>8s} {'95% CI':>18s} {'n':>5s}"
    )
    print(f"  {'-' * 30} {'-' * 8} {'-' * 18} {'-' * 8} {'-' * 18} {'-' * 5}")
    for pipeline, data in sorted(
        report["pipelines"].items(),
        key=lambda x: x[1].get("f1_mean", 0),
        reverse=True,
    ):
        f1_m = data.get("f1_mean", 0)
        f1_lo = data.get("f1_ci_lower", 0)
        f1_hi = data.get("f1_ci_upper", 0)
        em_m = data.get("em_mean", 0)
        em_lo = data.get("em_ci_lower", 0)
        em_hi = data.get("em_ci_upper", 0)
        n = data.get("n", 0)
        print(
            f"  {pipeline:<30s} {f1_m:>8.3f} [{f1_lo:.3f}, {f1_hi:.3f}] "
            f"{em_m:>8.3f} [{em_lo:.3f}, {em_hi:.3f}] {n:>5d}"
        )

    # Pairwise significance
    for metric in ["f1", "em"]:
        pairwise = report.get(f"pairwise_{metric}", {})
        if not pairwise:
            continue
        print(f"\n  Pairwise (Agentic vs others, {metric.upper()}):")
        for pipeline, data in sorted(
            pairwise.items(),
            key=lambda x: x[1].get("delta", 0),
            reverse=True,
        ):
            delta = data.get("delta", 0)
            boot_p = data.get("boot_p", 1)
            wilcox_p = data.get("wilcoxon_p", 1)
            d = data.get("cohens_d", 0)
            sig = (
                "***"
                if boot_p < 0.001
                else "**"
                if boot_p < 0.01
                else "*"
                if boot_p < 0.05
                else "n.s."
            )
            print(
                f"    vs {pipeline:<28s} Δ={delta:+.3f}  d={d:.3f}  "
                f"boot_p={boot_p:.4f}  wilcox_p={wilcox_p:.4f}  [{sig}]"
            )


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI and significance tests")
    parser.add_argument("path", nargs="?", help="Result directory")
    parser.add_argument("--pattern", help="Filter directories")
    parser.add_argument("--all", action="store_true", help="All result directories")
    parser.add_argument("--output", help="Save JSON report to file")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_base = project_root / "data" / "results"

    if args.path:
        dirs = [Path(args.path)]
    elif args.all or args.pattern:
        dirs = sorted(results_base.glob("202603*/"))
        if args.pattern:
            dirs = [d for d in dirs if args.pattern in d.name]
    else:
        parser.print_help()
        sys.exit(1)

    all_reports = []
    for d in dirs:
        if not d.is_dir():
            continue
        try:
            report = analyze_directory(d)
            all_reports.append(report)
            print_report(report)
        except Exception as e:
            logger.error(f"Error analyzing {d.name}: {e}")
            all_reports.append({"dir": d.name, "error": str(e)})

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
