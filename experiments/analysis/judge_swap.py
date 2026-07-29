"""Does the reported Judge column survive re-scoring with a different judge?

The panel in `judge_panel` shows the judges agree with the human annotators
and with each other on 200 items. This asks the larger question directly: score
every RQ1 result again with an alternative judge and see whether the pipeline
ranking each dataset reports actually moves.

Only verdicts marked judge_status "ok" count. A truncated or unparsed
completion produced no judgement, and scoring it as 0.0 would push exactly the
pipelines whose answers run longest toward looking worse.

Usage:
    uv run python -m experiments.analysis.judge_swap
    uv run python -m experiments.analysis.judge_swap --alternative minimax-m3
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

PIPELINES = (
    "naive_rag",
    "crag_replica",
    "single-pass",
    "loop_refinement",
    "agentic_(react)",
)


def _label(directory: Path) -> str:
    """Human-readable '<dataset> / <model>' for a results directory."""
    name = directory.name
    m = re.match(r"\d+_rq1_([a-z0-9]+)_n\d+_(.+)$", name)
    return f"{m.group(1)} / {m.group(2)}" if m else f"{name} / gpt-oss"


def _rates(directory: Path, suffix: str) -> tuple[dict[str, float], int]:
    """Acceptance rate per pipeline, and how many verdicts were unusable."""
    rates, unusable = {}, 0
    for pipeline in PIPELINES:
        matches = glob.glob(str(directory / f"rq1_{pipeline}{suffix}"))
        if not matches:
            continue
        scores = []
        with open(matches[0], encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("judge_status", "ok") != "ok":
                    unusable += 1
                    continue
                scores.append(float(r["llm_judge"]))
        if scores:
            rates[pipeline] = sum(scores) / len(scores)
    return rates, unusable


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pipeline ranking across judges")
    parser.add_argument("--alternative", default="minimax-m3")
    parser.add_argument("--paper-root", default="data/results/paper", type=Path)
    parser.add_argument("--openweight-root", default="data/results-openweight/final", type=Path)
    args = parser.parse_args()

    directories = sorted(args.paper_root.glob("*rq1_*")) + sorted(args.openweight_root.glob("*"))
    baseline_suffix = "_judged.jsonl"
    alt_suffix = f"_judged_{args.alternative}_*.jsonl"

    agree = disagree = 0
    total_unusable = 0
    gaps: list[tuple[str, dict, dict]] = []
    print(f"{'dataset / model':34s}{'judge':14s}" + "".join(f"{p[:9]:>11s}" for p in PIPELINES))
    for directory in directories:
        base, _ = _rates(directory, baseline_suffix)
        # gpt-oss results were scored under a judge-tagged name, not the plain one.
        if not base:
            base, _ = _rates(directory, "_judged_gpt-4-1-nano.jsonl")
        alt, unusable = _rates(directory, alt_suffix)
        total_unusable += unusable
        if not base or not alt:
            continue

        for label, rates in (("gpt-4.1-nano", base), (args.alternative, alt)):
            row = f"{_label(directory) if label == 'gpt-4.1-nano' else '':34s}{label:14s}"
            print(row + "".join(f"{rates.get(p, float('nan')):10.1%} " for p in PIPELINES))

        # The paper's claim is TARA over the fixed loop, not "TARA ranks first",
        # so that gap is what has to survive a change of judge. Which pipeline
        # happens to top a dataset is a brittle summary — several are separated
        # by well under a point on n=150-200, where `max` picks between ties.
        gaps.append((_label(directory), base, alt))
        same = (alt["agentic_(react)"] - alt["loop_refinement"] > 0) == (
            base["agentic_(react)"] - base["loop_refinement"] > 0
        )
        agree += same
        disagree += not same
        print(
            f"{'':34s}{'→ TARA-Loop':14s}"
            f"{base['agentic_(react)'] - base['loop_refinement']:+.1%} / "
            f"{alt['agentic_(react)'] - alt['loop_refinement']:+.1%}"
            f"  {'부호 동일' if same else '⚠️ 부호 반전'}\n"
        )

    print(f"{'dataset / model':34s}{'TARA-Loop (nano)':>20s}{'TARA-Loop (alt)':>20s}")
    for label, base, alt in gaps:
        b = base["agentic_(react)"] - base["loop_refinement"]
        a = alt["agentic_(react)"] - alt["loop_refinement"]
        print(f"{label:34s}{b:>19.1%}{a:>20.1%}")
    print(f"\nTARA-Loop 부호 유지: {agree}/{agree + disagree}개 조합")
    if total_unusable:
        print(f"사용불가로 제외된 판정: {total_unusable}건")


if __name__ == "__main__":
    main()
