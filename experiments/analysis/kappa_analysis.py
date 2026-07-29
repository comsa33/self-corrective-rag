"""Judge validation: inter-annotator and judge-human agreement.

The LLM judge is one of the paper's three headline metrics, so the claim that
it tracks human judgement has to be reproducible from the raw labels rather
than quoted from a spreadsheet. This module joins the blind labelling sheets
back to their key file, pairs them with the judge's verdicts, and reports
Cohen's kappa.

Confidence intervals come from a bootstrap rather than the asymptotic standard
error: the per-dataset cells are n=50 and several are near-degenerate, with one
label taking almost every item, and the normal approximation for kappa is not
trustworthy there.

Judge agreement is measured only on items where the two annotators agree, so
that "the judge is wrong" is never scored against a case the humans themselves
could not settle. The count of dropped items is reported alongside.

Usage:
    uv run python -m experiments.analysis.kappa_analysis \\
        --labels data/human_eval/round2 \\
        --key data/human_eval/round2/human_eval_round2_key.csv \\
        --judged-root data/results-openweight/final
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from collections import Counter
from pathlib import Path

DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "financebench")
PIPELINES = (
    "naive_rag",
    "crag_replica",
    "single-pass",
    "loop_refinement",
    "agentic_(react)",
)
TRUE_TOKENS = {"1", "1.0", "y", "yes", "true", "correct"}
FALSE_TOKENS = {"0", "0.0", "n", "no", "false", "incorrect"}


def normalise(value: object) -> int | None:
    """Map a label to 1/0. Annotators write Y/N; the judge writes 1.0/0.0."""
    token = str(value).strip().lower()
    if token in TRUE_TOKENS:
        return 1
    if token in FALSE_TOKENS:
        return 0
    return None


def cohens_kappa(pairs: list[tuple[int, int]]) -> float:
    """Cohen's kappa over (rater_a, rater_b) binary labels."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(1 for a, b in pairs if a == b) / n
    count_a, count_b = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(count_a[label] / n * count_b[label] / n for label in (0, 1))
    # Perfect expected agreement means one label took every item under both
    # raters; kappa is undefined there, and reporting 1.0 matches the intuition
    # that the raters never actually disagreed.
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def bootstrap_ci(
    pairs: list[tuple[int, int]], reps: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap interval for `cohens_kappa`, resampling items."""
    if not pairs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(pairs)
    values = []
    for _ in range(reps):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        kappa = cohens_kappa(sample)
        if kappa == kappa:  # drop NaN draws
            values.append(kappa)
    values.sort()
    return (values[int(0.025 * len(values))], values[int(0.975 * len(values))])


def load_labels(labels_dir: Path, key_path: Path) -> list[dict]:
    """Join annotator sheets to the key that maps sample_id back to a result."""
    with open(key_path, encoding="utf-8-sig") as handle:
        key = {row["sample_id"]: row for row in csv.DictReader(handle)}
    humans: dict[str, dict[int, int | None]] = {}
    sheets = sorted(labels_dir.glob("*annotator*.csv"))
    if not sheets:
        raise SystemExit(f"No *annotator*.csv under {labels_dir}")
    for index, sheet in enumerate(sheets, 1):
        with open(sheet, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                humans.setdefault(row["sample_id"], {})[index] = normalise(row["human_label"])

    unlabelled = [s for s, v in humans.items() if any(x is None for x in v.values())]
    if unlabelled:
        raise SystemExit(f"{len(unlabelled)} rows carry an unreadable label, e.g. {unlabelled[:5]}")
    if set(humans) != set(key):
        raise SystemExit("sample_id sets differ between the sheets and the key")

    return [{"sample_id": s, **key[s], "humans": humans[s]} for s in sorted(humans)]


def load_judge(judged_root: Path, judge_model: str) -> dict[tuple[str, str, str], int | None]:
    tag = judge_model.split("/")[-1].replace(":", "-").replace(".", "-")
    verdicts = {}
    for path in glob.glob(str(judged_root / "*" / f"rq1_*_judged_{tag}*.jsonl")):
        parts = Path(path)
        dataset = parts.parent.name
        pipeline = parts.name.split("_judged_")[0].removeprefix("rq1_")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    verdicts[(dataset, pipeline, str(record["id"]))] = normalise(
                        record["llm_judge"]
                    )
    return verdicts


def summarise(rows: list[dict], label: str) -> dict:
    human_pairs = [(r["humans"][1], r["humans"][2]) for r in rows]
    settled = [r for r in rows if r["humans"][1] == r["humans"][2]]
    judge_pairs = [(r["humans"][1], r["judge"]) for r in settled]
    agreement = (
        sum(1 for a, b in judge_pairs if a == b) / len(judge_pairs) if judge_pairs else float("nan")
    )
    return {
        "label": label,
        "n": len(rows),
        "disagreements": len(rows) - len(settled),
        "human_kappa": cohens_kappa(human_pairs),
        "human_ci": bootstrap_ci(human_pairs),
        "judge_kappa": cohens_kappa(judge_pairs),
        "judge_ci": bootstrap_ci(judge_pairs),
        "judge_agreement": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge/human agreement from blind label sheets")
    parser.add_argument("--labels", default="data/human_eval/round2", type=Path)
    parser.add_argument(
        "--key", default="data/human_eval/round2/human_eval_round2_key.csv", type=Path
    )
    parser.add_argument("--judged-root", default="data/results-openweight/final", type=Path)
    parser.add_argument("--judge-model", default="openai/gpt-4.1-nano")
    args = parser.parse_args()

    rows = load_labels(args.labels, args.key)
    verdicts = load_judge(args.judged_root, args.judge_model)
    missing = 0
    joined = []
    for row in rows:
        verdict = verdicts.get((row["dataset"], row["pipeline"], row["id"]))
        if verdict is None:
            missing += 1
            continue
        joined.append({**row, "judge": verdict})
    print(f"joined {len(joined)}/{len(rows)} items (no judge verdict for {missing})\n")

    reports = [summarise([r for r in joined if r["dataset"] == d], d) for d in DATASETS]
    reports = [r for r in reports if r["n"]]
    reports.append(summarise(joined, "ALL"))
    for report in reports:
        print(
            f"  {report['label']:18s} n={report['n']:3d}  disagreements={report['disagreements']:2d}\n"
            f"    inter-annotator  {report['human_kappa']:6.3f}"
            f"  95% CI [{report['human_ci'][0]:.3f}, {report['human_ci'][1]:.3f}]\n"
            f"    judge            {report['judge_kappa']:6.3f}"
            f"  95% CI [{report['judge_ci'][0]:.3f}, {report['judge_ci'][1]:.3f}]"
            f"   agreement {report['judge_agreement']:.1%}"
        )

    print(f"\n  {'pipeline':18s}{'human':>9s}{'judge':>9s}{'human-judge':>13s}")
    for pipeline in PIPELINES:
        subset = [r for r in joined if r["pipeline"] == pipeline]
        if not subset:
            continue
        human = sum(sum(r["humans"].values()) / 2 for r in subset) / len(subset)
        judge = sum(r["judge"] for r in subset) / len(subset)
        print(f"  {pipeline:18s}{human:9.1%}{judge:9.1%}{human - judge:+12.1%}")


if __name__ == "__main__":
    main()
