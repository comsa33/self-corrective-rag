"""Build a blind labelling sheet for validating the LLM judge.

Cohen's kappa only means something if the human labels are produced without
sight of the judge's verdict, so this sheet deliberately carries no judge
column. The verdicts are matched back by (dataset, pipeline, id) after the
sheet comes back filled in.

Sampling is stratified across pipelines so the sheet spans the full range of
answer quality — judging only the strongest pipeline would measure agreement
on easy cases and miss where the judge actually struggles.

Usage:
    uv run python scripts/make_human_eval_csv.py --per-dataset 50 --out human_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "financebench")
PIPELINES = (
    "naive_rag",
    "crag_replica",
    "single-pass",
    "loop_refinement",
    "agentic_(react)",
)


def load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build(results_root: Path, per_dataset: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    per_pipeline = per_dataset // len(PIPELINES)

    for dataset in DATASETS:
        directory = results_root / dataset
        if not directory.exists():
            continue
        for pipeline in PIPELINES:
            path = directory / f"rq1_{pipeline}.jsonl"
            if not path.exists():
                continue
            records = [r for r in load(path) if "error" not in r]
            for r in rng.sample(records, min(per_pipeline, len(records))):
                rows.append(
                    {
                        "dataset": dataset,
                        "pipeline": pipeline,
                        "id": r["id"],
                        "question": r.get("question", ""),
                        "reference_answer": r.get("reference", ""),
                        "predicted_answer": str(r.get("prediction", "")),
                        "human_label": "",  # 1 = correct, 0 = incorrect
                    }
                )

    # Shuffle so the labeller cannot infer the pipeline from row order.
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blind human-eval sheet")
    parser.add_argument("--results-root", default="data/results-openweight/final")
    parser.add_argument("--per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="human_eval.csv")
    args = parser.parse_args()

    rows = build(Path(args.results_root), args.per_dataset, args.seed)
    if not rows:
        raise SystemExit(f"No results found under {args.results_root}")

    out = Path(args.out)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_dataset: dict[str, int] = {}
    for r in rows:
        by_dataset[r["dataset"]] = by_dataset.get(r["dataset"], 0) + 1
    print(f"{len(rows)} rows -> {out}")
    for dataset, count in by_dataset.items():
        print(f"  {dataset}: {count}")
    print("\nhuman_label 열에 1(정답) 또는 0(오답)을 채워 주세요.")
    print("judge 판정은 의도적으로 빠져 있습니다 — 보고 나면 일치도가 무의미해집니다.")


if __name__ == "__main__":
    main()
