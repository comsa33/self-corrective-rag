"""Multi-judge robustness: does the reported Judge metric depend on its judge?

The paper scores answers with gpt-4.1-nano and reports that alongside EM and F1,
which invites the obvious question of whether the Judge column is an artifact of
that one small model. This module scores the same human-labelled panel with
several other judges and reports two things: how well each judge agrees with the
human annotators, and how well the judges agree with each other.

Panel items join on (pipeline, id), not id alone. The same question appears under
several pipelines — 200 rows carry only 184 distinct ids — so joining on id would
silently pair one pipeline's verdict with another's answer.

Judge-human agreement is measured only where the two annotators agreed, so a
judge is never marked wrong on a case the humans could not settle themselves.

Usage:
    uv run python -m experiments.analysis.judge_panel
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from itertools import combinations
from pathlib import Path

from experiments.analysis.kappa_analysis import bootstrap_ci, cohens_kappa, normalise

BASELINE = "gpt-4.1-nano"


def load_key(path: Path) -> dict[str, tuple[str, str]]:
    """sample_id -> (pipeline, id), the composite that identifies a panel item."""
    with open(path, encoding="utf-8-sig") as handle:
        return {r["sample_id"]: (r["pipeline"], r["id"]) for r in csv.DictReader(handle)}


def load_humans(labels_dir: Path) -> dict[str, dict[int, int | None]]:
    humans: dict[str, dict[int, int | None]] = {}
    for index, sheet in enumerate(sorted(labels_dir.glob("*annotator*.csv")), 1):
        with open(sheet, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                humans.setdefault(row["sample_id"], {})[index] = normalise(row["human_label"])
    return humans


def load_verdicts(pattern: str) -> dict[tuple[str, str], int | None]:
    verdicts: dict[tuple[str, str], int | None] = {}
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    verdicts[(r["pipeline"], str(r["id"]))] = normalise(r["llm_judge"])
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-judge agreement on the human panel")
    parser.add_argument("--panel-dir", default="data/results-judge/kappa-panel", type=Path)
    parser.add_argument(
        "--key", default="data/human_eval/round2/human_eval_round2_key.csv", type=Path
    )
    parser.add_argument("--labels", default="data/human_eval/round2", type=Path)
    parser.add_argument("--baseline-root", default="data/results-openweight/final", type=Path)
    args = parser.parse_args()

    key = load_key(args.key)
    humans = load_humans(args.labels)

    judges: dict[str, dict[tuple[str, str], int | None]] = {
        BASELINE: load_verdicts(f"{args.baseline_root}/*/rq1_*_judged_gpt-4-1-nano.jsonl")
    }
    for path in sorted(args.panel_dir.glob("panel200_judged_*.jsonl")):
        name = path.stem.removeprefix("panel200_judged_").rsplit("_", 1)[0]
        judges[name] = load_verdicts(str(path))

    # Only items both annotators settled: a judge should not be scored against a
    # case the humans themselves split on.
    settled = [s for s, v in humans.items() if v.get(1) is not None and v[1] == v.get(2)]
    dropped = len(humans) - len(settled)
    print(
        f"panel {len(humans)} items, {len(settled)} settled by both annotators ({dropped} split)\n"
    )

    print(f"{'judge':24s}{'vs human κ':>12s}{'95% CI':>20s}{'agreement':>12s}{'missing':>9s}")
    for name, verdicts in judges.items():
        pairs, missing = [], 0
        for sid in settled:
            v = verdicts.get(key[sid])
            if v is None:
                missing += 1
                continue
            pairs.append((humans[sid][1], v))
        k = cohens_kappa(pairs)
        lo, hi = bootstrap_ci(pairs)
        agree = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else float("nan")
        mark = "  <- reported" if name == BASELINE else ""
        print(f"{name:24s}{k:12.3f}   [{lo:5.3f}, {hi:5.3f}]{agree:11.1%}{missing:9d}{mark}")

    print(f"\n{'judge pair':52s}{'κ':>8s}")
    for a, b in combinations(judges, 2):
        pairs = [
            (judges[a][key[s]], judges[b][key[s]])
            for s in settled
            if judges[a].get(key[s]) is not None and judges[b].get(key[s]) is not None
        ]
        print(f"{a + ' / ' + b:52s}{cohens_kappa(pairs):8.3f}")

    # Ranking stability: if every judge orders the pipelines the same way, the
    # reported Judge column cannot be an artifact of the one that was used.
    pipelines = sorted({p for p, _ in key.values()})
    print(f"\n{'judge':24s}" + "".join(f"{p[:11]:>13s}" for p in pipelines))
    orders = {}
    for name, verdicts in judges.items():
        rates = {}
        for p in pipelines:
            vals = [
                verdicts[k_] for s, k_ in key.items() if k_[0] == p and verdicts.get(k_) is not None
            ]
            rates[p] = sum(vals) / len(vals) if vals else float("nan")
        orders[name] = tuple(sorted(rates, key=lambda p: -rates[p]))
        print(f"{name:24s}" + "".join(f"{rates[p]:12.1%} " for p in pipelines))

    print("\nranking by acceptance rate")
    for name, order in orders.items():
        print(f"  {name:24s}{' > '.join(order)}")
    top = {o[0] for o in orders.values()}
    print(
        f"\n  top pipeline agreed by every judge: {top == {orders[BASELINE][0]}} ({', '.join(sorted(top))})"
    )


if __name__ == "__main__":
    main()
