"""Load per-question result rows across the per-model result trees.

Every analysis that compares pipelines across models needs the same thing:
the ``rq1_<pipeline>.jsonl`` rows of each run, tagged with the model,
dataset and pipeline they belong to. Two directory layouts exist:

* **stamped** — ``<stamp>_<prefix>_<dataset>_n<N>_<model-tag>/`` under
  ``data/results/paper`` (Gemini and gpt-5-mini, March 2026);
* **flat** — ``<dataset>/`` under ``data/results-openweight/final``
  (gpt-oss:120b, July 2026), where the model is not in the path and must
  be supplied by the caller.

``*_judged*.jsonl`` files are judge re-scorings of the same rows and are
skipped, so that a pipeline is never counted twice.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique", "financebench")

# Model tag as it appears in a stamped directory name -> label used in tables.
STAMPED_MODEL_TAGS = {
    "gemini-3.1-flash-lite": "gemini-flash-lite",
    "gpt-5-mini": "gpt-5-mini",
}

# The three trees behind every three-model table in the paper.
DEFAULT_TREES: tuple[tuple[str, str | None], ...] = (
    ("data/results/paper", None),
    ("data/results-openweight/final", "gpt-oss-120b"),
)


def _read_rows(jsonl_path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if "error" not in r]


def _model_from_stamped_name(name: str) -> str | None:
    for tag, label in STAMPED_MODEL_TAGS.items():
        if name.endswith(tag):
            return label
    return None


def load_runs(results_dir: str | Path, prefix: str = "rq1", model: str | None = None) -> list[dict]:
    """Return ``{model, dataset, pipeline, rows}`` for every run under ``results_dir``.

    Args:
        results_dir: root of one result tree (either layout).
        prefix: experiment prefix of the jsonl files, e.g. ``rq1``.
        model: label to use for the flat layout, where the path carries none.
    """
    results_dir = Path(results_dir)
    records: list[dict] = []

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue

        if subdir.name in DATASETS:
            # flat layout
            dataset = subdir.name
            run_model = model
            if run_model is None:
                logger.warning(f"Skipping {subdir}: flat layout needs an explicit model label")
                continue
        elif f"_{prefix}_" in subdir.name:
            # stamped layout
            dataset = next((d for d in DATASETS if d in subdir.name), None)
            run_model = model or _model_from_stamped_name(subdir.name)
            if dataset is None or run_model is None:
                logger.warning(f"Skipping (dataset/model not recognized): {subdir.name}")
                continue
        else:
            continue

        for jsonl_path in sorted(subdir.glob(f"{prefix}_*.jsonl")):
            if "_judged" in jsonl_path.stem or jsonl_path.stem.endswith("_summary"):
                continue
            records.append(
                {
                    "model": run_model,
                    "dataset": dataset,
                    "pipeline": jsonl_path.stem[len(prefix) + 1 :],
                    "rows": _read_rows(jsonl_path),
                }
            )

    logger.info(f"Loaded {len(records)} pipeline runs from {results_dir}")
    return records


def load_default_trees(prefix: str = "rq1") -> list[dict]:
    """Load the Gemini, gpt-5-mini and gpt-oss trees that back the paper's tables."""
    records: list[dict] = []
    for root, model in DEFAULT_TREES:
        if Path(root).is_dir():
            records.extend(load_runs(root, prefix=prefix, model=model))
        else:
            logger.warning(f"Result tree missing: {root}")
    return records


def best_f1(prediction: str, row: dict) -> float:
    """Token F1 against the best-matching reference answer."""
    from agentic_rag.evaluation.metrics import token_f1

    references = row.get("all_references") or [row.get("reference", "")]
    return max(token_f1(prediction, ref) for ref in references if ref is not None)
