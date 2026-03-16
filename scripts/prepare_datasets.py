"""Download and prepare benchmark datasets for experiments.

Datasets:
  1. PopQA       — Short-form factoid QA (accuracy metric)
  2. HotpotQA    — Multi-hop reasoning QA (F1, EM metrics)
  3. Natural Questions (NQ) — Open-domain QA (accuracy metric)

Each dataset is saved as JSONL with a unified schema:
  {"id": str, "question": str, "answer": str, "metadata": {...}}

Usage:
  uv run python scripts/prepare_datasets.py [--dataset popqa|hotpotqa|nq|all]
  uv run python scripts/prepare_datasets.py --dataset all --sample 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.config.settings import settings


def prepare_popqa(output_dir: Path, sample_size: int | None = None) -> Path:
    """Prepare PopQA dataset.

    PopQA: 14k short-form factoid questions from Wikidata.
    Used by CRAG and Self-RAG for evaluation.
    """
    from datasets import load_dataset

    logger.info("Loading PopQA dataset...")
    ds = load_dataset("akariasai/PopQA", split="test")

    items = []
    for i, row in enumerate(ds):
        if sample_size and i >= sample_size:
            break
        items.append(
            {
                "id": f"popqa_{i}",
                "question": row["question"],
                "answer": row["possible_answers"][0] if row["possible_answers"] else row["obj"],
                "all_answers": row["possible_answers"] if row["possible_answers"] else [row["obj"]],
                "metadata": {
                    "subject": row.get("subj", ""),
                    "relation": row.get("prop", ""),
                    "object": row.get("obj", ""),
                    "popularity": row.get("s_pop", 0),
                },
            }
        )

    out_path = output_dir / "popqa.jsonl"
    _save_jsonl(items, out_path)
    return out_path


def prepare_hotpotqa(output_dir: Path, sample_size: int | None = None) -> Path:
    """Prepare HotpotQA dataset.

    HotpotQA: Multi-hop questions requiring reasoning over 2+ documents.
    """
    from datasets import load_dataset

    logger.info("Loading HotpotQA dataset...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    items = []
    for i, row in enumerate(ds):
        if sample_size and i >= sample_size:
            break

        # Collect supporting facts
        supporting_facts = []
        for title, sent_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
            strict=False,
        ):
            supporting_facts.append({"title": title, "sent_id": sent_id})

        # Collect context paragraphs as passages
        passages = []
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"], strict=False
        ):
            passages.append(
                {
                    "title": title,
                    "content": " ".join(sentences),
                }
            )

        items.append(
            {
                "id": f"hotpotqa_{row['id']}",
                "question": row["question"],
                "answer": row["answer"],
                "all_answers": [row["answer"]],
                "metadata": {
                    "type": row["type"],
                    "level": row["level"],
                    "supporting_facts": supporting_facts,
                },
                "passages": passages,
            }
        )

    out_path = output_dir / "hotpotqa.jsonl"
    _save_jsonl(items, out_path)
    return out_path


def prepare_nq(output_dir: Path, sample_size: int | None = None) -> Path:
    """Prepare Natural Questions (NQ) dataset.

    NQ: Open-domain questions from Google Search logs with
    human-verified answers from Wikipedia.
    """
    from datasets import load_dataset

    logger.info("Loading Natural Questions dataset...")
    ds = load_dataset("nq_open", split="validation")

    items = []
    for i, row in enumerate(ds):
        if sample_size and i >= sample_size:
            break
        items.append(
            {
                "id": f"nq_{i}",
                "question": row["question"],
                "answer": row["answer"][0] if row["answer"] else "",
                "all_answers": row["answer"],
                "metadata": {},
            }
        )

    out_path = output_dir / "natural_questions.jsonl"
    _save_jsonl(items, out_path)
    return out_path


def _save_jsonl(items: list[dict], path: Path) -> None:
    """Save items as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(items)} items to {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["popqa", "hotpotqa", "nq", "all"],
        default="all",
        help="Which dataset to prepare",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size (for quick testing). None = full dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.raw_dir,
        help="Output directory for datasets",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("popqa", "all"):
        prepare_popqa(output_dir, args.sample)

    if args.dataset in ("hotpotqa", "all"):
        prepare_hotpotqa(output_dir, args.sample)

    if args.dataset in ("nq", "all"):
        prepare_nq(output_dir, args.sample)

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
