"""Build retrieval indices from dataset passages.

Extracts passages from prepared datasets (HotpotQA context paragraphs,
FinanceBench evidence pages) and builds FAISS + BM25 hybrid indices
along with auxiliary section/term indices for agent tools.

Usage:
  uv run python scripts/build_index.py --dataset hotpotqa
  uv run python scripts/build_index.py --dataset financebench
  uv run python scripts/build_index.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.config.settings import settings
from agentic_rag.retriever.indexer import DocumentIndexer, Passage


def extract_passages_hotpotqa(raw_path: Path) -> list[Passage]:
    """Extract context paragraphs from HotpotQA dataset."""
    passages = []
    seen_titles = set()

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            for p in item.get("passages", []):
                title = p.get("title", "")
                content = p.get("content", "")
                if not content or title in seen_titles:
                    continue
                seen_titles.add(title)
                passages.append(
                    Passage(
                        id=f"hotpotqa_{title}",
                        title=title,
                        content=content,
                        source=f"hotpotqa/{title}",
                    )
                )

    logger.info(f"Extracted {len(passages)} unique passages from HotpotQA")
    return passages


def extract_passages_financebench(raw_path: Path) -> list[Passage]:
    """Extract evidence pages from FinanceBench dataset."""
    passages = []
    seen_ids = set()

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            for p in item.get("passages", []):
                title = p.get("title", "")
                content = p.get("content", "")
                source = p.get("source", "")
                page_num = p.get("page_num", 0)
                pid = f"fb_{source}_p{page_num}"

                if not content or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                passages.append(
                    Passage(
                        id=pid,
                        title=title,
                        content=content,
                        source=source,
                        metadata={"page_num": page_num},
                    )
                )

    logger.info(f"Extracted {len(passages)} unique passages from FinanceBench")
    return passages


DATASET_EXTRACTORS = {
    "hotpotqa": ("hotpotqa.jsonl", extract_passages_hotpotqa),
    "financebench": ("financebench.jsonl", extract_passages_financebench),
}


def build_index_for_dataset(dataset_name: str) -> None:
    """Build hybrid retrieval index for a dataset."""
    if dataset_name not in DATASET_EXTRACTORS:
        logger.error(
            f"No passage extractor for '{dataset_name}'. "
            f"Available: {list(DATASET_EXTRACTORS.keys())}"
        )
        return

    filename, extractor = DATASET_EXTRACTORS[dataset_name]
    raw_path = settings.raw_dir / filename
    if not raw_path.exists():
        logger.error(
            f"Dataset not found: {raw_path}\n"
            f"Run: uv run python scripts/prepare_datasets.py --dataset {dataset_name}"
        )
        return

    # Extract passages
    passages = extractor(raw_path)
    if not passages:
        logger.error(f"No passages extracted from {dataset_name}")
        return

    # Build index
    index_dir = settings.index_dir / dataset_name
    index_dir.mkdir(parents=True, exist_ok=True)

    indexer = DocumentIndexer()
    indexer.passages = indexer.chunk_passages(passages)
    indexer.retriever.build_index(indexer.passages)
    indexer.section_index.build(indexer.passages)
    indexer.term_index.build(indexer.passages)
    indexer.save(index_dir)

    logger.info(
        f"Index built for {dataset_name}: {len(indexer.passages)} chunks, saved to {index_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build retrieval indices from datasets")
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "financebench", "all"],
        default="all",
        help="Which dataset to build index for",
    )
    args = parser.parse_args()

    datasets = list(DATASET_EXTRACTORS.keys()) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        build_index_for_dataset(name)

    logger.info("Index building complete!")


if __name__ == "__main__":
    main()
