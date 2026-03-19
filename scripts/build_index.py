"""Build retrieval indices from dataset passages.

Extracts passages from prepared datasets (HotpotQA / 2WikiMultiHopQA /
MuSiQue context paragraphs, FinanceBench evidence pages) and builds
FAISS + BM25 hybrid indices along with auxiliary section/term indices
for agent tools.

Usage:
  uv run python scripts/build_index.py --dataset hotpotqa
  uv run python scripts/build_index.py --dataset 2wikimultihopqa
  uv run python scripts/build_index.py --dataset musique
  uv run python scripts/build_index.py --dataset financebench
  uv run python scripts/build_index.py --dataset all
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
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


def extract_passages_2wikimultihopqa(raw_path: Path) -> list[Passage]:
    """Extract context paragraphs from 2WikiMultiHopQA dataset.

    Same format as HotpotQA — context with title + content.
    """
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
                        id=f"2wiki_{title}",
                        title=title,
                        content=content,
                        source=f"2wikimultihopqa/{title}",
                    )
                )

    logger.info(f"Extracted {len(passages)} unique passages from 2WikiMultiHopQA")
    return passages


def extract_passages_musique(raw_path: Path) -> list[Passage]:
    """Extract paragraphs from MuSiQue dataset."""
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
                        id=f"musique_{title}",
                        title=title,
                        content=content,
                        source=f"musique/{title}",
                    )
                )

    logger.info(f"Extracted {len(passages)} unique passages from MuSiQue")
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


# ---------------------------------------------------------------------------
# Wikipedia (DPR psgs_w100.tsv) — large-scale index with batched build
# ---------------------------------------------------------------------------

WIKIPEDIA_TSV_GZ = "wikipedia/psgs_w100.tsv.gz"
EMBED_BATCH_SIZE = 50_000  # passages per embedding batch
BM25_BATCH_SIZE = 500_000  # passages per BM25 batch


def _iter_wikipedia_passages(tsv_gz_path: Path):
    """Yield Passage objects from the DPR Wikipedia TSV (gzipped).

    Format: id<TAB>text<TAB>title  (header row first)
    """
    count = 0
    with gzip.open(tsv_gz_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            pid, text, title = row[0], row[1], row[2]
            if not text:
                continue
            yield Passage(
                id=f"wiki_{pid}",
                title=title,
                content=text,
                source=f"wikipedia/{title}",
            )
            count += 1
            if count % 1_000_000 == 0:
                logger.info(f"  ... read {count:,} passages")

    logger.info(f"Total Wikipedia passages read: {count:,}")


def build_wikipedia_index(skip_faiss: bool = False, skip_bm25: bool = False) -> None:
    """Build FAISS + BM25 index for DPR Wikipedia (21M passages).

    Uses batched embedding to avoid OOM on large corpora.
    --skip-faiss: resume from Phase 3 when FAISS already built.
    --skip-bm25: skip BM25 entirely (dense-only retrieval for Wikipedia).
    """
    tsv_gz_path = settings.raw_dir / WIKIPEDIA_TSV_GZ
    if not tsv_gz_path.exists():
        logger.error(
            f"Wikipedia dump not found: {tsv_gz_path}\n"
            "Download: wget -O data/raw/wikipedia/psgs_w100.tsv.gz "
            "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
        )
        return

    index_dir = settings.index_dir / "wikipedia"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Read all passages and save passage metadata
    logger.info("Phase 1/4: Reading passages from TSV...")
    t0 = time.time()
    passages: list[Passage] = list(_iter_wikipedia_passages(tsv_gz_path))
    logger.info(f"Read {len(passages):,} passages in {time.time() - t0:.0f}s")

    # DPR passages are already ~100 words — no chunking needed
    if skip_faiss:
        logger.info("Phase 2/4: Skipping FAISS (--skip-faiss flag set, using existing index)")
    else:
        logger.info("Phase 2/4: Building FAISS index (batched embedding)...")
        t0 = time.time()
        _build_faiss_batched(passages, index_dir)
        logger.info(f"FAISS index built in {time.time() - t0:.0f}s")

    if skip_bm25:
        logger.info(
            "Phase 3/4: Skipping BM25 (--skip-bm25 flag set) — Wikipedia uses dense-only retrieval"
        )
    else:
        logger.info("Phase 3/4: Building BM25 index...")
        t0 = time.time()
        _build_bm25(passages, index_dir)
        logger.info(f"BM25 index built in {time.time() - t0:.0f}s")

    logger.info("Phase 4/4: Saving passage metadata + auxiliary indices...")
    t0 = time.time()
    _save_passages_and_aux(passages, index_dir)
    logger.info(f"Metadata saved in {time.time() - t0:.0f}s")

    logger.info(f"Wikipedia index complete: {len(passages):,} passages, saved to {index_dir}")


def _build_faiss_batched(passages: list[Passage], index_dir: Path) -> None:
    """Build FAISS index in batches to avoid OOM."""
    from agentic_rag.retriever.dense import DenseRetriever

    dense = DenseRetriever()
    embed_fn = dense._get_embed_fn()

    # Embed first batch to get dimension
    first_batch = [p.content for p in passages[: min(EMBED_BATCH_SIZE, len(passages))]]
    vecs = embed_fn(first_batch)
    dim = vecs.shape[1]
    faiss.normalize_L2(vecs)

    # Create index
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    all_ids = [p.id for p in passages[: len(first_batch)]]

    logger.info(f"  Batch 1: {index.ntotal:,} vectors (dim={dim})")

    # Process remaining batches
    for batch_start in range(EMBED_BATCH_SIZE, len(passages), EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, len(passages))
        batch_texts = [p.content for p in passages[batch_start:batch_end]]
        batch_ids = [p.id for p in passages[batch_start:batch_end]]

        vecs = embed_fn(batch_texts)
        faiss.normalize_L2(vecs)
        index.add(vecs)
        all_ids.extend(batch_ids)

        batch_num = batch_start // EMBED_BATCH_SIZE + 1
        logger.info(f"  Batch {batch_num + 1}: {index.ntotal:,} vectors total")

    # Save
    dense_dir = index_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(dense_dir / "faiss.index"))
    np.save(dense_dir / "passage_ids.npy", np.array(all_ids))
    logger.info(f"  FAISS saved: {index.ntotal:,} vectors")


def _build_bm25(passages: list[Passage], index_dir: Path) -> None:
    """Build BM25 index."""
    from agentic_rag.retriever.sparse import SparseRetriever

    sparse = SparseRetriever()
    sparse.build_index(passages)
    sparse.save(index_dir / "sparse")
    logger.info(f"  BM25 saved: {len(passages):,} documents")


def _save_passages_and_aux(passages: list[Passage], index_dir: Path) -> None:
    """Save passage JSONL and auxiliary indices (section/term)."""
    # Save passages
    with open(index_dir / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(
                json.dumps(
                    {"id": p.id, "title": p.title, "content": p.content, "source": p.source},
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Build section and term indices
    from agentic_rag.retriever.section_index import SectionIndex
    from agentic_rag.retriever.term_index import TermIndex

    section_idx = SectionIndex()
    section_idx.build(passages)
    section_idx.save(index_dir / "section_index.json")

    term_idx = TermIndex()
    term_idx.build(passages)
    term_idx.save(index_dir / "term_index.json")


# ---------------------------------------------------------------------------
# Registry and main
# ---------------------------------------------------------------------------

DATASET_EXTRACTORS = {
    "hotpotqa": ("hotpotqa.jsonl", extract_passages_hotpotqa),
    "2wikimultihopqa": ("2wikimultihopqa.jsonl", extract_passages_2wikimultihopqa),
    "musique": ("musique.jsonl", extract_passages_musique),
    "financebench": ("financebench.jsonl", extract_passages_financebench),
}

# Wikipedia uses a separate build path due to scale
LARGE_DATASETS = {"wikipedia"}


def build_index_for_dataset(
    dataset_name: str, skip_faiss: bool = False, skip_bm25: bool = False
) -> None:
    """Build hybrid retrieval index for a dataset."""
    if dataset_name == "wikipedia":
        build_wikipedia_index(skip_faiss=skip_faiss, skip_bm25=skip_bm25)
        return

    if dataset_name not in DATASET_EXTRACTORS:
        logger.error(
            f"No passage extractor for '{dataset_name}'. "
            f"Available: {list(DATASET_EXTRACTORS.keys()) + list(LARGE_DATASETS)}"
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
    all_datasets = list(DATASET_EXTRACTORS.keys()) + list(LARGE_DATASETS)
    parser = argparse.ArgumentParser(description="Build retrieval indices from datasets")
    parser.add_argument(
        "--dataset",
        choices=[*all_datasets, "all"],
        default="all",
        help="Which dataset to build index for",
    )
    parser.add_argument(
        "--skip-faiss",
        action="store_true",
        help="Skip FAISS phase (use existing index) and resume from BM25 — Wikipedia only",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip BM25 phase entirely — Wikipedia uses dense-only retrieval (avoids OOM on 21M)",
    )
    args = parser.parse_args()

    datasets = all_datasets if args.dataset == "all" else [args.dataset]
    for name in datasets:
        build_index_for_dataset(name, skip_faiss=args.skip_faiss, skip_bm25=args.skip_bm25)

    logger.info("Index building complete!")


if __name__ == "__main__":
    main()
