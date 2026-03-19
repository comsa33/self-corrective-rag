"""Download and prepare benchmark datasets for experiments.

Datasets (Multi-hop QA — agentic RAG evaluation):
  1. HotpotQA          — Multi-hop reasoning QA (F1, EM metrics)
  2. 2WikiMultiHopQA    — Multi-hop QA over 2 Wikipedia articles (F1, EM)
  3. MuSiQue            — Multi-hop QA with 2-4 hop reasoning (F1, EM)

Datasets (Enterprise/Domain-specific — practical applicability):
  4. FinanceBench — Financial QA over SEC filings (10-K/10-Q/8-K)

Each dataset is saved as JSONL with a unified schema:
  {"id": str, "question": str, "answer": str, "metadata": {...}}

HotpotQA, 2WikiMultiHopQA, and MuSiQue include context paragraphs
(supporting + distractors) as self-contained retrieval corpora.
FinanceBench includes evidence passages from SEC filings.

Usage:
  uv run python scripts/prepare_datasets.py [--dataset hotpotqa|2wikimultihopqa|musique|financebench|all]
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


def prepare_2wikimultihopqa(output_dir: Path, sample_size: int | None = None) -> Path:
    """Prepare 2WikiMultiHopQA dataset.

    2WikiMultiHopQA: Multi-hop questions requiring reasoning over 2 Wikipedia
    articles. Same format as HotpotQA (context with title+sentences).
    Used by Search-o1, KiRAG, PRISM, FAIR-RAG (2025-2026 standard).
    """
    from datasets import load_dataset

    logger.info("Loading 2WikiMultiHopQA dataset...")
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")

    items = []
    for i, row in enumerate(ds):
        if sample_size and i >= sample_size:
            break

        # Same structure as HotpotQA: supporting_facts + context
        supporting_facts = []
        for title, sent_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
            strict=False,
        ):
            supporting_facts.append({"title": title, "sent_id": sent_id})

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
                "id": f"2wiki_{row.get('_id', row.get('id', i))}",
                "question": row["question"],
                "answer": row["answer"],
                "all_answers": [row["answer"]],
                "metadata": {
                    "type": row.get("type", ""),
                    "supporting_facts": supporting_facts,
                },
                "passages": passages,
            }
        )

    out_path = output_dir / "2wikimultihopqa.jsonl"
    _save_jsonl(items, out_path)
    return out_path


def prepare_musique(output_dir: Path, sample_size: int | None = None) -> Path:
    """Prepare MuSiQue dataset.

    MuSiQue: Multi-hop questions with 2-4 reasoning hops and
    explicit question decomposition. Includes supporting + distractor
    paragraphs per question (self-contained corpus).
    Used by Search-o1, KiRAG, PRISM, FAIR-RAG (2025-2026 standard).
    """
    from datasets import load_dataset

    logger.info("Loading MuSiQue dataset...")
    ds = load_dataset("bdsaglam/musique", split="validation")

    items = []
    for row in ds:
        # Filter to answerable questions only
        if not row.get("answerable", True):
            continue

        if sample_size and len(items) >= sample_size:
            break

        # Extract paragraphs (supporting + distractors)
        passages = []
        supporting_indices = []
        for p in row.get("paragraphs", []):
            passages.append(
                {
                    "title": p.get("title", ""),
                    "content": p.get("paragraph_text", ""),
                }
            )
            if p.get("is_supporting", False):
                supporting_indices.append(p.get("idx", 0))

        # Build answer aliases list
        all_answers = [row["answer"]]
        for alias in row.get("answer_aliases", []):
            if alias and alias not in all_answers:
                all_answers.append(alias)

        # Extract question decomposition for metadata
        decomposition = []
        for step in row.get("question_decomposition", []):
            decomposition.append(
                {
                    "question": step.get("question", ""),
                    "answer": step.get("answer", ""),
                }
            )

        items.append(
            {
                "id": f"musique_{row['id']}",
                "question": row["question"],
                "answer": row["answer"],
                "all_answers": all_answers,
                "metadata": {
                    "answerable": row.get("answerable", True),
                    "n_hops": len(decomposition),
                    "decomposition": decomposition,
                    "supporting_paragraph_indices": supporting_indices,
                },
                "passages": passages,
            }
        )

    out_path = output_dir / "musique.jsonl"
    _save_jsonl(items, out_path)
    return out_path


def prepare_financebench(output_dir: Path, sample_size: int | None = None) -> Path:
    """Prepare FinanceBench dataset.

    FinanceBench: 150 financial QA pairs over SEC filings (10-K, 10-Q, 8-K)
    from publicly traded companies. Each question includes human-annotated
    answers with evidence passages extracted from the source documents.

    This dataset tests RAG on enterprise-grade structured documents with
    domain-specific terminology — validating C3 (section_index, term_index).

    Source: PatronusAI/financebench (CC-BY-NC-4.0, academic use permitted)
    """
    from datasets import load_dataset

    logger.info("Loading FinanceBench dataset...")
    ds = load_dataset("PatronusAI/financebench", split="train")

    items = []
    for i, row in enumerate(ds):
        if sample_size and i >= sample_size:
            break

        # Extract evidence passages from the structured evidence field
        evidence_passages = []
        for ev in row.get("evidence", []):
            evidence_passages.append(
                {
                    "title": f"{ev.get('doc_name', row.get('doc_name', ''))} (p.{ev.get('evidence_page_num', '?')})",
                    "content": ev.get("evidence_text_full_page", ev.get("evidence_text", "")),
                    "source": ev.get("doc_name", row.get("doc_name", "")),
                    "page_num": ev.get("evidence_page_num", 0),
                }
            )

        items.append(
            {
                "id": f"financebench_{row.get('financebench_id', i)}",
                "question": row["question"],
                "answer": row["answer"],
                "all_answers": [row["answer"]],
                "metadata": {
                    "company": row.get("company", ""),
                    "doc_name": row.get("doc_name", ""),
                    "doc_type": row.get("doc_type", ""),
                    "doc_period": row.get("doc_period", ""),
                    "question_type": row.get("question_type", ""),
                    "question_reasoning": row.get("question_reasoning", ""),
                    "gics_sector": row.get("gics_sector", ""),
                    "justification": row.get("justification", ""),
                },
                "passages": evidence_passages,
            }
        )

    out_path = output_dir / "financebench.jsonl"
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
        choices=["hotpotqa", "2wikimultihopqa", "musique", "financebench", "all"],
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

    if args.dataset in ("hotpotqa", "all"):
        prepare_hotpotqa(output_dir, args.sample)

    if args.dataset in ("2wikimultihopqa", "all"):
        prepare_2wikimultihopqa(output_dir, args.sample)

    if args.dataset in ("musique", "all"):
        prepare_musique(output_dir, args.sample)

    if args.dataset in ("financebench", "all"):
        prepare_financebench(output_dir, args.sample)

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
