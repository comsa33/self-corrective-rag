"""Shared experiment utilities.

Common setup, data loading, result saving, and reporting
functions used across all RQ and ablation experiment scripts.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import dspy
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

from agentic_rag.config.settings import make_lm, settings
from agentic_rag.evaluation.metrics import evaluate_batch
from agentic_rag.pipeline.base import BasePipeline
from agentic_rag.retriever.hybrid import HybridRetriever
from agentic_rag.retriever.indexer import DocumentIndexer

console = Console()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_experiment(seed: int | None = None) -> None:
    """Initialize experiment environment with reproducible seed."""
    seed = seed or settings.experiment.seed
    random.seed(seed)
    np.random.seed(seed)

    # Configure DSPy LM — API keys are read from environment variables
    # (OPENAI_API_KEY, GEMINI_API_KEY, etc.) by litellm automatically.
    # make_lm centralizes retry/timeout settings.
    dspy.configure(lm=make_lm(settings.model.generate_model))

    logger.info(f"Experiment initialized: seed={seed}")


def load_dataset(name: str, sample_size: int | None = None) -> list[dict]:
    """Load a prepared dataset from data/raw/."""
    path = settings.raw_dir / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run: uv run python scripts/prepare_datasets.py --dataset {name}"
        )

    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))

    if sample_size and len(items) > sample_size:
        items = random.sample(items, sample_size)
        logger.info(f"Sampled {sample_size} from {name}")

    logger.info(f"Loaded dataset '{name}': {len(items)} items")
    return items


DATASET_INDEX_MAP = {
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
    "financebench": "financebench",
}


def load_retriever(
    index_dir: Path | None = None, dataset_name: str | None = None
) -> tuple[HybridRetriever, DocumentIndexer]:
    """Load pre-built retrieval indices.

    Args:
        index_dir: Explicit index directory. Takes precedence.
        dataset_name: Dataset name to look up in data/indices/{name}/.
            PopQA and NQ are mapped to the shared Wikipedia index.
    """
    if index_dir is None and dataset_name:
        index_name = DATASET_INDEX_MAP.get(dataset_name, dataset_name)
        index_dir = settings.index_dir / index_name
    index_dir = index_dir or settings.index_dir
    indexer = DocumentIndexer()
    retriever = indexer.load(index_dir)
    return retriever, indexer


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------
def run_pipeline_on_dataset(
    pipeline: BasePipeline,
    dataset: list[dict],
    pipeline_name: str = "pipeline",
    request_delay: float = 0.0,
) -> list[dict]:
    """Run a pipeline on a dataset and collect results.

    Args:
        request_delay: Seconds to wait between items (for API rate limiting).

    Returns list of result dicts with predictions and metadata.
    """
    results = []

    for i, item in enumerate(dataset):
        if i > 0 and request_delay > 0:
            time.sleep(request_delay)
        question = item["question"]
        reference = item.get("answer", "")

        start = time.perf_counter()
        try:
            result = pipeline.run(question)
            latency = time.perf_counter() - start

            results.append(
                {
                    "id": item.get("id", str(i)),
                    "question": question,
                    "reference": reference,
                    "all_references": item.get("all_answers", [reference]),
                    "prediction": result.answer,
                    "footnotes": result.footnotes,
                    "retry_count": result.retry_count,
                    "action_history": result.action_history,
                    "evaluation_scores": result.evaluation_scores,
                    "agent_type": result.agent_type,
                    "passages_used": len(result.passages_used),
                    "total_passages_retrieved": result.total_passages_retrieved,
                    "llm_calls": result.llm_calls,
                    "latency_seconds": latency,
                    "pipeline": pipeline_name,
                    # Mediation analysis fields
                    "tool_score_trace": result.tool_score_trace,
                    "question_difficulty": _extract_question_difficulty(item),
                }
            )
        except Exception as e:
            logger.error(f"Error on item {i}: {e}")
            results.append(
                {
                    "id": item.get("id", str(i)),
                    "question": question,
                    "reference": reference,
                    "prediction": "",
                    "error": str(e),
                    "pipeline": pipeline_name,
                }
            )

        if (i + 1) % 10 == 0:
            logger.info(f"[{pipeline_name}] {i + 1}/{len(dataset)} done")

    return results


# ---------------------------------------------------------------------------
# Result saving & reporting
# ---------------------------------------------------------------------------
def save_results(
    results: list[dict],
    experiment_name: str,
    extra_metadata: dict | None = None,
    *,
    run_dir: Path | None = None,
    compute_llm_judge: bool = False,
) -> Path:
    """Save experiment results to data/results/<run_dir>/."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if run_dir is None:
        run_dir = settings.results_dir / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / f"{experiment_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Save summary
    valid = [r for r in results if "error" not in r]
    predictions = [r.get("prediction", "") for r in valid]
    references = [r.get("reference", "") for r in valid]
    questions = [r.get("question", "") for r in valid]

    summary = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "total_items": len(results),
        "errors": sum(1 for r in results if "error" in r),
        "settings": {
            "quality_threshold": settings.evaluation.quality_threshold,
            "max_retry": settings.evaluation.max_retry_count,
            "top_k": settings.retrieval.top_k,
            "hybrid_weight": settings.retrieval.hybrid_weight,
            "seed": settings.experiment.seed,
        },
    }

    if predictions and references:
        metrics = evaluate_batch(
            predictions,
            references,
            questions=questions,
            compute_bert_score=False,
            compute_llm_judge=compute_llm_judge,
        )
        summary["metrics"] = metrics

    if extra_metadata:
        summary["extra"] = extra_metadata

    summary_path = run_dir / f"{experiment_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_path}")
    return out_path


def print_comparison_table(
    all_results: dict[str, list[dict]],
    title: str = "Experiment Results",
    compute_llm_judge: bool = False,
) -> None:
    """Print a rich comparison table of pipeline results."""
    table = Table(title=title)
    table.add_column("Pipeline", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("EM", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Judge", justify="right")
    table.add_column("Avg Retries", justify="right")
    table.add_column("Avg Latency (s)", justify="right")
    table.add_column("Errors", justify="right")

    for name, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        errors = len(results) - len(valid)

        if not valid:
            table.add_row(name, str(len(results)), "-", "-", "-", "-", "-", str(errors))
            continue

        preds = [r["prediction"] for r in valid]
        refs = [r["reference"] for r in valid]
        questions = [r.get("question", "") for r in valid]
        metrics = evaluate_batch(
            preds,
            refs,
            questions=questions,
            compute_bert_score=False,
            compute_llm_judge=compute_llm_judge,
        )

        avg_retries = np.mean([r.get("retry_count", 0) for r in valid])
        avg_latency = np.mean([r.get("latency_seconds", 0) for r in valid])

        judge_str = f"{metrics['llm_judge']:.3f}" if "llm_judge" in metrics else "-"

        table.add_row(
            name,
            str(len(valid)),
            f"{metrics['exact_match']:.3f}",
            f"{metrics['f1']:.3f}",
            judge_str,
            f"{avg_retries:.1f}",
            f"{avg_latency:.1f}",
            str(errors),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Question difficulty extraction (for mediation analysis)
# ---------------------------------------------------------------------------
def _extract_question_difficulty(item: dict) -> dict:
    """Extract question difficulty metadata from a dataset item.

    Extracts hop count, entity count, and question type for use
    as moderating variables in mediation analysis (RQ2 extension).
    """
    import re

    question = item.get("question", "")
    q_lower = question.lower()
    metadata = item.get("metadata", {})

    # Hop count: prefer dataset gold metadata, fallback to heuristic
    # MuSiQue: metadata.n_hops (2-4)
    # HotpotQA: always 2-hop by design
    # 2WikiMultiHopQA: infer from len(supporting_facts)
    hop_count = (
        metadata.get("n_hops")  # MuSiQue gold standard
        or item.get("hop_count")
        or item.get("num_hops")
    )
    if not hop_count:
        supporting_facts = metadata.get("supporting_facts", [])
        if supporting_facts:
            # Number of distinct supporting docs ≈ hop count
            titles = {sf.get("title", sf) for sf in supporting_facts if isinstance(sf, dict)}
            hop_count = max(2, len(titles))  # multi-hop datasets are ≥2
        else:
            # Last resort heuristic
            bridge_words = ["who", "where", "when", "which", "whose"]
            hop_count = max(1, sum(1 for w in bridge_words if w in q_lower))

    # Entity count: named entities (capitalized multi-word spans)
    entities = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", question)
    entity_count = len(set(entities))

    # Question type: prefer dataset metadata, fallback to keyword heuristic
    # HotpotQA/2Wiki: metadata.type = comparison/bridge/compositional/inference
    dataset_type = metadata.get("type", "")
    if dataset_type in ("comparison",):
        question_type = "comparison"
    elif dataset_type in ("bridge", "compositional", "inference"):
        question_type = "bridge"
    elif any(w in q_lower for w in ["how many", "how much", "how long"]):
        question_type = "numerical"
    elif any(w in q_lower for w in ["where", "what place", "born in"]):
        question_type = "location"
    elif any(w in q_lower for w in ["when", "what year", "what date"]):
        question_type = "temporal"
    else:
        question_type = "factoid"

    return {
        "hop_count": hop_count,
        "entity_count": entity_count,
        "question_type": question_type,
    }
