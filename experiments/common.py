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

    # Turn the response cache off before any LM is built. At temperature=0 a
    # rerun would otherwise replay cached completions in milliseconds, which
    # leaves accuracy intact but makes measured latency meaningless.
    if settings.disable_llm_cache:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        logger.info("LLM response cache disabled (latency measurement mode)")

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
    checkpoint_dir: Path | None = None,
    max_item_retries: int = 2,
    retry_backoff: float = 30.0,
) -> list[dict]:
    """Run a pipeline on a dataset and collect results.

    Resumption is keyed on question id rather than position, and only items
    that completed successfully are treated as done. An item that failed —
    typically a provider rate limit — is retried on the next run instead of
    being frozen into the results as a permanent gap.

    Args:
        request_delay: Seconds to wait between items (for API rate limiting).
        checkpoint_dir: If provided, writes results after every item to
            ``checkpoint_dir/<pipeline_name>_checkpoint.jsonl``. Rerunning the
            same command resumes from that file.
        max_item_retries: Extra attempts per item before recording an error.
        retry_backoff: Base seconds to wait before retrying an item; doubles
            with each attempt so a rate-limited run backs off rather than
            burning its remaining quota.

    Returns list of result dicts with predictions and metadata.
    """
    results: list[dict] = []
    checkpoint_path = None
    done_ids: set[str] = set()

    # Resume from checkpoint if exists
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{pipeline_name}_checkpoint.jsonl"

        if checkpoint_path.exists():
            n_retry = 0
            with open(checkpoint_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    # Failed items are dropped so this run retries them.
                    if "error" in record:
                        n_retry += 1
                        continue
                    results.append(record)
                    done_ids.add(str(record.get("id")))
            logger.info(
                f"[{pipeline_name}] Resumed from checkpoint: "
                f"{len(done_ids)}/{len(dataset)} done, {n_retry} to retry"
            )

    def _save_checkpoint() -> None:
        """Write the checkpoint atomically so an interrupt cannot truncate it."""
        if checkpoint_path is None:
            return
        tmp_path = checkpoint_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        tmp_path.replace(checkpoint_path)

    processed = 0
    for i, item in enumerate(dataset):
        item_id = str(item.get("id", i))
        if item_id in done_ids:
            continue
        if processed > 0 and request_delay > 0:
            time.sleep(request_delay)
        question = item["question"]
        reference = item.get("answer", "")

        record: dict = {}
        for attempt in range(max_item_retries + 1):
            start = time.perf_counter()
            try:
                result = pipeline.run(question)
                latency = time.perf_counter() - start
                record = {
                    "id": item_id,
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
                done_ids.add(item_id)
                break
            except Exception as e:
                if attempt < max_item_retries:
                    wait = retry_backoff * (2**attempt)
                    logger.warning(
                        f"[{pipeline_name}] Item {i} ({item_id}) failed "
                        f"(attempt {attempt + 1}/{max_item_retries + 1}): {e}. "
                        f"Retrying in {wait:.0f}s"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Error on item {i} ({item_id}): {e}")
                record = {
                    "id": item_id,
                    "question": question,
                    "reference": reference,
                    "prediction": "",
                    "error": str(e),
                    "pipeline": pipeline_name,
                }

        results.append(record)
        processed += 1
        _save_checkpoint()

        if processed % 10 == 0:
            logger.info(f"[{pipeline_name}] {len(done_ids)}/{len(dataset)} done")

    n_failed = sum(1 for r in results if "error" in r)
    if n_failed:
        logger.warning(
            f"[{pipeline_name}] {n_failed} item(s) still failing. "
            f"Rerun the same command to retry only those."
        )

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

    # Every field here has cost this project a day at some point, because the
    # run that produced a number could not be told apart from a run that would
    # produce a different one:
    #   max_passages  — only the accumulating pipelines apply it, so baselines
    #                   carried 50 passages against the proposed method's 30 in
    #                   every published run, and nothing on disk said so.
    #   enabled_tools — RQ1 ran four tools where RQ3/RQ4 ran all of them, which
    #                   is the real reason Table 3 and Table 11 disagree.
    #   max_tokens    — a limit too low for a reasoning model truncates answers
    #                   silently; one model lost a whole dataset to it.
    #   model names   — previously recoverable only from the directory name.
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
            "max_passages": settings.retrieval.max_passages,
            "query_method": settings.retrieval.query_method,
            "enabled_tools": settings.agent.enabled_tools,
            "agent_max_iterations": settings.agent.max_iterations,
            "max_tokens": settings.model.max_tokens,
            "temperature": settings.model.temperature,
            "models": {
                "preprocess": settings.model.preprocess_model,
                "evaluate": settings.model.evaluate_model,
                "generate": settings.model.generate_model,
                "agent": settings.model.agent_model,
            },
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
