"""Run LLM-as-Judge on experiment result JSONL files.

Reads existing JSONL files, calls LLM-as-Judge for each prediction,
and saves results to a new file with _judged suffix.

Supports checkpoint/resume: skips already-judged items.

Usage:
    # Single directory (gpt-4.1-nano 추천 — 최저가, binary judge에 충분)
    uv run python scripts/run_llm_judge.py data/results/20260324_171259_rq1_hotpotqa_n200_gemini-3.1-flash-lite/ --judge-model openai/gpt-4.1-nano

    # All RQ1 directories
    uv run python scripts/run_llm_judge.py --pattern "rq1" --judge-model openai/gpt-4.1-nano

    # Specific JSONL file only
    uv run python scripts/run_llm_judge.py data/results/.../rq1_agentic_(react).jsonl --judge-model openai/gpt-4.1-nano
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from agentic_rag.evaluation.metrics import llm_judge_correctness
from experiments.common import setup_experiment


def judge_jsonl(jsonl_path: Path, delay: float = 0.0, judge_model: str | None = None) -> dict:
    """Run LLM-as-Judge on a single JSONL file.

    Returns summary dict with judge accuracy.
    """
    output_path = jsonl_path.with_name(jsonl_path.stem + "_judged.jsonl")

    # Load existing results
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))

    # Load checkpoint (already judged)
    judged = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                judged[item["id"]] = item

    logger.info(
        f"[{jsonl_path.name}] {len(items)} items, "
        f"{len(judged)} already judged, "
        f"{len(items) - len(judged)} remaining"
    )

    if len(judged) >= len(items):
        logger.info(f"[{jsonl_path.name}] Already complete, skipping")
        scores = [judged[item["id"]]["llm_judge"] for item in items if item["id"] in judged]
        return {
            "file": jsonl_path.name,
            "n": len(items),
            "judge_accuracy": sum(scores) / len(scores) if scores else 0.0,
            "status": "skipped",
        }

    # Open output in append mode
    scores = []
    with open(output_path, "a", encoding="utf-8") as out:
        for i, item in enumerate(items):
            if item["id"] in judged:
                scores.append(judged[item["id"]]["llm_judge"])
                continue

            prediction = item.get("prediction", "")
            reference = item.get("reference", "")
            question = item.get("question", "")

            try:
                score = llm_judge_correctness(
                    prediction, reference, question, judge_model=judge_model
                )
            except Exception as e:
                logger.warning(f"  [{i + 1}/{len(items)}] Error: {e}")
                score = 0.0

            scores.append(score)

            # Write judged item
            judged_item = {
                "id": item["id"],
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "llm_judge": score,
                "pipeline": item.get("pipeline", ""),
            }
            out.write(json.dumps(judged_item, ensure_ascii=False) + "\n")
            out.flush()

            if (i + 1) % 20 == 0:
                running_acc = sum(scores) / len(scores)
                logger.info(f"  [{i + 1}/{len(items)}] running judge_acc={running_acc:.3f}")

            if delay > 0:
                time.sleep(delay)

    accuracy = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"[{jsonl_path.name}] Done: judge_accuracy={accuracy:.3f}")

    return {
        "file": jsonl_path.name,
        "n": len(items),
        "judge_accuracy": accuracy,
        "status": "complete",
    }


def find_jsonl_files(base_dir: Path, pattern: str | None = None) -> list[Path]:
    """Find all JSONL files to process."""
    results_dir = base_dir / "data" / "results"
    jsonl_files = []

    dirs = sorted(results_dir.glob("202603*/"))
    if pattern:
        dirs = [d for d in dirs if any(d.name.find(p) >= 0 for p in pattern.split(","))]

    for d in dirs:
        for f in sorted(d.glob("*.jsonl")):
            # Skip already-judged files and checkpoint files
            if "_judged" in f.name or "checkpoint" in f.name:
                continue
            jsonl_files.append(f)

    return jsonl_files


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge on result JSONL files")
    parser.add_argument(
        "path",
        nargs="?",
        help="Result directory or specific JSONL file",
    )
    parser.add_argument(
        "--pattern",
        help="Filter directories by pattern (e.g., 'rq1', 'rq1_hotpotqa')",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between LLM calls in seconds (rate limiting)",
    )
    parser.add_argument(
        "--judge-model",
        default="openai/gpt-4.1-nano",
        help="Model to use for judging (default: openai/gpt-4.1-nano)",
    )
    args = parser.parse_args()

    setup_experiment()

    project_root = Path(__file__).resolve().parent.parent

    if args.path:
        path = Path(args.path)
        if path.is_file() and path.suffix == ".jsonl":
            files = [path]
        elif path.is_dir():
            files = sorted(
                f
                for f in path.glob("*.jsonl")
                if "_judged" not in f.name and "checkpoint" not in f.name
            )
        else:
            logger.error(f"Invalid path: {path}")
            sys.exit(1)
    else:
        files = find_jsonl_files(project_root, pattern=args.pattern)

    if not files:
        logger.error("No JSONL files found")
        sys.exit(1)

    logger.info(f"Found {len(files)} JSONL files to judge")
    logger.info(f"Judge model: {args.judge_model}")
    total_items = 0
    for f in files:
        with open(f) as fh:
            n = sum(1 for _ in fh)
        total_items += n
    logger.info(f"Total predictions: {total_items}")

    results = []
    for f in files:
        try:
            r = judge_jsonl(f, delay=args.delay, judge_model=args.judge_model)
            results.append(r)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")
            results.append({"file": f.name, "status": "error", "error": str(e)})

    # Print summary
    print("\n" + "=" * 60)
    print("LLM-as-Judge Summary")
    print("=" * 60)
    for r in results:
        status = r.get("status", "unknown")
        acc = r.get("judge_accuracy", 0)
        n = r.get("n", 0)
        print(f"  {r['file']:50s} n={n:4d} acc={acc:.3f} [{status}]")

    # Save summary
    if args.path:
        summary_dir = Path(args.path) if Path(args.path).is_dir() else Path(args.path).parent
    else:
        summary_dir = project_root / "data" / "results"
    summary_path = summary_dir / "llm_judge_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
