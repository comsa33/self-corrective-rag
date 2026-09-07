"""Unified config-driven experiment runner.

Runs experiments defined by YAML config files. Supports experiment configs
(multiple variants), ablation configs (directory of variants), and
single pipeline configs.

Usage:
    # Run a single experiment
    uv run python experiments/run.py --config configs/experiment/rq1.yaml --dataset popqa --sample 100

    # Run ablation study
    uv run python experiments/run.py --ablation --dataset popqa --sample 100

    # Run all experiments + ablation
    uv run python experiments/run.py --all --dataset popqa --sample 20

    # Run specific ablation variants
    uv run python experiments/run.py --ablation --variants "Full System" "w/o Iteration"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console

from agentic_rag.config.loader import (
    VariantConfig,
    apply_settings,
    load_ablation_configs,
    load_config,
    load_experiment_config,
)
from agentic_rag.config.settings import settings


def _model_tag() -> str:
    """Extract a short model tag from the current generate model for result dirs.

    e.g. 'gemini/gemini-3.1-flash-lite-preview' -> 'gemini-3.1-flash-lite'
         'openai/gpt-5-mini' -> 'gpt-5-mini'
    """
    model = settings.model.generate_model
    # Strip provider prefix
    name = model.split("/")[-1] if "/" in model else model
    # Strip common suffixes
    for suffix in ("-preview", "-latest"):
        name = name.removesuffix(suffix)
    return name


def _plan(
    stem: str,
    dataset_name: str,
    n: int,
    repeat_index: int | None,
    resume: str | None,
    force: bool,
) -> RunPlan:
    """Run id, result directory and checkpoint directory for this run.

    See `experiments.manifest.plan_run`: checkpoints are keyed by run id so
    nothing resumes by accident, `resume` continues a named run, and a
    finished run with the same key blocks a fresh start unless forced.
    """
    return plan_run(
        settings.results_dir,
        stem=stem,
        dataset=dataset_name,
        model_tag=_model_tag(),
        n=n,
        repeat_index=repeat_index,
        resume=resume,
        force=force,
    )


from experiments.common import (
    get_meter,
    load_dataset,
    load_retriever,
    print_comparison_table,
    run_pipeline_on_dataset,
    save_results,
    settings_snapshot,
    setup_experiment,
)
from experiments.manifest import RunPlan, build_manifest, plan_run

console = Console()

EXPERIMENT_CONFIGS = [
    "configs/experiment/rq1.yaml",
    "configs/experiment/rq2.yaml",
    "configs/experiment/rq3.yaml",
    "configs/experiment/rq4.yaml",
    "configs/experiment/rq5.yaml",
]


# ---------------------------------------------------------------------------
# Variant execution
# ---------------------------------------------------------------------------
_SETTINGS_SECTIONS = ("model", "retrieval", "evaluation", "experiment", "agent")

# The settings as this process started (defaults + .env), before any variant
# touched them. apply_settings only writes the keys a config mentions, so a
# field that base.yaml does not list -- llm_call_budget, for one -- would
# otherwise carry over from the previous variant to the next. Every variant
# is applied on top of this baseline, never on top of its predecessor.
_BASELINE_SETTINGS = {name: getattr(settings, name).model_dump() for name in _SETTINGS_SECTIONS}


def _variant_settings(variant: VariantConfig) -> dict:
    """base.yaml merged with this variant's section overrides."""
    merged = load_config("configs/base.yaml").copy()
    for section, overrides in variant.overrides.items():
        if section in merged:
            merged[section] = {**merged.get(section, {}), **overrides}
        else:
            merged[section] = overrides
    return merged


def _apply_variant(variant: VariantConfig) -> None:
    """Put the globals in exactly the state this variant runs under."""
    apply_settings(_BASELINE_SETTINGS)
    apply_settings(_variant_settings(variant))


def _variant_controls(variants: list[VariantConfig]) -> dict[str, dict]:
    """Per variant: pipeline kind, passage cap and LLM-call budget under its settings.

    Recorded in the manifest before the first question so the controlled
    scope of the run (M, call budget) is on disk even if the run is cut
    short. The globals are restored afterwards.
    """
    before = {name: getattr(settings, name).model_dump() for name in _SETTINGS_SECTIONS}
    controls: dict[str, dict] = {}
    try:
        for variant in variants:
            _apply_variant(variant)
            controls[variant.name] = {
                "pipeline": variant.pipeline,
                "passage_cap": variant.import_pipeline_class().passage_cap(),
                "llm_call_budget": settings.experiment.llm_call_budget,
            }
    finally:
        apply_settings(before)  # leave the globals as they were
    return controls


def _variant_passage_caps(variants: list[VariantConfig]) -> dict[str, int | None]:
    return {name: c["passage_cap"] for name, c in _variant_controls(variants).items()}


def _manifest_controls(variants: list[VariantConfig]) -> dict:
    """Keyword arguments for build_manifest describing every variant's controls."""
    controls = _variant_controls(variants)
    return {
        "max_passages_by_pipeline": {n: c["passage_cap"] for n, c in controls.items()},
        "llm_call_budget_by_pipeline": {n: c["llm_call_budget"] for n, c in controls.items()},
        "pipeline_by_variant": {n: c["pipeline"] for n, c in controls.items()},
    }


def _run_variant(
    variant: VariantConfig,
    dataset: list[dict],
    retriever,
    indexer,
    request_delay: float = 0.0,
    trainset: list | None = None,
    checkpoint_dir: Path | None = None,
    run_id: str | None = None,
    repeat_index: int | None = None,
) -> list[dict]:
    """Run a single variant: apply settings, create pipeline, execute.

    If variant.optimization is set ('bootstrap' or 'mipro'), applies the
    optimizer using pre-collected trainset before evaluating on dataset.
    """
    # Apply variant-specific settings on top of the process baseline
    _apply_variant(variant)

    # Import and create pipeline
    pipeline_cls = variant.import_pipeline_class()
    pipeline = pipeline_cls(retriever, indexer)

    # --- DSPy optimization (RQ5) ---
    if variant.optimization and trainset:
        _apply_optimization(variant.optimization, pipeline, trainset)

    # Snapshot while this variant's settings are in force. save_results runs
    # after every variant has finished, so reading the globals there would
    # record the last variant's settings for all of them.
    used = settings_snapshot()
    used["passage_cap"] = pipeline_cls.passage_cap()
    if hasattr(pipeline_cls, "effective_max_iters"):
        used["react_max_iters"] = pipeline_cls.effective_max_iters()

    slug = variant.name.lower().replace(" ", "_").replace("/", "_")
    results = run_pipeline_on_dataset(
        pipeline,
        dataset,
        slug,
        request_delay=request_delay,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        repeat_index=repeat_index,
    )
    return results, used


def _collect_training_data(
    pipeline,
    train_data: list[dict],
    dataset_name: str = "",
) -> list:
    """Collect training examples by running pipeline on train_data.

    Returns a list of dspy.Example objects for optimization.
    This is separated from _apply_optimization so the collection
    can be done once and shared across multiple optimization variants.

    Training data is cached to disk so subsequent runs skip collection.
    Cache path: data/optimization/{dataset_name}_trainset.json
    """
    from agentic_rag.optimization.collector import TrainingCollector

    # Check for cached trainset
    cache_dir = settings.data_dir / "optimization"
    cache_path = cache_dir / f"{dataset_name}_trainset.json" if dataset_name else None

    if cache_path and cache_path.exists():
        collector = TrainingCollector()
        collector.load(cache_path)
        trainset = collector.to_dspy_examples("GenerateSignature")
        logger.info(
            f"[Optimization] Loaded cached trainset: {len(trainset)} examples from {cache_path}"
        )
        return trainset

    # Collect by running pipeline
    logger.info(f"[Optimization] Collecting training data: {len(train_data)} examples")

    collector = TrainingCollector()
    for item in train_data:
        try:
            result = pipeline.run(item["question"])
            collector.add(
                "GenerateSignature",
                inputs={"question": item["question"], "context": result.answer[:500]},
                outputs={"answer": item.get("answer", "")},
            )
        except Exception as e:
            logger.warning(f"[Optimization] Training example failed: {e}")

    if collector.total_count < 3:
        logger.warning(
            f"[Optimization] Only {collector.total_count} examples collected, "
            f"need >= 3 for optimization"
        )
        return []

    # Save to disk for reuse
    if cache_path:
        collector.save(cache_path)
        logger.info(f"[Optimization] Trainset cached to {cache_path}")

    return collector.to_dspy_examples("GenerateSignature")


def _apply_optimization(
    optimization: str,
    pipeline,
    trainset: list,
) -> None:
    """Apply DSPy optimization (bootstrap/mipro) using pre-collected trainset.

    Trainset should be collected once via _collect_training_data() and
    shared across optimization variants to avoid redundant pipeline runs.
    """
    from agentic_rag.evaluation.metrics import token_f1

    if not trainset:
        logger.warning("[Optimization] Empty trainset, skipping optimization")
        return

    logger.info(f"[Optimization] Applying {optimization} with {len(trainset)} examples")

    def optimization_metric(example, prediction, trace=None) -> float:
        pred = getattr(prediction, "answer", "")
        ref = getattr(example, "answer", "")
        return token_f1(pred, ref)

    target_module = _find_dspy_module(pipeline)
    if target_module is None:
        logger.warning("[Optimization] No optimizable DSPy module found in pipeline")
        return

    if optimization == "bootstrap":
        from agentic_rag.optimization.bootstrap import optimize_bootstrap

        optimized = optimize_bootstrap(
            target_module, trainset, metric_fn=optimization_metric, max_demos=4
        )
        _replace_dspy_module(pipeline, optimized)
        logger.info("[Optimization] BootstrapFewShot optimization applied")

    elif optimization == "mipro":
        from agentic_rag.optimization.mipro import optimize_mipro

        optimized = optimize_mipro(target_module, trainset, metric_fn=optimization_metric)
        _replace_dspy_module(pipeline, optimized)
        logger.info("[Optimization] MIPROv2 optimization applied")

    else:
        logger.warning(f"[Optimization] Unknown optimization strategy: {optimization}")


def _find_dspy_module(pipeline):
    """Find the primary optimizable DSPy module in a pipeline."""
    import dspy

    # Check common attribute names for the generate module
    for attr_name in ("generator", "generate", "generate_module"):
        mod = getattr(pipeline, attr_name, None)
        if mod is not None and isinstance(mod, dspy.Module):
            return mod

    # Search all attributes for DSPy modules
    for attr_name in dir(pipeline):
        if attr_name.startswith("_"):
            continue
        mod = getattr(pipeline, attr_name, None)
        if isinstance(mod, dspy.Module):
            return mod

    return None


def _replace_dspy_module(pipeline, optimized_module):
    """Replace the pipeline's DSPy module with the optimized version."""
    import dspy

    for attr_name in ("generator", "generate", "generate_module"):
        if hasattr(pipeline, attr_name) and isinstance(getattr(pipeline, attr_name), dspy.Module):
            setattr(pipeline, attr_name, optimized_module)
            return

    # Fallback: replace first DSPy module found
    for attr_name in dir(pipeline):
        if attr_name.startswith("_"):
            continue
        if isinstance(getattr(pipeline, attr_name, None), dspy.Module):
            setattr(pipeline, attr_name, optimized_module)
            return


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    config_path: str,
    dataset_name: str = "popqa",
    sample_size: int | None = None,
    request_delay: float = 0.0,
    compute_llm_judge: bool = False,
    variant_names: list[str] | None = None,
    *,
    repeat_index: int | None = None,
    resume: str | None = None,
    force: bool = False,
) -> dict[str, list[dict]]:
    """Run an experiment defined by a YAML config file.

    Args:
        variant_names: Run only these variants. Pipelines differ by an order
            of magnitude in LLM calls per question, so splitting the expensive
            ones into their own process lets a run finish in the time of its
            slowest pipeline rather than their sum.
        repeat_index: k of a repeated run; goes into the run id, manifest
            and every row, and makes the run distinct from the k-less one.
        resume: run id of an interrupted run to continue under the same id.
        force: start even though a finished run with the same key exists.
    """
    exp = load_experiment_config(config_path)

    if variant_names:
        available = [v.name for v in exp.variants]
        unknown = set(variant_names) - set(available)
        if unknown:
            raise ValueError(f"Unknown variant(s) {sorted(unknown)}. Available: {available}")
        exp.variants = [v for v in exp.variants if v.name in variant_names]

    logger.info(f"Running experiment: {exp.name}")
    logger.info(f"  Description: {exp.description}")
    logger.info(f"  Variants: {len(exp.variants)}")

    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever(dataset_name=dataset_name)

    # Handle train/val split for optimization experiments (e.g., RQ5)
    test_data = dataset
    if exp.train_size is not None:
        train_end = exp.train_size
        val_end = train_end + (exp.val_size or 0)
        test_data = dataset[val_end:]
        if not test_data:
            raise ValueError(
                f"No test data remaining after train/val split "
                f"(dataset={len(dataset)}, train={train_end}, val={exp.val_size or 0}). "
                f"Increase --sample or reduce train_size/val_size in config."
            )
        logger.info(
            f"  Data split: train={train_end}, val={exp.val_size or 0}, test={len(test_data)}"
        )

    # Prepare train split and collect training data once for optimization variants
    trainset = None
    has_optimization = any(v.optimization for v in exp.variants)
    if exp.train_size is not None and has_optimization:
        train_data = dataset[: exp.train_size]
        # Collect training data once using DSPy Unoptimized pipeline (baseline)
        # This avoids redundant pipeline runs for Bootstrap + MIPROv2
        base_cfg = load_config("configs/base.yaml")
        apply_settings(base_cfg)
        from agentic_rag.pipeline.agentic import AgenticRAGPipeline

        collector_pipeline = AgenticRAGPipeline(retriever, indexer)
        trainset = _collect_training_data(collector_pipeline, train_data, dataset_name)
        logger.info(
            f"  Training data collected: {len(trainset)} examples (shared across optimizers)"
        )

    all_results: dict[str, list[dict]] = {}
    used_settings: dict[str, dict] = {}
    config_stem = Path(config_path).stem

    # The run directory exists before the first question is asked, so the
    # manifest can be written and the model snapshot checked up front. A run
    # that would produce numbers under the wrong snapshot stops here.
    plan = _plan(config_stem, dataset_name, len(test_data), repeat_index, resume, force)
    run_dir = plan.run_dir
    manifest = build_manifest(
        run_id=plan.run_id,
        experiment=exp.name,
        dataset=dataset_name,
        n=len(test_data),
        variants=[v.name for v in exp.variants],
        config_path=config_path,
        sample_size=sample_size,
        run_key=plan.run_key,
        repeat_index=plan.repeat_index,
        attempt=plan.attempt,
        **_manifest_controls(exp.variants),
    )
    manifest.run_preflight(run_dir, get_meter())

    for variant in exp.variants:
        logger.info(f"  Running variant: {variant.name}")
        slug = variant.name.lower().replace(" ", "_").replace("/", "_")
        results, used = _run_variant(
            variant,
            test_data,
            retriever,
            indexer,
            request_delay,
            trainset=trainset if variant.optimization else None,
            checkpoint_dir=plan.checkpoint_dir / slug,
            run_id=plan.run_id,
            repeat_index=plan.repeat_index,
        )
        all_results[variant.name] = results
        used_settings[variant.name] = used

    # Report & save — all variants share one run directory
    print_comparison_table(
        all_results,
        title=f"{exp.name} ({dataset_name})",
        compute_llm_judge=compute_llm_judge,
    )
    for name, results in all_results.items():
        slug = name.lower().replace(" ", "_").replace("/", "_")
        save_results(
            results,
            f"{config_stem}_{slug}",
            {"experiment": exp.name, "dataset": dataset_name, "variant": name},
            run_dir=run_dir,
            compute_llm_judge=compute_llm_judge,
            settings_used=used_settings.get(name),
        )
    manifest.finish_run(run_dir, get_meter(), all_results)

    return all_results


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
def run_ablation(
    dataset_name: str = "popqa",
    sample_size: int | None = None,
    variant_names: list[str] | None = None,
    request_delay: float = 0.0,
    compute_llm_judge: bool = False,
    *,
    repeat_index: int | None = None,
    resume: str | None = None,
    force: bool = False,
) -> dict[str, list[dict]]:
    """Run ablation study from configs/ablation/ directory."""
    variants = load_ablation_configs()

    if variant_names:
        available = [v.name for v in variants]
        unknown = set(variant_names) - set(available)
        if unknown:
            raise ValueError(f"Unknown variant(s) {sorted(unknown)}. Available: {available}")
        variants = [v for v in variants if v.name in variant_names]

    logger.info(f"Running ablation study: {len(variants)} variants")

    setup_experiment()
    dataset = load_dataset(dataset_name, sample_size)
    retriever, indexer = load_retriever(dataset_name=dataset_name)

    all_results: dict[str, list[dict]] = {}
    used_settings: dict[str, dict] = {}
    plan = _plan("ablation", dataset_name, len(dataset), repeat_index, resume, force)
    run_dir = plan.run_dir
    manifest = build_manifest(
        run_id=plan.run_id,
        experiment="ablation",
        dataset=dataset_name,
        n=len(dataset),
        variants=[v.name for v in variants],
        sample_size=sample_size,
        run_key=plan.run_key,
        repeat_index=plan.repeat_index,
        attempt=plan.attempt,
        **_manifest_controls(variants),
    )
    manifest.run_preflight(run_dir, get_meter())

    for variant in variants:
        logger.info(f"  Running ablation variant: {variant.name}")
        slug = variant.name.lower().replace(" ", "_").replace("/", "_")
        results, used = _run_variant(
            variant,
            dataset,
            retriever,
            indexer,
            request_delay,
            checkpoint_dir=plan.checkpoint_dir / slug,
            run_id=plan.run_id,
            repeat_index=plan.repeat_index,
        )
        all_results[variant.name] = results
        used_settings[variant.name] = used

    print_comparison_table(
        all_results,
        title=f"Ablation Study ({dataset_name})",
        compute_llm_judge=compute_llm_judge,
    )
    for name, results in all_results.items():
        slug = name.lower().replace(" ", "_").replace("/", "_")
        save_results(
            results,
            f"ablation_{slug}",
            {"experiment": "ablation", "dataset": dataset_name, "variant": name},
            run_dir=run_dir,
            compute_llm_judge=compute_llm_judge,
            settings_used=used_settings.get(name),
        )
    manifest.finish_run(run_dir, get_meter(), all_results)

    return all_results


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
def run_all(
    dataset_name: str = "popqa",
    sample_size: int | None = None,
    skip: list[str] | None = None,
    request_delay: float = 0.0,
    compute_llm_judge: bool = False,
    *,
    repeat_index: int | None = None,
    force: bool = False,
) -> None:
    """Run all experiments + ablation study."""
    skip = skip or []

    for config_path in EXPERIMENT_CONFIGS:
        exp_name = Path(config_path).stem  # e.g., "rq1"
        if exp_name in skip:
            logger.info(f"Skipping {exp_name}")
            continue
        try:
            run_experiment(
                config_path,
                dataset_name,
                sample_size,
                request_delay,
                compute_llm_judge,
                repeat_index=repeat_index,
                force=force,
            )
        except Exception as e:
            logger.error(f"{exp_name} failed: {e}")

    if "ablation" not in skip:
        try:
            run_ablation(
                dataset_name,
                sample_size,
                request_delay=request_delay,
                compute_llm_judge=compute_llm_judge,
                repeat_index=repeat_index,
                force=force,
            )
        except Exception as e:
            logger.error(f"Ablation failed: {e}")

    console.print("\n[bold green]All experiments complete![/]")
    console.print(f"Results saved to: {settings.results_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified config-driven experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  uv run python experiments/run.py --config configs/experiment/rq1.yaml --sample 100
  uv run python experiments/run.py --ablation --sample 50
  uv run python experiments/run.py --all --dataset popqa --sample 20
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study from configs/ablation/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all experiments + ablation",
    )
    parser.add_argument("--dataset", default="popqa")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=None,
        help="Experiments to skip when using --all (e.g., rq4 rq5 ablation)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Run only these variant names (works with --config and --ablation)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between items (for API rate limiting, e.g. 5.0)",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Compute LLM-as-Judge correctness metric (uses evaluate_model)",
    )
    parser.add_argument(
        "--repeat-index",
        type=int,
        default=None,
        metavar="K",
        help="Repetition number of this run; recorded in run id, manifest and rows",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help="Continue an interrupted run under the same run id (checkpoints are reused)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Start even though a finished run with the same key already exists",
    )

    args = parser.parse_args()

    if args.run_all:
        run_all(
            args.dataset,
            args.sample,
            args.skip,
            args.delay,
            args.llm_judge,
            repeat_index=args.repeat_index,
            force=args.force,
        )
    elif args.ablation:
        run_ablation(
            args.dataset,
            args.sample,
            args.variants,
            args.delay,
            args.llm_judge,
            repeat_index=args.repeat_index,
            resume=args.resume,
            force=args.force,
        )
    elif args.config:
        run_experiment(
            args.config,
            args.dataset,
            args.sample,
            args.delay,
            args.llm_judge,
            args.variants,
            repeat_index=args.repeat_index,
            resume=args.resume,
            force=args.force,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
