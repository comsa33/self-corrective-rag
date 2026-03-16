# Refactoring Plan: Agentic Self-Corrective RAG

## Overview

Paper direction pivot: RLM-centric Agentic RAG. Incremental refactoring from old structure to industry-standard academic code release.

## Phase 1: Package Structure (DONE)

- [x] 1-1. `src/` -> `agentic_rag/` package rename
- [x] 1-2. `config/` -> `agentic_rag/config/` move
- [x] 1-3. `pyproject.toml` update (package name, build targets, isort)
- [x] 1-4. All import paths updated (`src.` / `config.` -> `agentic_rag.`)
- [x] 1-5. 63 tests pass, ruff clean

## Phase 2: Pipeline Separation (DONE)

- [x] 2-1. Split `self_corrective.py` -> `agentic.py` (RLM main) + `loop.py` (for-loop baseline)
  - `_mixin.py`: SelfCorrectiveMixin — shared preprocessing/generation/agent routing
  - `agentic.py`: AgenticRAGPipeline — RLM as primary refinement path
  - `loop.py`: LoopRAGPipeline — for-loop with C1-C5 (ablation baseline)
  - `self_corrective.py`: backward compat shim (SelfCorrectiveRAGPipeline = LoopRAGPipeline)
- [x] 2-2. Rename `naive_rag.py` -> `naive.py`, `crag_replica.py` -> `crag.py`
- [x] 2-3. Create `agentic_rag/tools/` directory — 5 individual tool modules + registry
  - `tools/__init__.py` — TOOL_REGISTRY + create_tools(enabled_tools=) factory
  - `tools/search.py`, `structure.py`, `terminology.py`, `evaluate.py`, `inspect.py`
  - `pipeline/rlm_tools.py` → backward compat shim delegating to tools/
- [x] 2-4. core/types.py — SKIPPED (Passage in indexer, PipelineResult in base works fine, moving risks circular imports)
- [x] 2-5. All imports updated, backward compat shims in place
- [x] 2-6. `__init__.py` exports updated (AgenticRAGPipeline, LoopRAGPipeline added)
- [x] 2-7. 68 tests pass (5 new: pipeline imports, class hierarchy, tool registry), ruff clean

## Phase 3: YAML Config System (DONE)

- [x] 3-1. Add `pyyaml` dependency to pyproject.toml
- [x] 3-2. Create `configs/` directory structure
  - `configs/base.yaml` — shared defaults (model, retrieval, evaluation, experiment, rlm)
  - `configs/pipeline/{agentic,loop,naive,crag}.yaml`
  - `configs/experiment/{rq1..rq5}.yaml` — each with named variants
  - `configs/ablation/{full,wo_iteration,wo_accumulation,1d_evaluation,wo_refinement,wo_agent,manual_prompt,rlm_refinement}.yaml`
- [x] 3-3. Config loader in `agentic_rag/config/loader.py`
  - `load_config()` — Load YAML, merge with base, apply CLI overrides
  - `apply_settings()` — Mutate global settings singleton from config dict
  - `load_experiment_config()` → `ExperimentConfig` with `VariantConfig` list
  - `load_ablation_configs()` → list of `VariantConfig` from ablation dir
  - `VariantConfig.import_pipeline_class()` — dynamic pipeline import
  - `PIPELINE_REGISTRY` — maps {naive, crag, loop, agentic} → class paths
- [x] 3-4. Unified experiment runner: `experiments/run.py`
  - `--config configs/experiment/rq1.yaml` — run single experiment
  - `--ablation` — run all ablation configs
  - `--all` — run all experiments + ablation
  - `--dataset`, `--sample`, `--skip`, `--variants` CLI args
- [x] 3-5. Experiment logic migrated to YAML configs (run_rq*.py retained as-is for backward compat)
- [x] 3-6. 88 tests pass (20 new: config loader), ruff clean

## Phase 4: RQ/Experiment Restructuring (DONE)

- [x] 4-1. New RQ configs (RLM-centric direction)
  - RQ1: agentic_vs_baselines (Naive/CRAG/Single-Pass/Loop/Agentic)
  - RQ2: tool_usage_analysis (trajectory analysis, tool call patterns)
  - RQ3: 4d_eval_as_tool (RLM+4D vs RLM+1D vs RLM w/o eval)
  - RQ4: structure_aware_tools (full vs w/o section vs w/o term vs w/o both)
  - RQ5: dspy_optimization (Manual/Unopt/Bootstrap/MIPROv2)
- [x] 4-2. New ablation variants config (tool-level ablation)
  - Full (RLM + all tools)
  - w/o RLM (for-loop fallback)
  - w/o evaluate tool
  - w/o section_index tool
  - w/o term_index tool
  - w/o search (no re-retrieval)
  - 1D evaluation (in eval tool)
  - Manual Prompt (no DSPy)
- [x] 4-3. Create `experiments/analysis/` directory
  - `trajectory.py` — TrajectoryAnalyzer (tool sequences, bigrams, to_dataframe)
  - `tool_usage.py` — ToolUsageAnalyzer (per-tool metrics, impact, question type grouping)
  - `score_progression.py` — ScoreProgressionAnalyzer (per-dimension, improvement stats)
  - `visualize.py` — Paper figures (metric comparison, ablation impact, tool frequency, score progression)
- [x] 4-4. Update CLAUDE.md with final architecture (4 contributions, 5 RQs, full file tree)
- [x] 4-5. Added `rlm.enabled_tools` to RLMSettings for tool-level ablation
- [x] 4-6. 91 tests pass (23 config loader tests updated for new RQ structure), ruff clean

## Current Status

**Last completed**: Phase 4 (2026-03-16)
**Status**: All phases complete — ready for experiments

## Architecture (Target)

```
agentic_rag/
  __init__.py
  core/
    types.py                 -- Passage, PipelineResult, EvalScore
  config/
    __init__.py
    settings.py              -- Pydantic settings
    prompts.py               -- System prompts
    loader.py                -- YAML config loader (Phase 3)
  retriever/
    dense.py, sparse.py, hybrid.py, indexer.py
    section_index.py, term_index.py
  signatures/
    preprocess.py, evaluate.py, generate.py, agents.py
    rlm_refinement.py
  tools/                     -- RLM tools (1st class)
    __init__.py              -- registry + create_tools()
    search.py, structure.py, terminology.py
    evaluate.py, inspect.py
  pipeline/
    base.py                  -- BasePipeline
    agentic.py               -- AgenticRAGPipeline (RLM, PROPOSED)
    loop.py                  -- LoopRAGPipeline (for-loop, BASELINE)
    naive.py                 -- NaiveRAGPipeline
    crag.py                  -- CRAGReplicaPipeline
  evaluation/
    metrics.py, cost_tracker.py, human_eval.py
  optimization/
    bootstrap.py, mipro.py, collector.py
configs/
  base.yaml
  pipeline/*.yaml
  experiment/*.yaml
  ablation/*.yaml
experiments/
  run.py                     -- unified config-driven runner
  run_ablation.py
  analysis/
    trajectory.py, tool_usage.py, score_progression.py, visualize.py
tests/
  test_tools/, test_pipeline/, test_retriever/, test_signatures/
```
