# Agentic Self-Corrective RAG — Project Instructions

## Project Overview
PhD 논문 실험 코드베이스: Agentic Self-Corrective RAG — Autonomous Retrieval Refinement via Tool-Augmented Language Models
- **Target Journal**: Knowledge-Based Systems (Elsevier, SCIE Q1, IF 7.6)
- **Base Paper**: CRAG (Yan et al., 2024)
- **Comparison**: Self-RAG (Asai et al., 2024), Adaptive-RAG (Jeong et al., 2024)

## Tech Stack
- **Python 3.11** (uv managed)
- **DSPy 3.1+** — 8 Signatures + AgenticRefinementSignature, ChainOfThought/Predict/ReAct modules
- **FAISS + BM25 + RRF** — Hybrid retrieval
- **OpenAI API** via litellm — gpt-4o-mini (preprocess/evaluate), gpt-4o (generate/agent)
- **PyYAML** — YAML config system for experiment management
- **Ruff** for lint+format, **pre-commit** hooks, **pytest** for tests

## Running
```bash
uv run pytest tests/                                     # unit tests
uv run ruff check .                                      # lint
uv run ruff format .                                     # format
uv run python scripts/prepare_datasets.py --sample 500   # download data

# Config-driven experiments
uv run python experiments/run.py --config configs/experiment/rq1.yaml --sample 20
uv run python experiments/run.py --ablation --sample 20
uv run python experiments/run.py --all --sample 20
```

## 4 Core Contributions (ReAct-centric)
- **C1**: Tool-Augmented Agentic Refinement — ReAct + 6 tools (core method)
- **C2**: Multi-dimensional Quality Assessment as Tool — 4D eval as agent tool
- **C3**: Structure-Aware Retrieval Tools — section_index + term_index
- **C4**: DSPy Declarative Pipeline + ReAct Integration

## 5 Research Questions
- **RQ1**: Agentic vs baselines (Naive/CRAG/Loop/Agentic) — answer quality
- **RQ2**: Tool usage analysis — trajectory patterns, tool call frequency
- **RQ3**: 4D eval as tool — Agent+4D vs Agent+1D vs Agent w/o eval
- **RQ4**: Structure-aware tools — full vs w/o section vs w/o term
- **RQ5**: DSPy optimization — Manual/Unopt/Bootstrap/MIPROv2

## Key Parameters
- QUALITY_THRESHOLD = 55
- MAX_RETRY = 3
- MAX_PASSAGES = 30
- top_k = 50, hybrid_weight = 0.48

## Architecture
```
agentic_rag/
  __init__.py
  config/
    __init__.py
    settings.py              — Pydantic settings with ablation flags + AgentSettings
    prompts.py               — System prompts
    loader.py                — YAML config loader (load_config, ExperimentConfig, VariantConfig)
  retriever/
    dense.py                 — FAISS dense retrieval
    sparse.py                — BM25 sparse retrieval
    hybrid.py                — RRF hybrid fusion
    indexer.py               — Index builder (+ section/term index)
    section_index.py         — Document section/TOC index (C3)
    term_index.py            — User term → doc terminology mapping (C3)
  signatures/
    preprocess.py            — Query rephrasing & keyword extraction
    decompose.py             — Multi-hop query decomposition
    evaluate.py              — 4D quality assessment
    generate.py              — Answer generation
    agents.py                — 3-way agent routing
    agent.py                 — ReAct agentic refinement signature (C1)
  tools/                     — Agent tools (1st class, C1/C2/C3)
    __init__.py              — TOOL_REGISTRY + create_tools(enabled_tools=)
    search.py                — Hybrid retrieval tool
    decompose.py             — Multi-hop query decomposition tool
    structure.py             — Document section browsing tool
    terminology.py           — Term mapping tool
    evaluate.py              — 4D quality assessment tool
    inspect.py               — Passage detail tool
  pipeline/
    base.py                  — BasePipeline, PipelineResult
    _mixin.py                — SelfCorrectiveMixin (shared preprocess/generate/route)
    agentic.py               — AgenticRAGPipeline (ReAct, PROPOSED METHOD)
    loop.py                  — LoopRAGPipeline (for-loop, ABLATION BASELINE)
    naive.py                 — NaiveRAGPipeline
    crag.py                  — CRAGReplicaPipeline
  evaluation/
    metrics.py               — EM, F1, ROUGE-L, BERTScore, Faithfulness
    cost_tracker.py          — API cost/latency tracking
    human_eval.py            — 5-dimension human evaluation protocol
  optimization/
    bootstrap.py             — BootstrapFewShot
    mipro.py                 — MIPROv2
    collector.py             — TrainingCollector
configs/
  base.yaml                  — Shared defaults (model, retrieval, evaluation, agent)
  pipeline/{naive,crag,loop,agentic}.yaml
  experiment/{rq1..rq5}.yaml — Per-RQ experiment configs with variants
  ablation/*.yaml            — 8 tool-level ablation configs
experiments/
  run.py                     — Unified config-driven runner (--config/--ablation/--all)
  common.py                  — Shared experiment utilities
  analysis/
    trajectory.py            — ReAct trajectory analysis (tool sequences, patterns)
    tool_usage.py            — Per-tool effectiveness metrics
    score_progression.py     — Quality score changes across iterations
    visualize.py             — Paper figures (matplotlib)
tests/
  test_config_loader.py      — Config loading + merging tests
  test_pipeline.py           — Pipeline structure + ablation flag tests
  test_retriever.py          — Sparse retriever + RRF tests
  test_agent.py              — ReAct agent, tools, section/term index, trajectory parsing tests
  test_signatures.py         — DSPy signature structure tests
```

## Ablation Flags
**Settings-level** (in settings.experiment):
enable_iteration, enable_accumulation, enable_4d_evaluation,
enable_refinement, enable_agent_routing, enable_dspy, enable_agentic_refinement

**Tool-level** (in settings.agent):
enabled_tools — list of tool names: search, decompose, structure, terminology, evaluate, inspect
(null = all tools enabled)

## Reference Documents
- Migration context: /Users/ruo/Downloads/CLAUDE_CODE_CONTEXT.md
- Notion analysis: https://www.notion.so/3239a5c548288134ac61c4fb289fa68e
- CRAG comparison: https://www.notion.so/3239a5c548288122acc7dd6e10b1a607
- Original system (Atelier workflow + posicube_rag.py): /Users/ruo/posicube/knowledge-base/example-scenarios/self-corrective-rag-dspy/
