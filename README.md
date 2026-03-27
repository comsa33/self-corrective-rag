# TARA: Complexity-Adaptive Retrieval Refinement through Tool-Augmented ReAct Agents

[English](#overview) | [한국어](README_ko.md)

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![DSPy](https://img.shields.io/badge/DSPy-3.1+-green.svg)](https://github.com/stanfordnlp/dspy)

---

## Overview

**TARA** replaces fixed-loop self-corrective RAG with a **ReAct-based agent** equipped with six specialized tools for autonomous retrieval refinement. The entire pipeline is implemented as a **DSPy declarative program**, enabling automatic prompt optimization.

> **Target Journal**: Knowledge-Based Systems (Elsevier, SCIE Q1, IF 7.6)

### Key Contributions

1. **Tool-Augmented Agentic Refinement** — ReAct agent with 4 core + 2 domain-adaptive tools autonomously decides retrieval strategy
2. **Multi-dimensional Quality Assessment** — 4D evaluation (Relevance, Coverage, Specificity, Sufficiency) as an agent tool
3. **Structure-Aware Retrieval** — Document section browsing and terminology mapping for enterprise documents
4. **DSPy Declarative Pipeline** — Typed Signatures + BootstrapFewShot/MIPROv2 optimization

### Main Results

| Dataset | TARA (F1) | Loop (F1) | Delta | p-value |
|---------|-----------|-----------|-------|---------|
| **2WikiMultiHopQA** | **.584** | .495 | **+.089** | **<.001** |
| MuSiQue | **.438** | .399 | +.039 | .161 |
| HotpotQA | **.658** | .636 | +.022 | .772 |
| FinanceBench | .386 | **.400** | -.014 | .394 |

*Gemini Flash Lite, n=200, paired bootstrap significance test (Bonferroni-corrected)*

**Key finding**: The agentic advantage is **complexity-dependent** — substantial on 4-hop questions (+0.305 F1) and diminishes on simpler tasks.

---

## Architecture

```
                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Preprocessing  │  Query rephrasing + keyword extraction
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   3-Way Router  │  → Naive / Single-Pass / Agent
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │     ReAct Agent (DSPy)      │
              │                             │
              │  Thought → Action → Observe │
              │         (loop)              │
              │                             │
              │  Tools:                     │
              │  ├─ search_passages     ◄── Core
              │  ├─ decompose_query     ◄──
              │  ├─ evaluate_passages   ◄──
              │  ├─ get_passage_detail  ◄──
              │  ├─ list_document_sections  ◄── Domain-adaptive
              │  └─ get_terminology         ◄──
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │    Generate     │  Final answer with citations
                    └─────────────────┘
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/comsa33/self-corrective-rag.git
cd self-corrective-rag
cp .env.example .env    # Edit with your API keys
uv sync                 # Install dependencies
```

### 2. Configure `.env`

```env
GEMINI_API_KEY=your-key-here
PREPROCESS_MODEL=gemini/gemini-3.1-flash-lite-preview
EVALUATE_MODEL=gemini/gemini-3.1-flash-lite-preview
GENERATE_MODEL=gemini/gemini-3.1-flash-lite-preview
AGENT_MODEL=gemini/gemini-3.1-flash-lite-preview
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3. Prepare Data

```bash
uv run python scripts/prepare_datasets.py --sample 500
uv run python scripts/build_index.py --dataset all
```

### 4. Run Experiments

```bash
# Single RQ
uv run python experiments/run.py --config configs/experiment/rq1.yaml --sample 20

# All experiments
uv run python experiments/run.py --all --sample 200 --delay 0
```

---

## Repository Structure

```
agentic_rag/
  config/          Settings, prompts, YAML config loader
  retriever/       FAISS + BM25 hybrid retrieval, section/term indices
  signatures/      DSPy signatures (preprocess, evaluate, generate, agent)
  tools/           6 agent tools (search, decompose, evaluate, inspect, structure, terminology)
  pipeline/        Pipeline implementations (naive, crag, loop, agentic)
  evaluation/      Metrics (EM, F1, ROUGE-L, BERTScore), cost tracker
  optimization/    BootstrapFewShot, MIPROv2 wrappers

configs/
  base.yaml                  Shared defaults
  pipeline/*.yaml            Per-pipeline configs
  experiment/rq1..rq5.yaml   Per-RQ experiment configs
  ablation/*.yaml            8 tool-level ablation configs

experiments/
  run.py           Unified config-driven experiment runner
  common.py        Shared utilities
  analysis/        Trajectory, tool usage, score progression analysis

paper/
  main.tex                   Paper source (Elsevier elsarticle)
  sections/                  Per-section .tex files
  references.bib             77 references
  supplementary/             12 CSV files for reviewer verification

tests/                       pytest test suite
```

---

## Research Questions

| RQ | Question | Finding |
|----|----------|---------|
| **RQ1** | Agentic vs baselines? | +0.089 F1 on 2Wiki (p<.001), complexity-dependent |
| **RQ2** | Tool usage patterns? | decompose→search→evaluate convergence, 5-6 tools/question |
| **RQ3** | 4D vs 1D evaluation? | 1D ≈ 4D > w/o Eval — evaluation as quality gate |
| **RQ4** | Structure-aware tools? | Dataset-dependent: helpful on MuSiQue/FinanceBench, not Wikipedia |
| **RQ5** | DSPy optimization? | +0.071–0.164 F1 from Signatures, Bootstrap best on all datasets |

---

## Supplementary Materials

Pre-computed results for reviewer verification are in [`paper/supplementary/`](paper/supplementary/):

- Bootstrap CI and pairwise significance tests
- Refusal rate analysis across models
- Hop-level F1 breakdown
- 2×2 factorial synergy analysis
- LLM-as-Judge results

---

## Citation

```bibtex
@article{lee2026tara,
  title={TARA: Complexity-Adaptive Retrieval Refinement through Tool-Augmented ReAct Agents},
  author={Lee, Ruo},
  journal={Knowledge-Based Systems},
  year={2026},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
