# Supplementary Materials — TARA

Structured experiment results for reviewer verification.
All results generated from code at `experiments/run.py` with fixed seed=42 and temperature=0.

## Files

| File | Description | Rows |
|------|-------------|------|
| `rq1_main_results.csv` | RQ1: All pipelines × 4 datasets × 2 models (EM, F1, ROUGE-L) | 40 |
| `rq2_tool_usage.csv` | RQ2: Tool usage analysis per dataset | 4 |
| `rq3_evaluation_ablation.csv` | RQ3: 4D vs 1D vs w/o Eval (3 variants × 4 datasets) | 12 |
| `rq4_structure_ablation.csv` | RQ4: Structure tool ablation (4 variants × 4 datasets) | 16 |
| `rq5_dspy_optimization.csv` | RQ5: Manual/Unopt/Bootstrap/MIPROv2 × 4 datasets | 16 |
| `ablation_full.csv` | Consolidated ablation (all 8 variants × 4 datasets + bugfixed reruns) | 36 |
| `bootstrap_ci_pipelines.csv` | 95% Bootstrap CI for each pipeline × dataset × model | 40 |
| `pairwise_significance.csv` | Pairwise significance tests: Δ, p-value, Cohen's d, Bonferroni | 32 |
| `refusal_rates.csv` | Answer refusal rates per pipeline × dataset × model | 40 |
| `hop_level_breakdown.csv` | F1 by hop count (2-hop vs 4-hop) per pipeline | 30 |
| `synergy_2x2_factorial.csv` | 2×2 factorial: Agent/Loop × DSPy/Manual, interaction term | 4 |
| `llm_judge_results.csv` | LLM-as-Judge (gpt-4.1-nano) accuracy per pipeline × dataset | 40 |

## Experiment Settings

- **Primary model**: Gemini 3.1 Flash Lite Preview (`gemini/gemini-3.1-flash-lite-preview`)
- **Cross-model**: gpt-5-mini (`gpt-5-mini`, reasoning_effort=low)
- **LLM-as-Judge**: gpt-4.1-nano (independent evaluator)
- **Embedding**: all-MiniLM-L6-v2 (sentence-transformers)
- **Sample size**: n=200 per dataset (FinanceBench: n=150, full dataset)
- **Quality threshold (τ)**: 40
- **Max retry**: 3
- **Hybrid retrieval**: FAISS + BM25, RRF fusion (weight=0.48, top_k=50)
- **Seed**: 42
- **Temperature**: 0

## Reproducibility

```bash
# Install dependencies
uv sync

# Prepare data + indices
uv run python scripts/prepare_datasets.py --sample 500
uv run python scripts/build_index.py --dataset all

# Run experiments
uv run python experiments/run.py --config configs/experiment/rq1.yaml --sample 200
uv run python experiments/run.py --all --sample 200
```

## Per-Question Raw Data

Full per-question predictions, trajectories, and evaluation scores are available in JSONL format.
Each JSONL record contains:
- `question`, `prediction`, `reference` (text)
- `action_history` (tool call sequence)
- `evaluation_scores` (4D quality scores per iteration)
- `tool_score_trace` (score progression)
- `latency_seconds`, `llm_calls`, `passages_used`
- `question_difficulty` (hop_count, entity_count, question_type)

Contact the authors for access to full JSONL files (~50MB total).
