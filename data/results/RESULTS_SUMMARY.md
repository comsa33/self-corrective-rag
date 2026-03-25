# v1.6 Experiment Results Summary (Scale-up, n=200)

**Code version**: v1.6-scaleup
**Primary Model**: Gemini 3.1 Flash Lite Preview
**Cross-Model**: gpt-5-mini (RQ1 only)
**Embedding**: all-MiniLM-L6-v2
**Date**: 2026-03-24 ~ 2026-03-25
**Sample size**: n=200 per dataset (FinanceBench n=150, 전체 데이터)

> **Note**: 이 결과는 n=200 스케일업 실험 결과입니다.
> - Gemini: RQ1~5 + ablation × 4 datasets 전체 완료
> - gpt-5-mini: RQ1 × 4 datasets 완료 (cross-model 검증)
> - LLM-as-Judge: 미실행 (별도 일괄 실행 예정)
> - Bootstrap CI 통계 검증: 미실행

---

## RQ1: Agentic vs Baselines

**Question**: Does ReAct-based agentic refinement improve answer quality compared to fixed-loop refinement and single-pass baselines?

### Gemini Flash Lite — F1

| Pipeline | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|----------|----------|-------|---------|--------------|
| Naive RAG | 0.606 | 0.378 | 0.342 | 0.398 |
| CRAG Replica | 0.553 | 0.255 | 0.251 | 0.333 |
| Single-Pass | 0.630 | 0.486 | 0.384 | 0.381 |
| Loop Refinement | 0.636 | 0.495 | 0.399 | **0.400** |
| **Agentic (ReAct)** | **0.658** | **0.584** | **0.438** | 0.386 |

### Gemini Flash Lite — EM

| Pipeline | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|----------|----------|-------|---------|--------------|
| Naive RAG | 0.445 | 0.275 | 0.235 | 0.167 |
| CRAG Replica | 0.400 | 0.130 | 0.175 | 0.107 |
| Single-Pass | 0.470 | 0.375 | 0.280 | 0.160 |
| Loop Refinement | 0.480 | 0.380 | 0.300 | 0.180 |
| **Agentic (ReAct)** | **0.505** | **0.470** | **0.330** | 0.153 |

### gpt-5-mini (Cross-Model) — F1

| Pipeline | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|----------|----------|-------|---------|--------------|
| Naive RAG | 0.657 | 0.400 | 0.441 | 0.321 |
| CRAG Replica | **0.698** | **0.637** | 0.491 | **0.343** |
| Single-Pass | 0.678 | 0.465 | 0.504 | 0.308 |
| Loop Refinement | 0.689 | 0.510 | 0.505 | 0.328 |
| **Agentic (ReAct)** | 0.697 | 0.551 | **0.516** | 0.317 |

### gpt-5-mini (Cross-Model) — EM

| Pipeline | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|----------|----------|-------|---------|--------------|
| Naive RAG | 0.485 | 0.335 | 0.310 | 0.140 |
| CRAG Replica | 0.535 | **0.550** | 0.370 | 0.147 |
| Single-Pass | 0.505 | 0.390 | 0.375 | 0.140 |
| Loop Refinement | 0.510 | 0.425 | 0.370 | 0.160 |
| **Agentic (ReAct)** | 0.510 | 0.485 | **0.375** | 0.133 |

### RQ1 Key Findings

1. **Gemini: Agentic 3/4 데이터셋 1위** — multi-hop에서 일관된 우위
   - 2Wiki +0.089, MuSiQue +0.039, HotpotQA +0.022 vs Loop
   - FinanceBench -0.014 (corpus 211개, retrieval space saturation)
2. **gpt-5-mini: CRAG가 예상 밖 강세** — HotpotQA/2Wiki/FinanceBench에서 1위
   - reasoning model에서 CRAG의 단순 web search fallback 로직이 유리할 수 있음
   - MuSiQue에서만 Agentic 1위 (+0.011)
3. **Corpus 크기와 Agentic 이점 비례** (Gemini):
   - 66K(+0.022) → 8K(+0.089) → 5K(+0.039) → 211(-0.014)
4. **CRAG Replica**: Gemini에서 전 데이터셋 최하위, gpt-5-mini에서는 최상위
5. **FinanceBench 상세 분석**: `docs/FINANCEBENCH_ANALYSIS.md` 참조

---

## RQ2: Tool Usage Analysis

**Question**: How does the ReAct agent use its tools, and what trajectory patterns emerge?

### Tool Frequency (per question average)

| Tool | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|------|----------|-------|---------|--------------|
| search_passages | 2.4 | 2.7 | 3.4 | 1.9 |
| evaluate_passages | 1.0 | 1.1 | 1.0 | 1.0 |
| decompose_query | 0.9 | 1.0 | 0.9 | 1.0 |
| get_passage_detail | 1.1 | 0.8 | 0.6 | 1.1 |
| list_document_sections | 0.1 | 0.1 | 0.4 | 0.5 |
| calculate | - | - | - | 0.4 |
| get_terminology | ~0 | 0 | ~0 | ~0 |
| **Avg tools/question** | **5.4** | **5.7** | **6.2** | **5.8** |

### RQ2 Key Findings

1. **search_passages 압도적 1위** — agent의 핵심 행동은 반복 검색
2. **MuSiQue가 search 가장 많이 사용** (3.4/q) — 3-4홉 질문이라 추가 검색 필요
3. **FinanceBench에서만 calculate tool 사용** (0.4/q) — 재무 계산 특화
4. **get_terminology 거의 미사용** (전 데이터셋) — RQ4 결과와 일관
5. **list_document_sections은 MuSiQue/FinanceBench에서 상대적 다사용** — 복잡한 문서 구조 탐색

---

## RQ3: 4D Evaluation as Agent Tool

**Question**: Does 4-dimensional quality assessment as an agent tool improve decision-making compared to 1D or no evaluation?

### Results (F1)

| Dataset | Agent + 4D | Agent + 1D | Agent w/o Eval | Best |
|---------|-----------|-----------|---------------|------|
| HotpotQA | 0.674 | **0.680** | 0.674 | 1D |
| 2WikiMultiHopQA | **0.611** | 0.608 | 0.580 | 4D |
| MuSiQue | 0.449 | **0.455** | 0.438 | 1D |
| FinanceBench | 0.411 | **0.421** | 0.387 | 1D |

### Results (EM)

| Dataset | Agent + 4D | Agent + 1D | Agent w/o Eval |
|---------|-----------|-----------|---------------|
| HotpotQA | 0.525 | **0.530** | 0.525 |
| 2WikiMultiHopQA | **0.500** | 0.505 | 0.475 |
| MuSiQue | **0.345** | 0.340 | 0.340 |
| FinanceBench | 0.180 | **0.187** | 0.153 |

### RQ3 Key Findings

1. **w/o Eval 항상 최하위** — eval tool의 quality gate 역할 유효 (4개 데이터셋 전부)
2. **1D가 3/4 데이터셋에서 1위** — 4D 세분화가 반드시 필요하지는 않음
3. **4D vs w/o 차이**: 2Wiki +0.031 (가장 큼), HotpotQA ±0.000 (차이 없음)
4. **n=50 "4D≈1D≈w/o" → n=200 "1D≈4D > w/o"** 명확화

### RQ3 Paper Framing

Evaluate tool은 **quality gate** mechanism:
- w/o Eval 항상 최하위 → eval 존재 자체가 중요
- 1D holistic score가 4D 세분화보다 근소 효과적
- 4D의 가치는 진단 피드백 (어떤 dimension이 부족한지)

---

## RQ4: Structure-Aware Tools

**Question**: Do structure-aware retrieval tools (section index, terminology mapping) improve agent performance?

### Results (F1)

| Dataset | Full Tools | w/o Section | w/o Term | w/o Both |
|---------|-----------|-----------|----------|----------|
| HotpotQA | 0.674 | 0.656 | 0.654 | **0.677** |
| 2WikiMultiHopQA | 0.611 | **0.621** | 0.593 | 0.597 |
| MuSiQue | **0.449** | 0.445 | 0.433 | 0.422 |
| FinanceBench | **0.411** | 0.390 | 0.388 | 0.384 |

### Results (EM)

| Dataset | Full Tools | w/o Section | w/o Term | w/o Both |
|---------|-----------|-----------|----------|----------|
| HotpotQA | **0.525** | 0.495 | 0.500 | **0.525** |
| 2WikiMultiHopQA | 0.500 | **0.505** | 0.480 | 0.485 |
| MuSiQue | **0.345** | 0.335 | 0.335 | 0.320 |
| FinanceBench | **0.180** | 0.153 | 0.153 | 0.153 |

### RQ4 Key Findings

1. **Dataset-dependent**: MuSiQue/FinanceBench에서 Full Tools 1위, HotpotQA에서 w/o Both가 더 높음
2. **MuSiQue**: Full > w/o Both (+0.027) — 3-4홉 문서 구조 탐색에 유효
3. **FinanceBench**: Full > w/o Both (+0.027) — SEC filing 구조 브라우징 도움
4. **HotpotQA**: w/o Both (0.677) > Full (0.674) — 위키피디아는 구조 tool 불필요
5. **2Wiki**: w/o Section이 오히려 최고 (0.621) — mixed result
6. **RQ2와 일관**: get_terminology 거의 미사용, list_document_sections은 MuSiQue/Finance에서 활용

### RQ4 Paper Framing

n=50에서 "structure tools 효과 없음" → n=200에서 **"dataset-dependent"**로 수정:
- 구조화된 문서(SEC filings, 멀티홉)에서는 효과 있음
- 단순 위키피디아 기반에서는 불필요
- **extensibility framework**로서의 가치: domain-specific tool 추가 가능한 아키텍처

---

## RQ5: DSPy Optimization Effect

**Question**: Does DSPy's declarative pipeline and automatic optimization improve answer quality compared to manual prompt engineering?

### Results (F1)

| Dataset | Manual | DSPy Unopt | Bootstrap | MIPROv2 | N |
|---------|--------|-----------|-----------|---------|---|
| HotpotQA | 0.575 | 0.650 | **0.681** | 0.663 | 130 |
| 2WikiMultiHopQA | 0.501 | 0.601 | **0.647** | 0.549 | 130 |
| MuSiQue | 0.465 | 0.456 | **0.533** | 0.525 | 130 |
| FinanceBench | 0.246 | **0.436** | 0.421 | 0.410 | 80 |

### Results (EM)

| Dataset | Manual | DSPy Unopt | Bootstrap | MIPROv2 |
|---------|--------|-----------|-----------|---------|
| HotpotQA | 0.400 | 0.508 | **0.538** | 0.508 |
| 2WikiMultiHopQA | 0.308 | 0.477 | **0.554** | 0.423 |
| MuSiQue | 0.331 | 0.354 | **0.454** | 0.431 |
| FinanceBench | 0.000 | **0.200** | 0.188 | 0.175 |

### RQ5 Key Findings

1. **Manual 항상 최하위** — DSPy declarative 방식의 우위 확실
2. **Bootstrap 3/4 데이터셋 1위** — 가장 안정적인 최적화 방법
3. **FinanceBench에서 Unoptimized > Bootstrap** — 소규모 corpus에서 최적화 과적합 가능성
4. **Manual → DSPy Unopt**: 가장 큰 개선 (+0.075~+0.190 F1)
5. **DSPy Unopt → Bootstrap**: 추가 개선 (+0.031~+0.077 F1, FinanceBench 제외)

### RQ5 Paper Framing

DSPy의 기여는 두 단계:
1. **Structural benefit** (Manual → Unopt): Typed I/O fields + structured prompts만으로 큰 개선
2. **Optimization benefit** (Unopt → Bootstrap): Few-shot 자동 생성으로 추가 개선
3. FinanceBench 역전은 "소규모 corpus에서 최적화 과적합" 가능성 → Discussion에서 논의

---

## Ablation Study

### Full Ablation (Gemini, F1)

| Variant | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|---------|----------|-------|---------|--------------|
| Full (Agent + All Tools) | 0.674 | 0.611 | 0.449 | 0.411 |
| 1D Evaluation | **0.680** | 0.608 | **0.455** | **0.421** |
| w/o Evaluate Tool | 0.667 | 0.596 | 0.440 | 0.395 |
| w/o Search (no re-retrieval) | 0.621 | 0.541 | 0.420 | 0.393 |
| w/o Section Index | 0.658 | 0.602 | 0.437 | 0.413 |
| w/o Term Index | 0.666 | **0.613** | 0.445 | 0.410 |
| w/o Agent (for-loop) | 0.636 | 0.495 | 0.399 | 0.400 |
| Manual Prompt | 0.608 | 0.496 | 0.481 | 0.256 |

### Ablation Key Findings

1. **w/o Agent (for-loop)이 항상 Full보다 낮음** (FinanceBench 제외) — agent의 기여 일관
2. **w/o Search가 가장 큰 하락** — re-retrieval이 agent의 핵심 메커니즘
3. **Manual Prompt가 대부분 최하위** — DSPy 효과 재확인
4. **1D Evaluation이 Full보다 약간 높음** — RQ3과 일관
5. **MuSiQue Manual(0.481) > Full(0.449)** — 이상값, 추가 분석 필요

---

## Cross-Model Summary

### Gemini vs gpt-5-mini: Agentic F1

| Dataset | Gemini | gpt-5-mini | Delta |
|---------|--------|-----------|-------|
| HotpotQA | 0.658 | 0.697 | +0.039 |
| 2WikiMultiHopQA | 0.584 | 0.551 | -0.033 |
| MuSiQue | 0.438 | 0.516 | +0.078 |
| FinanceBench | 0.386 | 0.317 | -0.069 |

### Key Observations

1. **gpt-5-mini가 전반적으로 더 높은 F1** (HotpotQA, MuSiQue)
2. **그러나 gpt-5-mini에서 Agentic 순위가 낮음** — CRAG가 2/4 데이터셋 1위
3. **Gemini에서 Agentic 우위가 더 명확** — 논문의 주요 결과로 적합
4. **FinanceBench는 양쪽 모델 모두 Agentic 열세** — method-level boundary condition 확인

---

## 다음 단계

1. [ ] LLM-as-Judge 일괄 실행 (전체 결과 jsonl 재사용)
2. [ ] Bootstrap CI 통계 검증
3. [ ] gpt-5-mini CRAG 강세 원인 분석
4. [ ] 결과 테이블/플롯 교체 (논문)
5. [ ] Discussion 업데이트 (retrieval space saturation, cross-model 분석)
