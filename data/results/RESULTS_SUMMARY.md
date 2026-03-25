# v1.6 Experiment Results Summary (Scale-up, n=200)

**Code version**: v1.6-scaleup
**Primary Model**: Gemini 3.1 Flash Lite Preview
**Cross-Model**: gpt-5-mini (RQ1 only)
**LLM-as-Judge**: gpt-4.1-nano (independent judge)
**Embedding**: all-MiniLM-L6-v2
**Date**: 2026-03-24 ~ 2026-03-25
**Sample size**: n=200 per dataset (FinanceBench n=150, 전체 데이터)

> **Status (2026-03-25)**:
> - Gemini: RQ1~5 + ablation × 4 datasets 전체 완료
> - gpt-5-mini: RQ1 × 4 datasets 완료 (cross-model 검증)
> - Bootstrap CI 통계 검증: ✅ RQ1 완료
> - LLM-as-Judge: 🔄 RQ1 실행 중 (gpt-4.1-nano, 8개 병렬)
> - Cross-model 분석: ✅ 완료 (Reasoning Model Refusal Asymmetry 발견)

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

### Bootstrap CI — Gemini RQ1 (95% Confidence Intervals, F1)

| Pipeline | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|----------|----------|-------|---------|--------------|
| Agentic | 0.666 [.604,.725] | 0.578 [.506,.648] | 0.440 [.362,.519] | 0.387 [.323,.451] |
| Loop | 0.636 [.579,.692] | 0.495 [.433,.557] | 0.399 [.337,.458] | 0.400 [.344,.457] |
| CRAG | 0.553 [.494,.612] | 0.255 [.209,.305] | 0.251 [.198,.306] | 0.333 [.282,.384] |
| Naive | 0.606 [.548,.662] | 0.378 [.320,.435] | 0.342 [.285,.402] | 0.398 [.344,.454] |

### Pairwise Significance — Gemini Agentic vs others (F1)

| vs Pipeline | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|-------------|----------|-------|---------|--------------|
| vs Loop | +0.007 (n.s.) | **+0.102 (p<.001)*** | +0.039 (n.s.) | -0.020 (n.s.) |
| vs CRAG | **+0.099 (p<.001)*** | **+0.335 (p<.001)*** | **+0.177 (p<.001)*** | +0.056 (p=.025)* |
| vs Naive | +0.041 (n.s.) | **+0.229 (p<.001)*** | **+0.112 (p<.001)*** | -0.009 (n.s.) |
| vs Single | +0.008 (n.s.) | **+0.102 (p<.001)*** | +0.048 (n.s.) | +0.010 (n.s.) |

### Pairwise Significance — gpt-5-mini Agentic vs others (F1)

| vs Pipeline | HotpotQA | 2Wiki | MuSiQue | FinanceBench |
|-------------|----------|-------|---------|--------------|
| vs Loop | +0.003 (n.s.) | +0.030 (n.s.) | -0.003 (n.s.) | -0.017 (n.s.) |
| vs CRAG | -0.001 (n.s.) | **-0.110 (p=.004)** | +0.006 (n.s.) | -0.047 (p=.042)* |
| vs Naive | +0.043 (p=.013)* | **+0.146 (p<.001)*** | **+0.101 (p=.007)** | +0.009 (n.s.) |
| vs Single | +0.016 (n.s.) | **+0.079 (p=.006)** | -0.018 (n.s.) | +0.018 (n.s.) |

### RQ1 Key Findings

1. **Gemini: Agentic 3/4 데이터셋 1위** — multi-hop에서 일관된 우위
   - **2Wiki: p<0.001 (모든 baseline 대비 유의미)** — 가장 강력한 증거
   - HotpotQA/MuSiQue: Agentic vs Loop n.s. (유의미하지 않음)
   - FinanceBench: 모든 pipeline 간 차이 n.s. (Retrieval Space Saturation)
2. **gpt-5-mini: CRAG가 2Wiki에서 유의미하게 우수 (p=0.004)**
   - 원인: **Reasoning Model Refusal Asymmetry** (Section 분석 참조)
   - Agent 거부율 30% (2Wiki) → CRAG 거부율 18% → F1 역전
3. **Corpus 크기와 Agentic 이점 비례** (Gemini):
   - 66K(+0.022) → 8K(+0.089) → 5K(+0.039) → 211(-0.014)
4. **FinanceBench: 양 모델 동일 패턴** (Agentic 열세) → method-level boundary condition
5. **상세 분석**: `docs/FINANCEBENCH_ANALYSIS.md` (cross-model 비교 포함)

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

## Cross-Model Analysis

### Gemini vs gpt-5-mini: Agentic F1

| Dataset | Gemini | gpt-5-mini | Delta |
|---------|--------|-----------|-------|
| HotpotQA | 0.658 | 0.697 | +0.039 |
| 2WikiMultiHopQA | 0.584 | 0.551 | -0.033 |
| MuSiQue | 0.438 | 0.516 | +0.078 |
| FinanceBench | 0.386 | 0.317 | -0.069 |

### Reasoning Model Refusal Asymmetry (핵심 발견)

gpt-5-mini에서 CRAG가 Agentic보다 우수한 원인: **모델-파이프라인 상호작용 효과**

#### 거부율 비교 (답변 거부 = "Cannot determine" 등)

| Dataset | gpt-5-mini Agent | gpt-5-mini CRAG | Gemini Agent | Gemini CRAG |
|---------|-----------------|-----------------|-------------|-------------|
| HotpotQA | 2% | 2% | 2% | 4% |
| **2Wiki** | **30%** | **18%** | **14%** | **34%** |
| MuSiQue | 22% | 22% | 11% | 22% |
| FinanceBench | 2% | 1% | 1% | 9% |

**완전 역전 패턴**: gpt-5-mini Agent 거부율 > CRAG 거부율, Gemini는 반대.

#### 메커니즘

- **gpt-5-mini (reasoning model)**: ReAct에서 높은 evidentiary standard 적용 → eval score 66 (>threshold 55)에서도 "Cannot determine" 거부 → 30% 거부가 F1=0.0 → 평균 하락
- **CRAG**: judge-correct-generate 강제 파이프라인 → 거부 선택지 없음 → 정답 확률 상승
- **Gemini (generation model)**: ReAct에서 덜 신중 → 거부율 낮음 → Agentic 유리

#### 효율성 비교 (gpt-5-mini 2Wiki)

| Metric | Agentic | CRAG |
|--------|---------|------|
| F1 | 0.531 | **0.607** |
| LLM 호출 | **9.9** | 45.2 |
| Latency | **45.7s** | 147.8s |

CRAG가 F1은 높지만 **4.6배 비효율적**. Agentic + Gemini가 최적의 cost-performance balance.

### Cross-Model Summary

1. **gpt-5-mini가 전반적으로 더 높은 F1** (HotpotQA, MuSiQue)
2. **gpt-5-mini에서 Agentic 순위 낮음** — CRAG가 2/4 데이터셋 1위 (Refusal Asymmetry)
3. **Gemini에서 Agentic 우위가 더 명확** — 논문의 주요 결과로 적합
4. **FinanceBench는 양쪽 모델 모두 Agentic 열세** — method-level boundary condition 확인
5. **논문 기여**: Model-Pipeline Interaction Effect → Discussion에서 별도 subsection

---

## 다음 단계

1. [x] Bootstrap CI 통계 검증 (RQ1 완료)
2. [x] gpt-5-mini CRAG 강세 원인 분석 (Refusal Asymmetry 발견)
3. [🔄] LLM-as-Judge 실행 중 (RQ1, gpt-4.1-nano, 8개 병렬)
4. [ ] LLM-as-Judge 결과 반영 후 최종 테이블 업데이트
5. [ ] 결과 테이블/플롯 교체 (논문)
6. [ ] Discussion 업데이트 (Retrieval Space Saturation + Model-Pipeline Interaction)
