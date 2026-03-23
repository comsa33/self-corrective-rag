# Paper Plan — Agentic Self-Corrective RAG

**작성일**: 2026-03-23
**타겟 저널**: KBS (Knowledge-Based Systems, SCIE Q1, IF 7.6)

---

## 논문 제목 (Working Title)

**Agentic Self-Corrective RAG: Autonomous Retrieval Refinement via Tool-Augmented Language Models**

---

## Core Contribution

**하나의 통합 프레임워크**: ReAct-based tool-augmented agent가 retrieval을 자율적으로 refinement하는 프레임워크. DSPy declarative pipeline 위에 구현하여 자동 최적화 가능.

### 핵심 메시지
> Complex multi-hop QA에서 agentic refinement이 효과적이며, DSPy declarative pipeline이 이를 가능케 한다.

---

## 논문 구조 (Outline)

### 1. Introduction
- Multi-hop QA의 어려움: 단순 retrieve-and-read 한계
- Self-corrective RAG의 발전 (CRAG → Loop → Agentic)
- 기존 접근의 한계: 고정 루프, 수동 프롬프트, 도구 활용 부재
- 제안: ReAct agent + 6 tools + DSPy pipeline
- Contribution 요약 (1개 통합 프레임워크)

### 2. Related Work
- RAG: Naive → Self-RAG → CRAG → Adaptive-RAG
- Agentic RAG: ReAct, tool use in LLMs
- DSPy: Declarative LM programming, optimization
- Multi-hop QA: HotpotQA, 2Wiki, MuSiQue

### 3. Method
- 3.1 Overall Architecture (Figure: system diagram)
- 3.2 ReAct Agent + Tool Design (6 tools 설명)
- 3.3 Self-Corrective Refinement Loop
- 3.4 DSPy Declarative Pipeline (Signatures, optimization)
- 3.5 Hybrid Retrieval (FAISS + BM25 + RRF)

### 4. Experimental Setup
- 4.1 Datasets: HotpotQA, 2WikiMultiHopQA, MuSiQue, FinanceBench
- 4.2 Baselines: Naive, CRAG Replica, Single-Pass, Loop Refinement
- 4.3 Metrics: EM, F1, LLM-as-Judge (primary), ROUGE-L (secondary)
- 4.4 Implementation: Models, hyperparameters, infrastructure

### 5. Results & Analysis
- 5.1 RQ1: Agentic vs Baselines (main result table)
- 5.2 RQ2: Tool Usage & Trajectory Analysis
- 5.3 RQ5: DSPy Optimization Effect

### 6. Ablation Studies
- 6.1 RQ3: Evaluation Tool Variants (4D/1D/w/o)
- 6.2 RQ4: Structure-Aware Tools (honest negative)

### 7. Discussion
- Complexity-dependent improvement pattern
- Latency-quality trade-off
- When agentic refinement is (not) worth it
- DSPy의 structural benefit vs optimization benefit
- Limitations

### 8. Conclusion & Future Work

---

## Research Questions → 논문 위치 매핑

| RQ | 질문 | 강도 | 논문 위치 |
|----|------|------|----------|
| RQ1 | Agentic vs baselines | **강** | §5.1 Main Results |
| RQ2 | Tool usage patterns | **강** | §5.2 Analysis |
| RQ3 | Eval tool variants | 중 | §6.1 Ablation |
| RQ4 | Structure-aware tools | 중 | §6.2 Ablation (negative) |
| RQ5 | DSPy optimization | **강** | §5.3 Main Results |

---

## 필요한 Figures

1. **System Architecture Diagram** — 전체 프레임워크 (ReAct + tools + DSPy)
2. **Pipeline Comparison** — Naive/CRAG/Loop/Agentic 흐름도
3. **Main Results Bar Chart** — F1 across 4 datasets × 5 pipelines
4. **Tool Trajectory Sankey/Flow** — Agent tool usage patterns
5. **DSPy Optimization Effect** — Manual vs Unopt vs Bootstrap vs MIPROv2
6. **Complexity vs Improvement** — Hop count별 개선폭 (핵심 narrative)
7. **Latency-Quality Trade-off** — Pareto frontier

---

## 현재 상태

- [x] 실험 완료 (n=50 pilot, v1.5)
- [x] Contribution 구조 재편
- [x] 타겟 저널 결정
- [ ] 논문 초고 작성
- [ ] 실험 스케일업 (n=500, cross-model)
- [ ] 통계 검증 추가
- [ ] 최종 제출
