# Final Pre-Submission Review: TARA — KBS

**Date**: 2026-03-27
**Reviewers**: 2x Claude KBS Reviewer (methodology + novelty/framing)
**Verdict**: Minor Revision (both), Confidence 4/5

---

## Review Summary

### Overall: KBS 통과 가능성 높음

두 리뷰어 모두 Minor Revision. 실험 방법론의 성실함과 negative result 보고의 정직함을 높이 평가.

### Biggest Risk: "2Wiki 1개에서만 significant"

| Dataset | vs Loop Δ F1 | p-value | Cohen's d | 유의? |
|---------|-------------|---------|-----------|------|
| 2WikiMultiHopQA | +0.089 | <0.001 | 0.235 | **Yes** |
| MuSiQue | +0.039 | 0.161 | 0.087 | No |
| HotpotQA | +0.022 | 0.772 | 0.016 | No |
| FinanceBench | -0.014 | 0.394 | 0.058 | No |

---

## Strengths (Both Reviewers)

1. **통계 방법론 우수** — Bootstrap CI, Cohen's d, sensitivity analysis는 RAG 분야 평균 이상
2. **Negative result 정직 보고** — FinanceBench 열세, terminology tool 미사용, MuSiQue anomaly
3. **Reasoning Model Refusal Asymmetry** — 가장 novel한 기여, qualitative evidence 포함
4. **Ablation 체계적** — 5 RQ + consolidated table, 2×2 factorial independence test
5. **DSPy structural vs optimization 분리** — 깔끔한 방법론적 기여

---

## Weaknesses & Action Plan

### 🟡 Major — 조치 가능 (텍스트 수정)

| # | 지적 | 대응 | 상태 |
|---|------|------|------|
| M1 | 2Wiki 1개에서만 significant | 2Wiki question type별 breakdown 분석 추가 | ⏳ |
| M2 | Missing recent agentic RAG 비교 | Related Work에 Search-o1, PlanRAG 등 1 paragraph 추가 | ⏳ |
| M3 | IRCoT/Self-RAG published 수치 간접 비교 | Discussion에 corpus 차이 명시한 비교 테이블 또는 텍스트 추가 | ⏳ |
| M4 | "Emergent behavior" 과장 | Signature가 protocol 제안 → "instruction-guided convergence"로 톤 조정 | ⏳ |

### 🟡 Major — 이미 대응 완료

| # | 지적 | 대응 | 상태 |
|---|------|------|------|
| W1 | Novelty (agent+DSPy incremental) | 2×2 factorial → complementary contributions | ✅ P1 |
| W2 | Baseline 공정성 | setup.tex: trained components confound | ✅ P3 |
| W3/4 | 통계 유의성 | Effect size + sensitivity analysis | ✅ P0 |
| W5 | Cross-model 톤 | OR-Bench + AbstentionBench citation + qualitative evidence | ✅ P2 |
| W7 | Tool 수 과장 | "4 core + 2 domain-adaptive" | ✅ P4 |

### 🟢 Minor — 방어 가능 (추가 조치 불필요)

| # | 지적 | 방어 |
|---|------|------|
| CRAG 약화 | 이미 setup.tex에서 인정, closed-corpus 필연적 제약 |
| Closed corpus 일반화 | Limitation에 명시, enterprise 시나리오 프레이밍 |
| n=200 사후 정당화 | Prior work 인용 (Self-RAG, CRAG), sensitivity analysis |
| DSPy 별도 논문 같음 | 2×2 factorial로 independence 입증, KBS는 applied systems 저널 |
| Human eval 없음 | LLM-as-Judge로 대체, scope 밖 |
| Latency 선택적 배포 | Future work, Adaptive-RAG 이미 인용 |
| Calculate tool | method.tex에 "optional" 명시 |

---

## Priority Action Items

1. **M4**: "Emergent behavior" 톤 조정 (가장 쉬움)
2. **M2**: Recent agentic RAG 비교 paragraph
3. **M1**: 2Wiki question type breakdown (데이터 분석 필요)
4. **M3**: IRCoT/Self-RAG 간접 비교 (검색 필요)

---

## Reviewer Questions — 답변 준비

| Q | 질문 | 답변 방향 |
|---|------|----------|
| Q1 | 2Wiki hop별 breakdown? | 데이터에서 추출 가능 — 4-hop에서 gain 집중이면 complexity narrative 강화 |
| Q2 | 대규모 corpus 실험? | Future work. FinanceBench saturation이 간접 증거 |
| Q3 | gpt-5-mini 전 dataset 거부율? | 이미 JSONL에 있음, 테이블로 정리 가능 |
| Q4 | tau=40, delta=5 sensitivity? | Pilot에서 결정, 본 실험에서 미변경. 한계로 인정 |
| Q5 | Emergent vs instruction-following? | Signature는 protocol 제안, 실제 순서/빈도는 agent 결정 → partial emergence |
| Q6 | Seed variance? | seed=42 + temperature=0, LLM non-determinism은 한계로 인정 |
| Q7 | RQ5 n=130 vs RQ1 n=200? | Train/val split 필연적, 동일 질문 subset이 아님 → 한계로 인정 |
