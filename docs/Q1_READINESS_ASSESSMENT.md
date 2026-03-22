# SCIE Q1 (KBS) 제출 수준 평가

**평가일**: 2026-03-23
**현재 상태**: v1.5 실험 전체 완료 (RQ1~5 + LLM-as-Judge)

---

## 강점 (Publishable Qualities)

1. **명확한 연구 구조**: 5 RQ + 4 Contribution + 4 Dataset 체계적 설계
2. **C1 (Agentic Refinement)**: Complex multi-hop에서 일관된 1위 — 2Wiki F1 +0.069, MuSiQue +0.051. 복잡도에 비례하는 개선 패턴은 설득력 있음
3. **C4 (DSPy)**: Manual→DSPy Unopt +0.1~0.2 F1은 큰 개선. 실용적 기여가 명확
4. **Honest negative results** (C3): 학술적으로 정직한 보고는 오히려 credibility를 높임
5. **Tool trajectory 분석** (RQ2): Agent 행동 패턴의 정량적 분석은 독자적 가치 있음
6. **다양한 평가**: EM + F1 + LLM-as-Judge + ROUGE-L, latency 포함

---

## 심각한 약점 (Major Concerns)

### 1. n=50 — 가장 치명적인 문제
- KBS/Q1 수준 RAG 논문은 **최소 500~전체 데이터셋** 사용
- Self-RAG: PopQA 14K, NQ 3.6K 전체 사용
- CRAG: PopQA 14K + 별도 벤치마크
- n=50에서 F1 차이 0.016~0.069는 **통계적 유의성 검증 불가**
- Reviewer 예상: *"Sample size of 50 is insufficient for meaningful conclusions"*

### 2. 모델 선택: Gemini Flash Lite
- 최저가 모델로 전체 실험 수행 → generalizability 의문
- 비교 논문들: GPT-4, GPT-3.5-turbo, Llama-2-13B 등 사용
- Reviewer 예상: *"Do these results hold with stronger models?"*

### 3. 4개 Contribution 중 2개가 약함
- **C2 (4D Eval)**: max delta 0.036, 방향조차 일관되지 않음. "Quality gate" framing은 contribution이라 부르기 약함
- **C3 (Structure Tools)**: Negative result. Full < w/o Both. Extensibility framework은 실험적 기여가 아님
- → 실질적으로 **C1 + C4 두 개의 contribution**만 결과로 지지됨

### 4. 통계적 검증 부재
- p-value, confidence interval, effect size 없음
- n=50에서 F1 0.714 vs 0.726 차이가 유의한지 알 수 없음
- Q1 저널은 **statistical significance test 필수**

### 5. Baseline 공정성 문제
- "CRAG Replica"는 원논문 CRAG가 아닌 simplified replication
- Retriever가 다름 (원논문: Contriever on DPR Wikipedia, 본 연구: FAISS+BM25)
- 원논문 수치와 직접 비교 불가 → *"unfair comparison"* 지적 가능

### 6. Latency 비용
- Agentic 14-16s vs Naive 0.3s — **50배 느림**
- 이 trade-off에 대한 심도 있는 discussion 필요

---

## 중간 수준 약점 (Moderate Concerns)

| 항목 | 이슈 |
|------|------|
| HotpotQA 무개선 | 가장 널리 쓰이는 벤치마크에서 Naive와 동일한 F1 |
| Novelty | ReAct + Tools + RAG 조합은 engineering integration에 가까움 |
| 절대 성능 | MuSiQue F1 0.453, FinanceBench EM 0.220 — SOTA와 큰 격차 |
| Single embedding | all-MiniLM-L6-v2만 사용, 다른 embedding 미검증 |

---

## 종합 판단

| 항목 | 현재 수준 | Q1 기준 |
|------|----------|---------|
| 연구 설계 | ★★★★☆ | ★★★★★ |
| 실험 규모 | ★★☆☆☆ | ★★★★★ |
| 결과 강도 | ★★★☆☆ | ★★★★☆ |
| 통계 검증 | ★☆☆☆☆ | ★★★★★ |
| Novelty | ★★★☆☆ | ★★★★☆ |
| Baseline 공정성 | ★★☆☆☆ | ★★★★★ |

**현재 상태로 KBS 제출 시: Reject 가능성 높음 (Major Revision 이상)**

---

## Q1 제출을 위한 필수 보완 사항

### 반드시 필요 (Must-have)

1. **Sample size 확대** — 최소 n=500, 이상적으로 전체 데이터셋. 이것만으로도 credibility가 크게 달라짐
2. **통계적 유의성 검증** — Bootstrap confidence interval 또는 paired t-test/Wilcoxon signed-rank test
3. **최소 2개 이상 모델로 실험** — GPT-4o-mini + Gemini Flash 등 cross-model validation
4. **Contribution 구조 정리** — C2/C3를 별도 contribution으로 세우지 말고, C1의 하위 분석 또는 ablation study로 재편

### 강하게 권장 (Should-have)

5. **CRAG/Self-RAG 원논문과 동일 retriever(Contriever+DPR corpus)로 재실험**, 또는 최소한 원논문 수치를 별도 테이블로 인용하여 비교
6. **Latency-quality trade-off 분석** — Pareto frontier 그래프 등
