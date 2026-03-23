# SCIE Q1 제출 수준 평가 & 전략

**평가일**: 2026-03-23
**현재 상태**: v1.5 실험 전체 완료 (RQ1~5 + LLM-as-Judge, n=50)
**결론**: Contribution 재구성 + 실험 스케일업 후 제출

---

## 1. 강점 (Publishable Qualities)

1. **명확한 연구 구조**: 5 RQ + 4 Dataset 체계적 설계
2. **Agentic Refinement**: Complex multi-hop에서 일관된 1위 — 2Wiki F1 +0.069, MuSiQue +0.051. 복잡도에 비례하는 개선 패턴
3. **DSPy Pipeline**: Manual→DSPy Unopt +0.1~0.2 F1. 가장 큰 개선폭
4. **Honest negative results**: 학술적 credibility 향상
5. **Tool trajectory 분석**: Agent 행동 패턴의 정량적 분석
6. **다양한 평가**: EM + F1 + LLM-as-Judge + ROUGE-L, latency 포함

---

## 2. 약점 & 보완 계획

### 심각 (Must-fix before submission)

| # | 약점 | 보완 계획 | 시점 |
|---|------|----------|------|
| 1 | n=50 샘플 | n=500+ 스케일업 | 초고 완성 후 |
| 2 | Gemini Flash Lite only | +GPT-4o-mini cross-model | 초고 완성 후 |
| 3 | 통계 검증 없음 | Bootstrap CI + paired test | 스케일업 시 |
| 4 | C2/C3 독립 contribution 약함 | **해결: 1개 통합 프레임워크로 재구성 (아래 참조)** | ✅ 완료 |

### 중간 (Address in paper)

| 약점 | 대응 |
|------|------|
| Baseline 공정성 (CRAG Replica) | 원논문 수치 별도 테이블 인용 + 동일 조건 비교임을 명시 |
| Latency 50배 | Latency-quality trade-off 분석 + discussion 섹션 |
| HotpotQA 무개선 | "복잡도 비례 개선" narrative로 프레이밍 (simple에서는 overhead) |
| Single embedding | Limitation으로 명시 |

---

## 3. Contribution 재구성 (2026-03-23 확정)

### 기존 (문제)
4개 독립 contribution → C2/C3 결과 약해서 "2/4 실패"로 보임

### 재구성 (해결)
**1개 통합 프레임워크 + ablation studies**

```
[Main Contribution: Agentic Self-Corrective RAG Framework]
│
├── Method: ReAct agent + 6 tools (구 C1) — RQ1, RQ2
│   → Agentic이 complex multi-hop에서 일관 우위
│   → Agent가 structured protocol 자발적 형성
│
├── Implementation: DSPy declarative pipeline (구 C4) — RQ5
│   → Signature만으로 +0.1~0.2 F1
│   → Bootstrap/MIPROv2 최적화는 dataset-dependent
│
├── Ablation 1: Eval tool variants (구 C2) — RQ3
│   → Quality gate mechanism (accuracy boost 아님)
│
├── Ablation 2: Structure-aware tools (구 C3) — RQ4
│   → Honest negative: well-chunked corpus에서 효과 없음
│
└── Analysis: Tool trajectory patterns — RQ2
    → decompose→search→evaluate 패턴 자발적 형성
```

**핵심 메시지**: "Complex multi-hop QA에서 agentic refinement이 효과적이며, DSPy declarative pipeline이 이를 가능케 한다."

**근거**: Q1 비교 논문들도 core contribution 1개
- Self-RAG: self-reflection tokens (1개)
- CRAG: corrective retrieval action (1개)
- Adaptive-RAG: complexity-adaptive routing (1개)

---

## 4. 타겟 저널

| 순위 | 저널 | IF | 리뷰 기간 | APC | 적합도 |
|------|------|-----|----------|-----|--------|
| 1순위 | **KBS** | 7.6 | ~5.5개월 | 선택 | ★★★★★ |
| 2순위 | **EAAI** | 8.0 | ~3-6개월 | 선택 | ★★★★☆ |
| 안전망 | **JKSUCIS** | 6.1 | ~3-6개월 | **무료** | ★★★☆☆ |
| 대안 | **IP&M** | 7.4 | ~3-6개월 | 선택 | ★★★★☆ |
| 대안 | **Info Sciences** | 8.1 | ~4-8개월 | 선택 | ★★★★☆ |

- **ESWA** (IF 7.5): 리뷰 7-14개월으로 일정 리스크 → 비추
- **Applied Intelligence**: Q2 하락 (IF 3.4) → 제외
- **AI Review**: Survey 전용 → 제외

---

## 5. 로드맵

```
Phase 1: 논문 초고 작성 (n=50 기반)
         - Method, Experiments, Results, Discussion
         - Figure & 아키텍처 다이어그램
         - Related Work

Phase 2: 실험 스케일업
         - n=500+ (4 datasets)
         - +GPT-4o-mini (cross-model)
         - Bootstrap CI 통계 검증

Phase 3: 결과 교체 & 마무리
         - Result 테이블 숫자 교체
         - 통계 검증 결과 추가
         - Discussion/Limitation 보완

Phase 4: KBS 제출
         → Accept: 졸업요건 충족
         → Reject: EAAI 또는 JKSUCIS로 전환
```

---

## 6. 졸업요건 참고

- SCIE Q1 = 5점 (1편 = 졸업요건 충족)
- SCIE Q2 = 4점, Scopus = 3점, KCI = 2점
- 박사 1.5년 과정 (2026-03 시작, 6학기 × 3개월)
- 리뷰 ~6개월 → 입학 후 6개월 내 제출 목표 (2026-09까지)
