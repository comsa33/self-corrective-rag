# Reviewer Report: TARA — KBS Submission

**Date**: 2026-03-26
**Target Journal**: Knowledge-Based Systems (Elsevier, SCIE Q1, IF 7.6)
**Overall Recommendation**: Major Revision

---

## Summary

본 논문은 고정 루프 기반 self-corrective RAG를 ReAct 에이전트 + 6개 도구로 대체하는 TARA 프레임워크를 제안한다. DSPy 선언적 파이프라인과 결합하여 4개 데이터셋에서 평가하였다.

---

## Strengths

### S1. 실험 설계가 체계적
- n=200, paired bootstrap significance testing, Bonferroni correction, 95% CI까지 갖춤
- 5개 RQ + ablation + cross-model validation은 Q1 저널 수준에 부합

### S2. 솔직한 결과 보고
- FinanceBench에서의 성능 저하, terminology tool의 미미한 기여 등 부정적 결과를 숨기지 않고 분석함
- "Reasoning Model Refusal Asymmetry" 발견은 흥미롭고 실용적 시사점이 있음

### S3. Ablation이 충실
- RQ3(평가 도구), RQ4(구조 도구) 각각 변형 실험으로 컴포넌트별 기여도를 분리

### S4. 재현성
- 코드 공개, YAML config, fixed seed, CLI 자동화 등 재현 인프라가 잘 갖춰짐

---

## Weaknesses (Major)

### W1. Novelty가 약함 — 기존 도구의 조합에 가까움
- ReAct는 Yao et al. (2023), DSPy는 Khattab et al. (2024), hybrid retrieval은 표준 기법
- 핵심 contribution이 "이 셋을 합쳤다"에 가까움
- **"왜 이 조합이 비자명(non-trivial)한가"에 대한 이론적/직관적 근거가 부족**
- 6개 tool 설계의 원칙(왜 6개인가? 왜 이 6개인가?)에 대한 정당화가 없음

### W2. 베이스라인이 공정하지 않음
- CRAG Replica에서 web search → internal re-retrieval, T5-large evaluator → LLM-based eval로 교체. 이는 CRAG의 핵심 설계를 변형한 것
- 논문에서 "이 적응이 CRAG에 불리하다"고 인정하지만, 그렇다면 **왜 이 베이스라인을 포함하는가?** 공정하지 않은 비교는 리뷰어에게 인상이 좋지 않음
- Self-RAG, Adaptive-RAG가 Related Work에는 있지만 **실험 비교 대상에서 빠짐**. 이들이 직접 비교 대상이 아닌 이유가 불충분

### W3. 4개 데이터셋 중 1개에서만 통계적으로 유의미한 개선
- 2WikiMultiHopQA에서만 p<0.001. HotpotQA, MuSiQue는 not significant, FinanceBench는 오히려 하락
- 리뷰어는 "4개 중 1개에서만 유의미하면 robust하다고 할 수 있는가?"라고 질문할 것
- 특히 MuSiQue가 "가장 어려운 데이터셋"이라고 했는데 여기서 유의미하지 않은 점이 약점

### W4. 샘플 사이즈가 작음 (n=200)
- HotpotQA validation set은 7,405개, 2WikiMultiHopQA는 12,576개. 200개는 전체의 ~2%
- 샘플링 편향(sampling bias) 우려가 있고, 리뷰어가 "왜 더 크게 하지 않았는가" 질문할 가능성 높음
- "API cost" 때문이라면 Limitation에 명시하고, 200개가 충분한 statistical power를 갖는다는 power analysis를 추가해야 함

### W5. Cross-model validation에서 TARA가 CRAG에 지는 문제
- gpt-5-mini에서 CRAG Replica가 TARA를 이기는 것은 심각한 약점
- "Refusal Asymmetry"로 설명하지만, 리뷰어는 **"그러면 TARA가 model-specific하다는 것 아닌가?"**라고 해석할 수 있음
- 이를 contribution으로 포장했지만, 실질적으로는 generalizability 부재의 증거

---

## Weaknesses (Minor)

### W6. Latency 오버헤드 과소평가
- 14-17초/질문은 "acceptable for enterprise"라고 주장하지만, Loop의 2-4초 대비 4-5배
- F1 +0.089를 위해 4-5배 비용을 지불하는 것이 실용적인지에 대한 분석이 부족

### W7. Terminology tool이 사실상 무용
- `get_terminology`는 모든 데이터셋에서 <0.1 calls/question
- 6개 tool이라고 주장하지만 실질적으로 5개 (또는 4개)
- **이를 contribution에서 "6 tools"라고 강조하는 것은 과장**

### W8. 저자가 1명
- KBS에서 single-author 논문이 불가능하진 않지만, 지도교수 공저가 없으면 리뷰어가 연구 감독(supervision) 부재를 우려할 수 있음
- `\author[1]{Ruo Lee}` 한 명뿐이고 corresponding author 지정도 없음

### W9. 모델 선택의 불투명성
- "Gemini Flash Lite"는 preview 모델이고, "gpt-5-mini"도 최신 모델
- 재현성 측면에서 이 모델들이 deprecate되면 결과 재현이 불가. 이에 대한 논의 필요

### W10. Table/Figure가 과다
- 54페이지 분량에 테이블과 figure가 상당히 많음. KBS는 보통 15-20페이지 내외
- `review` 모드라 두 배 간격이긴 하지만, 컨텐츠 자체도 축약 필요

---

## Specific Questions a Reviewer Would Ask

1. **Self-RAG, Adaptive-RAG와 직접 비교하지 않은 이유는?** Related Work에서 주요 비교 대상으로 논하면서 실험에서 빠진 것은 설득력이 약함
2. **n=200이 충분한 statistical power를 갖는다는 근거는?** Power analysis 또는 effect size 기반 정당화 필요
3. **"emergent behavior"라고 부르기엔** agent의 행동이 signature docstring에 protocol로 이미 권장되어 있음. 이것이 진정한 emergence인가, guided behavior인가?
4. **FinanceBench에서 Naive RAG가 최고인데**, 이것은 RAG refinement 자체의 가치를 의문시하는 결과 아닌가?
5. **CRAG를 불공정하게 비교한 뒤 "consistently ranks lowest"라고 서술하는 것**은 misleading하지 않은가?

---

## Recommendations for Revision

| 우선순위 | 항목 | 대응 방안 |
|---------|------|----------|
| **필수** | W1. Novelty 강화 | 조합의 비자명성에 대한 이론적 근거 추가, tool 설계 원칙 정당화 |
| **필수** | W2. 베이스라인 공정성 | Self-RAG/Adaptive-RAG 실험 추가 또는 제외 사유 명확화 |
| **필수** | W3. 유의성 보강 | n 증가 또는 power analysis 추가, effect size 보고 |
| **필수** | W5. Cross-model 해석 | "finding"이 아닌 "limitation"으로 톤 조정, 추가 모델 실험 |
| 권장 | W2. CRAG 서술 톤 | "lowest"가 아니라 "구현 제약으로 인해 원본 대비 불리"로 수정 |
| 권장 | W7. Tool 수 표현 | "6 tools" → "5 core tools + 1 optional" 또는 terminology 유용 시나리오 실험 |
| 권장 | W8. 공저자 확보 | 지도교수 공저 추가 강력 권장 |
| 권장 | W10. 분량 조절 | KBS 형식에 맞게 축약, 일부 table을 supplementary로 이동 |

---

## Overall Assessment

실험의 체계성과 솔직한 보고는 강점이지만, **novelty 부족**(기존 기법 조합), **제한적 유의성**(4개 중 1개), **불공정 베이스라인**(CRAG 변형)이 Q1 수준에서는 걸림돌이다.

특히 W1(novelty)과 W3(유의성)은 에디터가 desk reject 판단 시 가장 먼저 보는 항목이므로 **abstract과 introduction에서 contribution을 더 날카롭게** 정리하는 것이 중요하다.

Major revision으로 위 사항을 보완하면 재투고 시 경쟁력이 높아질 것이다.
