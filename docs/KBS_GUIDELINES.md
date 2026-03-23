# KBS (Knowledge-Based Systems) 논문 작성 가이드

**저널**: Knowledge-Based Systems (Elsevier)
**ISSN**: 0950-7051 | **IF**: 7.6~9.6 | **Q1** Computer Science / AI
**리뷰**: Single-blind, 최소 2명, ~5.5개월

---

## 1. 형식 요구사항

| 항목 | 요구사항 |
|------|----------|
| 분량 | **20 double-spaced pages 이하** (tables, figures 포함) |
| Abstract | **250 words 이하** |
| Keywords | **1-7개**, "and"/"of" 포함 multi-word 피할 것 |
| Highlights | **3-5개 bullet**, 각 **85자 이하** |
| References | 60-80개 권장 (AI/NLP 기준) |
| Figures | 5-10개 |
| Tables | 4-8개 |
| Citation Style | Numbered (elsarticle-num) |
| Review Type | Single-blind (저자 공개) |
| 필수 제출물 | CRediT Author Statement, Data Availability, AI Declaration |

---

## 2. LaTeX 설정

```latex
\documentclass[preprint,review,number]{elsarticle}
\bibliographystyle{elsarticle-num}

% preprint = single column 제출 포맷
% review = 리뷰어용 넓은 줄간격
% number = 번호 인용 스타일 (KBS 표준)
```

**Overleaf 템플릿**: https://www.overleaf.com/latex/templates/elsevier-article-elsarticle-template/vdzfjgjbckgz

**주의**: 모든 파일 (tex, bbl, bst, figures)은 **같은 폴더 레벨**에 위치해야 함 (하위폴더 불가)

### Frontmatter 구조

```latex
\begin{frontmatter}
  \title{Agentic Self-Corrective RAG: Autonomous Retrieval
         Refinement via Tool-Augmented Language Models}

  \author[1]{Author Name\corref{cor1}}
  \cortext[cor1]{Corresponding author}
  \ead{email@example.com}
  \affiliation[1]{organization={University}, city={City}, country={Country}}

  \begin{abstract}
    ... (max 250 words)
  \end{abstract}

  \begin{highlights}
    \item Highlight 1 (max 85 chars)
    \item Highlight 2
    \item Highlight 3
  \end{highlights}

  \begin{keyword}
    retrieval-augmented generation \sep
    agentic refinement \sep
    tool-augmented LLM \sep
    DSPy \sep
    multi-hop question answering
  \end{keyword}
\end{frontmatter}
```

---

## 3. 논문 섹션 구조

### Abstract (250 words 이내)
1. Context/Problem (1-2문장): Multi-hop QA + RAG의 한계
2. Gap (1문장): 기존 고정 루프/수동 프롬프트의 문제
3. Method (2-3문장): ReAct agent + 6 tools + DSPy pipeline
4. Results (2-3문장): 핵심 수치 (2Wiki +0.069, DSPy +0.1~0.2)
5. Significance (1문장): 실무적 함의

### 1. Introduction (1.5-2 pages)
**Swales CARS 모델**:
1. **Establish Territory**: Multi-hop QA의 중요성, RAG의 발전
2. **Establish Niche**: 기존 접근의 한계 (고정 루프, 도구 부재, 수동 프롬프트)
3. **Occupy the Niche**: 제안 방법 + contribution 명시

끝에 numbered contribution list:
> The main contributions of this paper are as follows:
> 1. We propose an agentic self-corrective RAG framework...
> 2. We provide comprehensive empirical evaluation...

### 2. Related Work (2-3 pages)
**Hybrid 방식 권장**: 서술형 subsection + 비교 테이블

구성:
- 2.1 Retrieval-Augmented Generation (Naive → Advanced)
- 2.2 Self-Corrective RAG (CRAG, Self-RAG, Adaptive-RAG)
- 2.3 Tool-Augmented LLM Agents (ReAct, Toolformer)
- 2.4 Declarative LM Programming (DSPy)

**비교 테이블** (Table 1):

| Method | Iterative | Tool Use | Declarative | Multi-hop | Agent |
|--------|-----------|----------|-------------|-----------|-------|
| Naive RAG | - | - | - | - | - |
| Self-RAG | - | - | - | O | - |
| CRAG | O | Web Search | - | - | - |
| Adaptive-RAG | - | - | - | O | Routing |
| **Ours** | **O** | **6 Tools** | **DSPy** | **O** | **ReAct** |

### 3. Method (4-5 pages)
- 3.1 Problem Formulation (수학적 정의)
- 3.2 Overall Architecture (Figure 1: 전체 시스템 다이어그램)
- 3.3 ReAct Agent & Tool Design
  - 6 tools 각각 설명 (search, decompose, evaluate, structure, terminology, inspect)
  - Algorithm pseudocode for agentic loop
- 3.4 Self-Corrective Refinement Mechanism
- 3.5 DSPy Declarative Pipeline
  - Signature 설계
  - Optimization (BootstrapFewShot, MIPROv2)
- 3.6 Hybrid Retrieval (FAISS + BM25 + RRF)

### 4. Experiments (5-7 pages)
- 4.1 Setup
  - Datasets (Table: 통계), Baselines, Metrics, Implementation Details
- 4.2 Main Results: RQ1 (Table: 4 datasets × 5 pipelines × 3 metrics)
- 4.3 Tool Usage Analysis: RQ2 (Figure: tool distribution, trajectory patterns)
- 4.4 DSPy Optimization Effect: RQ5 (Table: 4 optimization variants)

### 5. Ablation Studies (2-3 pages)
- 5.1 Evaluation Tool Variants: RQ3 (4D/1D/w/o)
- 5.2 Structure-Aware Tools: RQ4 (honest negative)

### 6. Discussion (1-2 pages)
- Complexity-dependent improvement pattern
- Latency-quality trade-off analysis
- When agentic refinement is (not) worth it
- Practical implications for enterprise RAG
- **Limitations** (필수! 없으면 reject 사유)

### 7. Conclusion & Future Work (0.5-1 page)

---

## 4. Reviewer가 보는 것

### Accept 요인
- 명확하고 새로운 contribution
- 강한 실험 검증 + 적절한 baseline
- 통계적 유의성 제시
- 충실한 ablation study
- 재현성 (코드/데이터 공개)
- 최신 문헌 커버 (2024-2025)

### Reject 사유 (SciRev 기반)
1. 결과 분석/discussion 부족
2. Limitation 분석 없음
3. Baseline 비교 약함
4. 개선폭이 미미하거나 통계적으로 불유의
5. Scope 불일치 (desk reject)
6. Novelty 불명확

### 대응 전략
- "knowledge-based systems" 관점으로 포지셔닝 (단순 NLP 아님)
- 모든 주장에 citation 또는 실험적 evidence
- Limitation 섹션 반드시 포함
- Reviewer comment에 성실하고 존중하는 응답

---

## 5. Figures 계획

| # | Figure | 용도 |
|---|--------|------|
| 1 | System Architecture | 전체 프레임워크 |
| 2 | Pipeline Comparison | Naive/CRAG/Loop/Agentic 흐름 |
| 3 | ReAct Agent Workflow | Tool 호출 예시 |
| 4 | Main Results Chart | F1 across datasets |
| 5 | Tool Usage Distribution | RQ2 시각화 |
| 6 | DSPy Optimization | Manual vs Unopt vs Bootstrap vs MIPROv2 |
| 7 | Complexity vs Improvement | 핵심 narrative |

## 6. Tables 계획

| # | Table | 용도 |
|---|-------|------|
| 1 | Related Work Comparison | Feature 비교 매트릭스 |
| 2 | Dataset Statistics | 4 datasets 통계 |
| 3 | Main Results (RQ1) | 5 pipelines × 4 datasets × EM/F1/Judge |
| 4 | Tool Usage (RQ2) | Tool frequency per dataset |
| 5 | DSPy Results (RQ5) | 4 variants × 4 datasets |
| 6 | Eval Ablation (RQ3) | 3 variants × 4 datasets |
| 7 | Structure Ablation (RQ4) | 4 variants on FinanceBench |
| 8 | Latency Comparison | All pipelines latency |

---

## 7. Highlights 예시 (각 85자 이내)

1. `ReAct agent with 6 tools autonomously refines retrieval for multi-hop QA`
2. `Agentic approach improves F1 by 0.051-0.069 on complex multi-hop datasets`
3. `DSPy signatures alone boost F1 by 0.1-0.2 over manual prompt engineering`
4. `Comprehensive ablation reveals evaluation tool serves as quality gate`
5. `Evaluated on 4 datasets including enterprise financial documents`

---

## 8. 참고 논문 (KBS RAG/LLM 관련)

- "A comprehensive survey on integrating large language models with knowledge-based methods" (KBS, Apr 2025)
- "An advanced RAG system for manufacturing quality control" (KBS, 2024)
- SCMRAG: Self-Corrective Multihop RAG (AAMAS 2025) — 가장 직접적 competitor
- Agentic RAG Survey (arXiv 2501.09136)
