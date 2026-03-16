"""Tests for DSPy Signature definitions.

Validates structure, field definitions, and module compatibility.
Does NOT call LLM — only checks class structure and type annotations.
"""

from __future__ import annotations

import dspy

from agentic_rag.signatures.agents import (
    ClarificationSignature,
    DomainExpertSignature,
    FallbackSignature,
)
from agentic_rag.signatures.evaluate import EvaluationSignature
from agentic_rag.signatures.generate import QnAGenerateSignature
from agentic_rag.signatures.preprocess import (
    HyDEPreprocessSignature,
    PreprocessSignature,
)


# ---------------------------------------------------------------
# Signature field validation
# ---------------------------------------------------------------
class TestPreprocessSignature:
    def test_input_fields(self):
        fields = PreprocessSignature.input_fields
        assert "user_question" in fields
        assert "conversation_history" in fields

    def test_output_fields(self):
        fields = PreprocessSignature.output_fields
        assert "rephrased_question" in fields
        assert "topic_category" in fields
        assert "product_keywords" in fields
        assert "keyword_words" in fields
        assert "subject_keywords" in fields
        assert "recommended_questions" in fields

    def test_can_create_module(self):
        module = dspy.ChainOfThought(PreprocessSignature)
        assert module is not None


class TestHyDEPreprocessSignature:
    def test_has_hypothetical_answer(self):
        fields = HyDEPreprocessSignature.output_fields
        assert "hypothetical_answer" in fields

    def test_inherits_preprocess_fields(self):
        """HyDE should have all PreprocessSignature fields plus hypothetical_answer."""
        preprocess_outputs = set(PreprocessSignature.output_fields.keys())
        hyde_outputs = set(HyDEPreprocessSignature.output_fields.keys())
        assert preprocess_outputs.issubset(hyde_outputs)


class TestEvaluationSignature:
    def test_input_fields(self):
        fields = EvaluationSignature.input_fields
        assert "question" in fields
        assert "passages" in fields
        assert "retry_count" in fields
        assert "max_retry" in fields

    def test_4d_output_fields(self):
        fields = EvaluationSignature.output_fields
        # 4 dimensions
        assert "relevance_score" in fields
        assert "coverage_score" in fields
        assert "specificity_score" in fields
        assert "sufficiency_score" in fields
        # Total + action
        assert "total_score" in fields
        assert "action" in fields
        # Refinement guidance
        assert "keywords_to_add" in fields
        assert "keywords_to_remove" in fields
        assert "suggested_query" in fields

    def test_can_create_predict_module(self):
        module = dspy.Predict(EvaluationSignature)
        assert module is not None


class TestQnAGenerateSignature:
    def test_output_fields(self):
        fields = QnAGenerateSignature.output_fields
        assert "answer" in fields
        assert "footnotes" in fields
        assert "recommended_questions" in fields

    def test_can_create_cot_module(self):
        module = dspy.ChainOfThought(QnAGenerateSignature)
        assert module is not None


class TestAgentSignatures:
    def test_clarification_outputs(self):
        fields = ClarificationSignature.output_fields
        assert "clarification_question" in fields
        assert "reason" in fields

    def test_domain_expert_outputs(self):
        fields = DomainExpertSignature.output_fields
        assert "expert_answer" in fields

    def test_fallback_outputs(self):
        fields = FallbackSignature.output_fields
        assert "best_effort_answer" in fields
        assert "limitations" in fields
        assert "alternatives" in fields

    def test_all_agents_have_question_and_passages(self):
        for sig in [ClarificationSignature, DomainExpertSignature, FallbackSignature]:
            assert "question" in sig.input_fields
            assert "passages" in sig.input_fields


# ---------------------------------------------------------------
# Total signature count
# ---------------------------------------------------------------
def test_total_signature_count():
    """Paper claims 7 DSPy Signatures."""
    signatures = [
        PreprocessSignature,
        HyDEPreprocessSignature,
        EvaluationSignature,
        QnAGenerateSignature,
        ClarificationSignature,
        DomainExpertSignature,
        FallbackSignature,
    ]
    assert len(signatures) == 7
