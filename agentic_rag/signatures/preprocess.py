"""Preprocessing DSPy Signatures.

Two variants:
  - PreprocessSignature:     standard question rephrasing + keyword extraction
  - HyDEPreprocessSignature: adds Hypothetical Document Embedding generation
"""

import dspy


class PreprocessSignature(dspy.Signature):
    """Rephrase the user question into a standalone form and extract search keywords.

    Given a user question and optional conversation history, produce:
    1. A self-contained rephrased question (no pronouns referencing history).
    2. A topic category from the predefined list.
    3. Three keyword groups for hybrid search (product, core, subject).
    4. Three recommended follow-up questions.
    """

    # --- Inputs ---
    user_question: str = dspy.InputField(desc="The user's original question.")
    conversation_history: str = dspy.InputField(
        desc="Previous conversation turns as a JSON string. Empty string if none.",
        default="",
    )

    # --- Outputs ---
    rephrased_question: str = dspy.OutputField(
        desc="The question rephrased as a standalone, self-contained query."
    )
    topic_category: str = dspy.OutputField(
        desc=(
            "One of the 13 predefined categories: "
            "일반, DSL 함수, 시나리오, API, 아키텍처, DevOps, "
            "데이터, 보안, 성능, UI/UX, 통합, 트러블슈팅, 기타"
        )
    )
    product_keywords: list[str] = dspy.OutputField(
        desc="Keywords related to specific products or services."
    )
    keyword_words: list[str] = dspy.OutputField(
        desc="Core search keywords extracted from the question."
    )
    subject_keywords: list[str] = dspy.OutputField(
        desc="Subject-level keywords for topic matching."
    )
    recommended_questions: list[str] = dspy.OutputField(
        desc="Three recommended follow-up questions."
    )


class HyDEPreprocessSignature(dspy.Signature):
    """Rephrase, extract keywords, AND generate a hypothetical answer (HyDE).

    Extends PreprocessSignature by producing a plausible hypothetical answer
    that will be embedded and used as the vector search query, improving
    retrieval recall for complex or abstract questions.
    """

    # --- Inputs ---
    user_question: str = dspy.InputField(desc="The user's original question.")
    conversation_history: str = dspy.InputField(
        desc="Previous conversation turns as a JSON string. Empty string if none.",
        default="",
    )

    # --- Outputs (same as PreprocessSignature + hypothetical_answer) ---
    rephrased_question: str = dspy.OutputField(
        desc="The question rephrased as a standalone, self-contained query."
    )
    topic_category: str = dspy.OutputField(
        desc=(
            "One of the 13 predefined categories: "
            "일반, DSL 함수, 시나리오, API, 아키텍처, DevOps, "
            "데이터, 보안, 성능, UI/UX, 통합, 트러블슈팅, 기타"
        )
    )
    product_keywords: list[str] = dspy.OutputField(
        desc="Keywords related to specific products or services."
    )
    keyword_words: list[str] = dspy.OutputField(
        desc="Core search keywords extracted from the question."
    )
    subject_keywords: list[str] = dspy.OutputField(
        desc="Subject-level keywords for topic matching."
    )
    recommended_questions: list[str] = dspy.OutputField(
        desc="Three recommended follow-up questions."
    )
    hypothetical_answer: str = dspy.OutputField(
        desc=(
            "A plausible hypothetical answer to the question. "
            "Used as the embedding query for dense retrieval (HyDE technique)."
        )
    )
