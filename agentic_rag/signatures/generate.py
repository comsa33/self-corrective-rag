"""Answer Generation DSPy Signature.

Generates the final answer from accumulated passages using ChainOfThought
to produce step-by-step reasoning before the answer.
"""

import dspy


class QnAGenerateSignature(dspy.Signature):
    """Generate a comprehensive answer based on the retrieved passages.

    The answer must:
    1. Be grounded in the provided passages (no hallucination).
    2. Include footnote references to source passages used.
    3. Suggest follow-up questions for deeper exploration.

    Uses ChainOfThought to produce explicit reasoning before the answer.
    """

    # --- Inputs ---
    question: str = dspy.InputField(desc="The user's question (rephrased, standalone).")
    passages: str = dspy.InputField(desc="Retrieved passages formatted as context string.")
    system_prompt: str = dspy.InputField(
        desc="System-level instructions guiding the answer style and scope.",
        default="You are a helpful knowledge assistant. Answer based on the provided passages.",
    )

    # --- Outputs ---
    answer: str = dspy.OutputField(desc="The final comprehensive answer grounded in the passages.")
    footnotes: str = dspy.OutputField(
        desc="Footnote references listing passage IDs and titles used in the answer."
    )
    recommended_questions: list[str] = dspy.OutputField(
        desc="Three recommended follow-up questions for further exploration."
    )
