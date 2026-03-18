"""Answer Generation DSPy Signature.

Generates the final answer from accumulated passages using ChainOfThought
to produce step-by-step reasoning before the answer.
"""

import dspy


class QnAGenerateSignature(dspy.Signature):
    """Generate an accurate, concise answer based on the retrieved passages.

    The answer must:
    1. Be grounded in the provided passages (no hallucination).
    2. For factoid questions (who/what/when/where), give a short direct answer
       (a name, year, place, or brief phrase — not a lengthy explanation).
    3. For complex questions, provide a focused 1-2 sentence answer.
    4. Include footnote references to source passages used.
    5. Suggest follow-up questions for deeper exploration.

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
    answer: str = dspy.OutputField(
        desc=(
            "The answer grounded in the passages. "
            "For factoid questions (who, what, when, where), respond with a concise phrase or "
            "short sentence containing just the key fact (e.g. '1755', 'Dutch', 'Marie Curie'). "
            "Avoid unnecessary elaboration for simple factoid questions."
        )
    )
    footnotes: str = dspy.OutputField(
        desc="Footnote references listing passage IDs and titles used in the answer."
    )
    recommended_questions: list[str] = dspy.OutputField(
        desc="Three recommended follow-up questions for further exploration."
    )
