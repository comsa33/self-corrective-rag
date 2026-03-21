"""Answer Generation DSPy Signature.

Generates the final answer from accumulated passages using ChainOfThought
to produce step-by-step reasoning before the answer.
"""

import dspy


class QnAGenerateSignature(dspy.Signature):
    """Generate a concise answer based on the retrieved passages.

    Answer format rules:
    - For factoid questions (who/what/when/where/which/how many), answer with
      ONLY the key fact and its unit if applicable.
      Good: "1755", "Dutch", "Kevin Spacey", "40 members", "Greyia"
      Bad: "The answer is 1755", "Greyia has three species while Calibanus..."
    - For comparison questions ("which X is Y-er"), answer with just the entity name.
    - Put ALL reasoning in the chain-of-thought, NOT in the answer field.
    - The answer must be grounded in the passages (no hallucination).
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
            "ONLY the bare factual answer. "
            "For factoid questions: just the entity/number/name (e.g. '1755', 'Dutch', "
            "'Kevin Spacey', 'Greyia'). "
            "NEVER start with 'The', 'Based on', or any filler. Just the answer itself."
        )
    )
    footnotes: str = dspy.OutputField(
        desc="Footnote references listing passage IDs and titles used in the answer."
    )
    recommended_questions: list[str] = dspy.OutputField(
        desc="Three recommended follow-up questions for further exploration."
    )
