"""Answer Generation DSPy Signature.

Generates the final answer from accumulated passages using ChainOfThought
to produce step-by-step reasoning before the answer.
"""

import dspy


class QnAGenerateSignature(dspy.Signature):
    """Generate an answer based on the retrieved passages.

    Answer format rules:
    - For factoid questions (who/what/when/where/which/how many), answer with
      ONLY the key fact and its unit if applicable.
      Good: "1755", "Dutch", "Kevin Spacey", "40 members"
      Bad: "The answer is 1755"
    - For yes/no questions that ask for explanation ("If X, explain why"),
      answer with the verdict AND the explanation with specific numbers.
      Good: "No. Gross margins declined by 0.8% from 19.4% to 18.6%."
      Bad: "No"
    - For analytical questions (what drove X, how did Y change), provide
      the key finding with supporting data from the passages.
      Good: "Operating margin decreased 1.7% due to higher cost of sales
      and increased litigation expenses."
      Bad: "Operating margin decreased"
    - Match the answer's detail level to the question's complexity.
    - Put step-by-step reasoning in the chain-of-thought, NOT in the answer.
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
            "The answer, matching the question's expected detail level. "
            "For factoid questions: just the entity/number (e.g. '1755', 'Dutch'). "
            "For yes/no + explain questions: verdict + explanation with numbers. "
            "For analytical questions: finding + supporting data. "
            "NEVER start with 'The answer is', 'Based on', or filler."
        )
    )
    footnotes: str = dspy.OutputField(
        desc="Footnote references listing passage IDs and titles used in the answer."
    )
    recommended_questions: list[str] = dspy.OutputField(
        desc="Three recommended follow-up questions for further exploration."
    )
