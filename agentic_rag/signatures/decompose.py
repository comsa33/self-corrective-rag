"""Query Decomposition DSPy Signature.

Decomposes complex multi-hop questions into simpler sub-questions
that can be answered independently and then synthesized.
"""

import dspy


class DecomposeQuerySignature(dspy.Signature):
    """Decompose a complex question into simpler sub-questions.

    Given a complex or multi-hop question, break it down into 2-4
    independent sub-questions that, when answered together, provide
    enough information to answer the original question.

    Rules:
    1. Each sub-question should be self-contained and answerable independently.
    2. Sub-questions should cover different aspects needed to answer the original.
    3. For simple, single-hop questions, return just the original question.
    4. Order sub-questions by dependency (answer earlier ones first if needed).
    """

    # --- Inputs ---
    question: str = dspy.InputField(desc="The complex or multi-hop question to decompose.")

    # --- Outputs ---
    is_multi_hop: bool = dspy.OutputField(
        desc="True if the question requires multiple reasoning steps, False for simple questions."
    )
    sub_questions: list[str] = dspy.OutputField(
        desc="List of 1-4 simpler sub-questions. Single-element list if not multi-hop."
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of the decomposition strategy.")
