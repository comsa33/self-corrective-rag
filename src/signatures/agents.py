"""Agent Routing DSPy Signatures.

Three agent types handle cases where retrieval quality remains insufficient
after max retries (Contribution C4: 3-Way Agent Routing):

  1. ClarificationSignature — ambiguous questions → ask for clarification
  2. DomainExpertSignature  — technical questions → expert-level answer
  3. FallbackSignature      — other cases → best-effort + acknowledge limits
"""

import dspy


class ClarificationSignature(dspy.Signature):
    """Generate a clarification question when the user's intent is ambiguous.

    Used when evaluation determines the question could have multiple
    interpretations or lacks sufficient specificity for accurate retrieval.
    """

    # --- Inputs ---
    question: str = dspy.InputField(desc="The user's original/rephrased question.")
    passages: str = dspy.InputField(desc="Retrieved passages (may be partially relevant).")

    # --- Outputs ---
    clarification_question: str = dspy.OutputField(
        desc="A single, specific clarification question to disambiguate the user's intent."
    )
    reason: str = dspy.OutputField(desc="Explanation of why the question is ambiguous or unclear.")


class DomainExpertSignature(dspy.Signature):
    """Provide an expert-level technical answer when retrieval is insufficient.

    Activated for technical questions (DSL, API, architecture, DevOps)
    where the LLM's parametric knowledge can supplement retrieved passages.
    Uses ChainOfThought for structured reasoning.
    """

    # --- Inputs ---
    question: str = dspy.InputField(desc="The user's technical question.")
    passages: str = dspy.InputField(desc="Retrieved passages (may be partially relevant).")

    # --- Outputs ---
    expert_answer: str = dspy.OutputField(
        desc=(
            "Expert-level answer incorporating domain knowledge. "
            "May include code examples, architecture diagrams, "
            "and implementation details."
        )
    )


class FallbackSignature(dspy.Signature):
    """Provide a best-effort answer while acknowledging limitations.

    Used as the last resort when neither retrieval nor domain expertise
    can fully address the question. Transparently communicates what
    can and cannot be answered, and suggests alternative resources.
    """

    # --- Inputs ---
    question: str = dspy.InputField(desc="The user's question.")
    passages: str = dspy.InputField(desc="Retrieved passages (may be low relevance).")

    # --- Outputs ---
    best_effort_answer: str = dspy.OutputField(
        desc="The best possible answer given available information."
    )
    limitations: str = dspy.OutputField(
        desc="Honest description of what could not be answered and why."
    )
    alternatives: str = dspy.OutputField(
        desc="Suggested alternative resources, documentation, or next steps."
    )
