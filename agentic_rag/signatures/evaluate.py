"""4-Dimensional Quality Evaluation DSPy Signature.

Core contribution (C2): replaces single-score relevance judgement with
a 4D assessment — Relevance, Coverage, Specificity, Sufficiency — that
drives the self-corrective loop decision (output / refine / route_to_agent).
"""

import dspy


class EvaluationSignature(dspy.Signature):
    """Evaluate retrieved passages on 4 quality dimensions and decide the next action.

    Scoring Guidelines
    ==================
    ■ Relevance (0-30):
      - At least one passage related to the topic → 20+
      - Passages directly answerable → 25+
      - Multiple perfectly relevant passages → 30

    ■ Coverage (0-25):
      - Covers at least one key aspect → 15+
      - Covers most key aspects → 20+
      - Covers all aspects → 25

    ■ Specificity (0-25):
      - Only general descriptions → 10
      - Contains concrete info (names, dates, facts, code, API) → 15+
      - Detailed examples or configuration values → 20+

    ■ Sufficiency (0-20):
      - Partial answer possible from passages → 12+
      - Sufficient answer possible → 16+
      - Complete answer possible → 20
      - IMPORTANT: For factoid questions (who/what/when/where/which),
        if ANY passage contains the answer entity, score sufficiency >= 16.

    Decision Rules (Progressive Leniency)
    ======================================
    The effective threshold decreases with each retry to avoid over-iteration:
      effective_threshold = QUALITY_THRESHOLD - (retry_count * 5)

    - total_score >= effective_threshold → action = "output"
    - total_score < effective_threshold AND retry_count < max_retry → action = "refine"
    - retry_count >= max_retry → action = "output" (always generate on final retry)

    Example: If QUALITY_THRESHOLD=40, retry_count=2:
      effective_threshold = 40 - 10 = 30. Score 32 → "output".

    When action is "refine", provide targeted feedback:
      - keywords_to_add:    terms that should be included in the next search
      - keywords_to_remove: noisy terms that hurt retrieval precision
      - suggested_query:    an improved search query for the next iteration
    """

    # --- Inputs ---
    question: str = dspy.InputField(desc="The user's question (rephrased, standalone).")
    passages: str = dspy.InputField(desc="Retrieved passages formatted as context string.")
    retry_count: int = dspy.InputField(desc="Current retry iteration (0-based).")
    max_retry: int = dspy.InputField(desc="Maximum allowed retries (e.g. 3).")

    # --- Outputs: 4D scores ---
    relevance_score: int = dspy.OutputField(
        desc="Relevance score (0-30): topical relevance of passages."
    )
    coverage_score: int = dspy.OutputField(
        desc="Coverage score (0-25): breadth of key aspects covered."
    )
    specificity_score: int = dspy.OutputField(
        desc="Specificity score (0-25): concreteness and detail level."
    )
    sufficiency_score: int = dspy.OutputField(
        desc="Sufficiency score (0-20): ability to generate a complete answer."
    )
    total_score: int = dspy.OutputField(
        desc="Total quality score (0-100): sum of the 4 dimension scores."
    )

    # --- Outputs: action decision ---
    action: str = dspy.OutputField(desc='Next action: "output" | "refine" | "route_to_agent".')

    # --- Outputs: refinement feedback (used when action == "refine") ---
    keywords_to_add: list[str] = dspy.OutputField(
        desc="Keywords to add for the next retrieval attempt."
    )
    keywords_to_remove: list[str] = dspy.OutputField(
        desc="Noisy keywords to remove for the next retrieval attempt."
    )
    suggested_query: str = dspy.OutputField(
        desc="An improved search query for the next retrieval iteration."
    )

    # --- Outputs: reasoning ---
    reasoning: str = dspy.OutputField(
        desc="Explanation of the evaluation rationale and action decision."
    )
