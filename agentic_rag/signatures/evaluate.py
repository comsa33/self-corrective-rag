"""4-Dimensional Quality Evaluation DSPy Signature.

Core contribution (C2): replaces single-score relevance judgement with
a 4D assessment — Relevance, Coverage, Specificity, Sufficiency — that
drives the self-corrective loop decision (output / refine / route_to_agent).
"""

import dspy


class EvaluationSignature(dspy.Signature):
    """Evaluate retrieved passages on 4 quality dimensions and decide the next action.

    Scoring Guidelines — Be strict and calibrated
    ================================================
    Score each dimension independently. Do NOT default to high scores.
    Ask yourself: "Can I actually answer the question from these passages?"

    ■ Relevance (0-30):
      - No passage mentions the topic at all → 0-5
      - Some passages tangentially related → 10-15
      - At least one passage directly about the topic → 16-22
      - Multiple passages directly relevant with key entities → 23-27
      - Perfect coverage of all relevant entities/topics → 28-30

    ■ Coverage (0-25):
      - Missing most key aspects needed to answer → 0-8
      - Covers one key aspect but misses others → 9-14
      - Covers most key aspects with some gaps → 15-19
      - Covers all key aspects of the question → 20-25

    ■ Specificity (0-25):
      - Only vague or general descriptions → 0-8
      - Some concrete info but missing key details → 9-14
      - Contains specific facts (names, dates, numbers) → 15-19
      - Detailed, precise information directly answering the question → 20-25

    ■ Sufficiency (0-20):
      - Cannot answer the question from these passages → 0-5
      - Partial answer possible but significant gaps remain → 6-11
      - Sufficient for a reasonable answer with minor gaps → 12-16
      - Complete, confident answer possible → 17-20

    CALIBRATION RULES — Read carefully before scoring:
    - Score of 90-100: ONLY if every part of the question is directly and
      explicitly answered with specific evidence in the passages. This is
      exceptionally rare.
    - Score of 70-89: Passages contain the answer but with minor gaps,
      ambiguity, or missing supporting details.
    - Score of 50-69: Passages are relevant but incomplete — some key
      information is missing or only indirectly available.
    - Score of 30-49: Passages are tangentially related but cannot fully
      answer the question.
    - Score of 0-29: Passages are irrelevant or nearly useless.

    Ask yourself: "If I gave only these passages to a person, could they
    write a complete, accurate answer?" Score based on that honestly.

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


class Evaluation1DSignature(dspy.Signature):
    """Evaluate retrieved passages with a single overall quality score.

    Score the passages on a 0-100 scale based on how well they can
    answer the question. Consider relevance, completeness, and specificity
    holistically — do NOT break into sub-dimensions.

    CALIBRATION RULES — Read carefully before scoring:
    - Score of 90-100: ONLY if every part of the question is directly and
      explicitly answered with specific evidence. Exceptionally rare.
    - Score of 70-89: Passages contain the answer but with minor gaps.
    - Score of 50-69: Relevant but incomplete — some key info missing.
    - Score of 30-49: Tangentially related but cannot fully answer.
    - Score of 0-29: Irrelevant or nearly useless.

    Ask yourself: "If I gave only these passages to a person, could they
    write a complete, accurate answer?" Score based on that honestly.

    Decision Rules (Progressive Leniency)
    ======================================
    The effective threshold decreases with each retry to avoid over-iteration:
      effective_threshold = QUALITY_THRESHOLD - (retry_count * 5)

    - total_score >= effective_threshold → action = "output"
    - total_score < effective_threshold AND retry_count < max_retry → action = "refine"
    - retry_count >= max_retry → action = "output" (always generate on final retry)

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

    # --- Outputs ---
    total_score: int = dspy.OutputField(
        desc="Overall quality score (0-100): holistic assessment of passage quality."
    )
    action: str = dspy.OutputField(desc='Next action: "output" | "refine".')

    # --- Outputs: refinement feedback (same as 4D, enables fair comparison) ---
    keywords_to_add: list[str] = dspy.OutputField(
        desc="Keywords to add for the next retrieval attempt."
    )
    keywords_to_remove: list[str] = dspy.OutputField(
        desc="Noisy keywords to remove for the next retrieval attempt."
    )
    suggested_query: str = dspy.OutputField(
        desc="An improved search query for the next retrieval iteration."
    )

    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why passages are sufficient or insufficient."
    )
