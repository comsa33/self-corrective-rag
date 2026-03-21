"""Agentic Retrieval Refinement Signature for dspy.ReAct.

Defines the DSPy Signature used by the ReAct agent to autonomously refine
retrieval quality through iterative tool use — searching documents,
browsing structure, mapping terminology, and evaluating passage quality.
"""

import dspy


class AgenticRefinementSignature(dspy.Signature):
    """Autonomously refine retrieval quality for a user question.

    You are a retrieval refinement agent. Your goal is to find passages that
    score at or above the quality threshold on a 4-dimensional assessment
    (Relevance, Coverage, Specificity, Sufficiency — total 0-100).

    IMPORTANT: Only use the tools listed in the tool enum. Do NOT attempt to
    call any tool not in the provided list — it will fail and waste a step.

    Structured protocol (follow this order)
    ========================================
    STEP 1: Call decompose_query to check if multi-hop.
    STEP 2: Call search_passages with initial query. If multi-hop, search
            EACH sub-question separately.
    STEP 3: Call evaluate_passages ONCE on the combined passage set
            (retry_count=0).
    STEP 4: If total < quality_threshold:
            a. Use evaluation feedback (suggested_query, keywords_to_add).
            b. Call search_passages ONE more time with the refined query.
            c. Call evaluate_passages again with retry_count=1.
    STEP 5: For numerical questions, use calculate() if available.
    STEP 6: Call finish. Do NOT continue searching if:
            - You already searched 3+ times with no score improvement of 10+
            - You are retrieving the same or similar passages
            - Evaluation score >= quality_threshold

    IMPORTANT: Limit yourself to at most 2 evaluate_passages calls and
    3 search_passages calls. Finish promptly with the best available passages.
    """

    # --- Inputs ---
    question: str = dspy.InputField(desc="The user's question (rephrased, standalone).")
    initial_query: str = dspy.InputField(desc="Initial search query from preprocessing.")
    initial_keywords: list[str] = dspy.InputField(
        desc="Initial search keywords from preprocessing."
    )
    quality_threshold: int = dspy.InputField(
        desc="Minimum total score (0-100) for passage quality."
    )
    max_passages: int = dspy.InputField(desc="Maximum number of passages to accumulate.")

    # --- Outputs (extracted by ReAct after finish) ---
    final_passages: list[str] = dspy.OutputField(
        desc="List of selected passage IDs for answer generation."
    )
    final_action: str = dspy.OutputField(
        desc='"output" if quality is sufficient, "route_to_agent" if unable to find adequate passages.'
    )
