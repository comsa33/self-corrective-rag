"""Agentic Retrieval Refinement Signature for dspy.ReAct.

Defines the DSPy Signature used by the ReAct agent to autonomously refine
retrieval quality through iterative tool use — searching documents,
browsing structure, mapping terminology, and evaluating passage quality.
"""

import dspy


class AgenticRefinementSignature(dspy.Signature):
    """Autonomously refine retrieval quality for a user question.

    You are a retrieval refinement agent with access to tools for searching
    an internal document corpus. Your goal is to find passages that score
    at or above the quality threshold on a 4-dimensional assessment
    (Relevance, Coverage, Specificity, Sufficiency — total 0-100).

    Available tools
    ===============
    - search_passages(query, top_k=10): Search documents with a query. Returns
      JSON list of {id, title, score}.
    - decompose_query(question): Decompose a complex multi-hop question into
      simpler sub-questions. Returns {is_multi_hop, sub_questions, reasoning}.
    - list_document_sections(keyword): Browse document table of contents.
      Returns matching sections with source and passage counts.
    - get_terminology(user_term): Map user language to document terminology.
      Returns a list of document-specific terms matching the user term.
    - evaluate_passages(question, passage_ids_json, retry_count=0): Run 4D
      quality evaluation on selected passages. Returns {relevance, coverage,
      specificity, sufficiency, total, action, suggested_query, keywords_to_add}.
    - get_passage_detail(passage_id): Read the full content of a passage.
    - calculate(expression): Evaluate a mathematical expression.
      Examples: "365 * 480 / 1200", "round((3502 - 3017) / 3017 * 100, 1)"

    Structured protocol (follow this order)
    ========================================
    STEP 1: Always start with decompose_query to check if multi-hop.
    STEP 2: Search with initial query. If multi-hop, search EACH sub-question
            separately to gather evidence for each reasoning step.
    STEP 3: Collect all passage IDs and call evaluate_passages ONCE on the
            combined set. Pass retry_count=0 for this first evaluation.
    STEP 4: If total < quality_threshold:
            a. Use evaluation feedback (suggested_query, keywords_to_add).
            b. Search ONE more time with the refined query.
            c. Call evaluate_passages again with retry_count=1.
    STEP 5: For numerical questions, use calculate() with extracted figures.
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
