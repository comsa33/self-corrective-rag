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
      JSON list of {id, title, content_preview, score}.
    - decompose_query(question): Decompose a complex multi-hop question into
      simpler sub-questions. Returns {is_multi_hop, sub_questions, reasoning}.
      Use this early for questions that involve multiple entities or reasoning steps.
    - list_document_sections(keyword): Browse document table of contents.
      Returns matching sections with source and passage counts.
    - get_terminology(user_term): Map user language to document terminology.
      Returns a list of document-specific terms matching the user term.
    - evaluate_passages(question, passage_ids_json): Run 4D quality evaluation
      on selected passages. Returns {relevance, coverage, specificity,
      sufficiency, total, action, reasoning, suggested_query, keywords_to_add}.
    - get_passage_detail(passage_id): Read the full content of a passage.
    - calculate(expression): Evaluate a mathematical expression and return the
      numeric result. Use for financial ratios, growth rates, percentages, or
      any computation needed to answer the question.
      Examples: "365 * 480 / 1200", "round((3502 - 3017) / 3017 * 100, 1)"

    Strategy guidelines
    ===================
    1. For complex questions, start with decompose_query to identify sub-questions.
    2. Search with the initial query and keywords. For multi-hop questions,
       search each sub-question separately.
    3. Evaluate the retrieved passages with evaluate_passages.
    4. If quality is insufficient (total < quality_threshold):
       a. Use the evaluation feedback (suggested_query, keywords_to_add) to
          construct improved queries.
       b. Use list_document_sections to discover document structure.
       c. Use get_terminology to find correct document-specific terms.
       d. Search again with refined queries incorporating discovered terms.
    5. Accumulate promising passages across searches (up to max_passages).
       Drop low-quality passages if you exceed the limit.
    6. For questions requiring numerical computation (ratios, growth rates,
       percentages), extract the relevant figures from passages and use
       calculate() to compute the final answer before calling finish.
    7. When evaluation passes or you cannot improve further, call finish.
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
