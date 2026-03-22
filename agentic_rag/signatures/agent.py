"""Agentic Retrieval Refinement Signature for dspy.ReAct.

Defines the DSPy Signature used by the ReAct agent to autonomously refine
retrieval quality through iterative tool use — searching documents,
browsing structure, mapping terminology, and evaluating passage quality.

The signature docstring (which serves as the ReAct prompt) is generated
dynamically to match the evaluation mode (4D / 1D / no eval), ensuring
the agent protocol is consistent with the tools actually available.
"""

import dspy

# ---------------------------------------------------------------------------
# Protocol fragments — composed based on evaluation mode
# ---------------------------------------------------------------------------

_HEADER = """\
Autonomously refine retrieval quality for a user question.

You are a retrieval refinement agent. Your goal is to find passages that \
score at or above the quality threshold on {scoring_description}.

IMPORTANT: Only use the tools listed in the tool enum. Do NOT attempt to \
call any tool not in the provided list — it will fail and waste a step.

Progressive information disclosure
===================================
search_passages returns PREVIEWS only (first 200 chars). To read full \
content, call get_passage_detail(passage_id) on promising passages. \
Use list_document_sections to browse document structure and identify \
relevant sections before searching.

Structured protocol (follow this order)
========================================
STEP 1: Call decompose_query to check if multi-hop.
STEP 2: Use list_document_sections (if available) to identify relevant \
document sections. Then call search_passages with initial query. \
If multi-hop, search EACH sub-question separately.
STEP 2b: Call get_passage_detail on the most relevant passages from \
search results to read their full content before deciding next steps."""

_EVAL_STEPS = """
STEP 3: Call evaluate_passages ONCE on the combined passage set \
(retry_count=0).
STEP 4: If total < quality_threshold:
        a. Use evaluation feedback (suggested_query, keywords_to_add).
        b. Call search_passages ONE more time with the refined query.
        c. Call evaluate_passages again with retry_count=1."""

_NO_EVAL_STEPS = """
STEP 3: Review the search results. If the passages seem insufficient \
for answering the question:
        a. Reformulate the query using different terms or sub-questions.
        b. Call search_passages ONE more time with the refined query."""

_FOOTER_WITH_EVAL = """
STEP {calc_step}: For numerical questions, use calculate() if available.
STEP {finish_step}: Call finish. Do NOT continue searching if:
        - You already searched 3+ times with no score improvement of 10+
        - You are retrieving the same or similar passages
        - Evaluation score >= quality_threshold

IMPORTANT: Limit yourself to at most 2 evaluate_passages calls and \
3 search_passages calls. Finish promptly with the best available passages."""

_FOOTER_NO_EVAL = """
STEP {calc_step}: For numerical questions, use calculate() if available.
STEP {finish_step}: Call finish. Do NOT continue searching if:
        - You already searched 3+ times
        - You are retrieving the same or similar passages

IMPORTANT: Limit yourself to at most 3 search_passages calls. \
Finish promptly with the best available passages."""

_SCORING_4D = (
    "a 4-dimensional assessment (Relevance, Coverage, Specificity, Sufficiency — total 0-100)"
)
_SCORING_1D = "an overall quality assessment (0-100)"
_SCORING_NO_EVAL = "an overall quality assessment (0-100)"


def _build_docstring(*, enable_4d: bool = True, has_evaluate: bool = True) -> str:
    """Build the agent signature docstring for the given evaluation mode."""
    if has_evaluate:
        scoring = _SCORING_4D if enable_4d else _SCORING_1D
        steps = _EVAL_STEPS
        footer = _FOOTER_WITH_EVAL.format(calc_step=5, finish_step=6)
    else:
        scoring = _SCORING_NO_EVAL
        steps = _NO_EVAL_STEPS
        footer = _FOOTER_NO_EVAL.format(calc_step=4, finish_step=5)

    return _HEADER.format(scoring_description=scoring) + steps + footer


def make_agent_signature(
    *, enable_4d: bool = True, has_evaluate: bool = True
) -> type[dspy.Signature]:
    """Create an AgenticRefinementSignature with mode-appropriate protocol.

    Args:
        enable_4d: True for 4D evaluation, False for 1D holistic scoring.
        has_evaluate: True if evaluate_passages tool is available.

    Returns:
        A dspy.Signature subclass with the correct docstring/protocol.
    """
    docstring = _build_docstring(enable_4d=enable_4d, has_evaluate=has_evaluate)

    class _AgenticRefinementSignature(dspy.Signature):
        __doc__ = docstring

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
            desc=(
                "ALL passage IDs found during search that may be relevant. "
                "Include every passage from search results — do not filter aggressively. "
                "More passages is better; the generation step will handle selection."
            )
        )
        final_action: str = dspy.OutputField(
            desc=(
                '"output" — always output the best available passages, even if quality '
                "is below threshold. Never leave final_passages empty."
            )
        )

    return _AgenticRefinementSignature


# Default export for backward compatibility (4D + evaluate)
AgenticRefinementSignature = make_agent_signature(enable_4d=True, has_evaluate=True)
