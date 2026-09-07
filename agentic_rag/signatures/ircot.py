"""IRCoT reasoning-step signature (Trivedi et al., ACL 2023).

One interleaving step: given the question and the paragraphs collected so
far, write the *next* chain-of-thought sentence. The sentence is then used
as the next retrieval query. The paper's reader stops when a sentence
contains "answer is:", so the instruction asks for exactly that form once
the answer is known.
"""

from __future__ import annotations

import dspy


class IRCoTStepSignature(dspy.Signature):
    """Write the next sentence of a step-by-step reasoning chain.

    Rules:
    - Write exactly ONE sentence that advances the reasoning using the
      passages; name the specific fact the next step still needs.
    - Do not repeat sentences already in the reasoning so far.
    - When the passages already support the final answer, write the sentence
      in the form "So the answer is: <answer>." and nothing else.
    """

    question: str = dspy.InputField(desc="The multi-hop question.")
    passages: str = dspy.InputField(desc="Paragraphs collected so far, formatted as context.")
    reasoning_so_far: str = dspy.InputField(
        desc="Chain-of-thought sentences generated in earlier steps, in order (may be empty)."
    )
    next_sentence: str = dspy.OutputField(
        desc="The next reasoning sentence, or 'So the answer is: <answer>.' when done."
    )
