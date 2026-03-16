"""Human evaluation protocol for RAG quality assessment.

Provides structured templates for human evaluators to rate
answer quality on multiple dimensions, following standard
NLP evaluation practices for knowledge-intensive tasks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class HumanEvalItem:
    """A single item for human evaluation."""

    id: str
    question: str
    reference_answer: str
    # One entry per pipeline variant
    predictions: dict[str, str] = field(default_factory=dict)
    # Human scores per variant per dimension
    scores: dict[str, dict[str, int]] = field(default_factory=dict)
    notes: str = ""


@dataclass
class HumanEvalDimension:
    """Definition of an evaluation dimension."""

    name: str
    description: str
    scale_min: int
    scale_max: int
    guidelines: str


# Standard evaluation dimensions
EVAL_DIMENSIONS = [
    HumanEvalDimension(
        name="correctness",
        description="Factual accuracy of the answer",
        scale_min=1,
        scale_max=5,
        guidelines=(
            "1: Completely incorrect or irrelevant\n"
            "2: Contains major factual errors\n"
            "3: Partially correct with some errors\n"
            "4: Mostly correct with minor issues\n"
            "5: Fully correct and accurate"
        ),
    ),
    HumanEvalDimension(
        name="completeness",
        description="How thoroughly the answer covers the question",
        scale_min=1,
        scale_max=5,
        guidelines=(
            "1: Barely addresses the question\n"
            "2: Covers only one aspect\n"
            "3: Covers main points but misses some\n"
            "4: Covers most aspects adequately\n"
            "5: Comprehensive, covers all aspects"
        ),
    ),
    HumanEvalDimension(
        name="relevance",
        description="How relevant the answer is to the question",
        scale_min=1,
        scale_max=5,
        guidelines=(
            "1: Completely off-topic\n"
            "2: Tangentially related\n"
            "3: Somewhat relevant\n"
            "4: Directly relevant\n"
            "5: Precisely addresses the question"
        ),
    ),
    HumanEvalDimension(
        name="faithfulness",
        description="Whether the answer is grounded in provided sources",
        scale_min=1,
        scale_max=5,
        guidelines=(
            "1: Contains fabricated information\n"
            "2: Mostly unsupported claims\n"
            "3: Mix of supported and unsupported\n"
            "4: Mostly grounded in sources\n"
            "5: Fully grounded, no hallucination"
        ),
    ),
    HumanEvalDimension(
        name="usefulness",
        description="Practical usefulness of the answer to the user",
        scale_min=1,
        scale_max=5,
        guidelines=(
            "1: Not useful at all\n"
            "2: Slightly useful\n"
            "3: Moderately useful\n"
            "4: Very useful\n"
            "5: Extremely useful, actionable"
        ),
    ),
]


class HumanEvalProtocol:
    """Generate and manage human evaluation sheets."""

    def __init__(
        self,
        dimensions: list[HumanEvalDimension] | None = None,
    ):
        self.dimensions = dimensions or EVAL_DIMENSIONS
        self.items: list[HumanEvalItem] = []

    def add_item(
        self,
        id: str,
        question: str,
        reference_answer: str,
        predictions: dict[str, str],
    ) -> None:
        """Add an evaluation item with predictions from multiple pipelines."""
        self.items.append(
            HumanEvalItem(
                id=id,
                question=question,
                reference_answer=reference_answer,
                predictions=predictions,
            )
        )

    def export_sheet(self, path: Path) -> None:
        """Export evaluation sheet as JSON for evaluators."""
        sheet = {
            "instructions": self._instructions(),
            "dimensions": [asdict(d) for d in self.dimensions],
            "items": [asdict(item) for item in self.items],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sheet, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(self.items)} items to {path}")

    def load_completed(self, path: Path) -> None:
        """Load a completed evaluation sheet."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.items = [HumanEvalItem(**item) for item in data.get("items", [])]

    def compute_agreement(self, other_path: Path) -> dict:
        """Compute inter-annotator agreement (Cohen's kappa placeholder)."""
        # Load second evaluator's scores
        with open(other_path, encoding="utf-8") as f:
            other_data = json.load(f)

        # Basic agreement computation
        agreements = []
        for item1, item2_data in zip(self.items, other_data.get("items", []), strict=False):
            for variant in item1.scores:
                for dim in item1.scores[variant]:
                    s1 = item1.scores[variant][dim]
                    s2 = item2_data.get("scores", {}).get(variant, {}).get(dim)
                    if s2 is not None:
                        agreements.append(abs(s1 - s2) <= 1)

        if agreements:
            return {
                "exact_agreement": sum(1 for a in agreements if a) / len(agreements),
                "total_comparisons": len(agreements),
            }
        return {"exact_agreement": 0.0, "total_comparisons": 0}

    def aggregate_scores(self) -> dict[str, dict[str, float]]:
        """Compute mean scores per pipeline variant per dimension."""
        from collections import defaultdict

        import numpy as np

        scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

        for item in self.items:
            for variant, dims in item.scores.items():
                for dim, score in dims.items():
                    scores[variant][dim].append(score)

        return {
            variant: {dim: float(np.mean(vals)) for dim, vals in dims.items()}
            for variant, dims in scores.items()
        }

    @staticmethod
    def _instructions() -> str:
        return (
            "Human Evaluation Instructions\n"
            "==============================\n"
            "For each question, rate every pipeline variant on each dimension.\n"
            "Use the provided scale and guidelines.\n"
            "Evaluate INDEPENDENTLY — do not compare variants while scoring.\n"
            "Record any notes about edge cases or disagreements.\n"
            "Variants are anonymized; do not try to identify the method."
        )
