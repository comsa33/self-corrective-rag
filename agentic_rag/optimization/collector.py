"""Training data collector for DSPy optimizer.

Collects successful pipeline executions as training examples
for BootstrapFewShot and MIPROv2 optimizers.
Mirrors the DspyCollector from the original system.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class TrainingExample:
    """A single training example for DSPy optimization."""

    signature_name: str  # e.g., "PreprocessSignature", "EvaluationSignature"
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class TrainingCollector:
    """Collect and manage training examples for DSPy optimizers."""

    def __init__(self):
        self.examples: dict[str, list[TrainingExample]] = {}

    def add(
        self,
        signature_name: str,
        inputs: dict,
        outputs: dict,
        **metadata,
    ) -> None:
        """Record a successful execution as a training example."""
        ex = TrainingExample(
            signature_name=signature_name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        self.examples.setdefault(signature_name, []).append(ex)

    def get_examples(self, signature_name: str) -> list[TrainingExample]:
        """Get all examples for a specific signature."""
        return self.examples.get(signature_name, [])

    def to_dspy_examples(self, signature_name: str) -> list:
        """Convert collected examples to dspy.Example format."""
        import dspy

        examples = []
        for ex in self.get_examples(signature_name):
            data = {**ex.inputs, **ex.outputs}
            example = dspy.Example(**data).with_inputs(*ex.inputs.keys())
            examples.append(example)
        return examples

    @property
    def total_count(self) -> int:
        return sum(len(v) for v in self.examples.values())

    def summary(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.examples.items()}

    def save(self, path: Path) -> None:
        """Save collected examples to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {sig: [asdict(ex) for ex in exs] for sig, exs in self.examples.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {self.total_count} training examples to {path}")

    def load(self, path: Path) -> None:
        """Load examples from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.examples = {sig: [TrainingExample(**ex) for ex in exs] for sig, exs in data.items()}
        logger.info(f"Loaded {self.total_count} training examples from {path}")
