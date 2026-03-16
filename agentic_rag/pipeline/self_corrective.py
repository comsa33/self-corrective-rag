"""Backward-compatibility shim.

Maps the old SelfCorrectiveRAGPipeline name to the new split:
- LoopRAGPipeline (for-loop refinement, C1-C5)
- AgenticRAGPipeline (RLM refinement, proposed method)

SelfCorrectiveRAGPipeline is kept as an alias for LoopRAGPipeline
so that existing experiment scripts continue to work during migration.
It will be removed after Phase 4 (experiment restructuring).
"""

from agentic_rag.pipeline.agentic import AgenticRAGPipeline
from agentic_rag.pipeline.loop import LoopRAGPipeline

# Backward compat: old name → loop variant
SelfCorrectiveRAGPipeline = LoopRAGPipeline

__all__ = [
    "AgenticRAGPipeline",
    "LoopRAGPipeline",
    "SelfCorrectiveRAGPipeline",
]
