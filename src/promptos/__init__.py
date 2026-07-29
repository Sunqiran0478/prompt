"""PromptOS: local, auditable prompt optimization for arbitrary tasks."""
from .core import PromptOptimizer, Sample, SignalSpec, TaskSpec
from .provenance import Annotation, apply_annotations

__all__ = ["Annotation", "PromptOptimizer", "Sample", "SignalSpec", "TaskSpec", "apply_annotations"]
