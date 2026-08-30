"""Evaluation harness: measures critique precision, not critique quantity."""

from .evaluator import evaluate, render_markdown
from .models import EvaluationReport, GoldClaim, GoldStandard, MetricResult

__all__ = [
    "evaluate",
    "render_markdown",
    "GoldStandard",
    "GoldClaim",
    "EvaluationReport",
    "MetricResult",
]
