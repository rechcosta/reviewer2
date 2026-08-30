"""Data structures for the evaluation harness.

A *gold standard* file describes what a human reviewer expects for one
video: which claims should be extracted, and how each of them should be
classified. Reviewer2's own output is then scored against it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..errors import Reviewer2Error
from ..models import Classification, Severity


class GoldClaim(BaseModel):
    """One claim a human expects Reviewer2 to find and how to judge it."""

    text: str
    classification: Classification = Classification.CORRECT
    severity: Optional[Severity] = None
    topic: str = ""
    timestamp: str = ""
    #: Excerpts that must appear as evidence; a substring match is enough.
    expected_evidence: List[str] = Field(default_factory=list)
    notes: str = ""


class GoldStandard(BaseModel):
    """The human annotation for one reviewed video."""

    video: str
    references: List[str] = Field(default_factory=list)
    claims: List[GoldClaim] = Field(default_factory=list)
    #: Claims the system must NOT report as problems (known-correct statements).
    should_not_flag: List[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "GoldStandard":
        """Read a gold standard from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise Reviewer2Error(
                f"Gold standard file not found: {path}",
                module="metrics",
                stage="load",
                cause="The path passed to the evaluation does not exist.",
                action="Create the annotation file (see examples/gold_standard.json).",
            )
        try:
            return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise Reviewer2Error(
                f"Could not parse the gold standard: {path}",
                module="metrics",
                stage="load",
                cause=str(exc)[:300],
                action="Check the JSON syntax and the field names against examples/gold_standard.json.",
            ) from exc


class MetricResult(BaseModel):
    """One scored metric."""

    name: str
    value: float
    numerator: float = 0.0
    denominator: float = 0.0
    detail: str = ""

    @property
    def as_percentage(self) -> str:
        return f"{self.value:.1%}"


class EvaluationReport(BaseModel):
    """Everything the evaluation harness computed for one run."""

    video: str
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)
    matched_claims: List[str] = Field(default_factory=list)
    missed_claims: List[str] = Field(default_factory=list)
    spurious_critiques: List[str] = Field(default_factory=list)
    calibration_bins: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    def get(self, name: str) -> float:
        """Numeric value of a metric (0.0 when it was not computed)."""
        metric = self.metrics.get(name)
        return metric.value if metric else 0.0
