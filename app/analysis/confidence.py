"""Confidence calibration.

The confidence printed in the report is not simply what the model said. It
combines the model's own confidence with measurable factors: transcription
quality, retrieval scores, amount of evidence, agreement between sources and
whether the quotes could be verified in the corpus.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..logging_utils import get_logger
from ..models import (
    Claim,
    ConfidenceLevel,
    Evidence,
    EvidenceRelation,
    EvidenceStatus,
    RetrievalTrace,
)

logger = get_logger(__name__)

# Weights of each factor; they sum to 1.0.
WEIGHTS: Dict[str, float] = {
    "model": 0.40,
    "evidence": 0.25,
    "retrieval": 0.15,
    "transcription": 0.10,
    "agreement": 0.10,
}


def calibrate(
    *,
    claim: Claim,
    model_confidence: ConfidenceLevel,
    evidence: Sequence[Evidence],
    trace: Optional[RetrievalTrace],
    evidence_status: EvidenceStatus,
) -> tuple[ConfidenceLevel, float, Dict[str, float]]:
    """Return the calibrated level, its numeric score and the factor breakdown."""
    factors: Dict[str, float] = {}

    factors["model"] = model_confidence.score

    verified = [item for item in evidence if item.verified]
    if not verified:
        factors["evidence"] = 0.10
    elif len(verified) == 1:
        factors["evidence"] = 0.60
    elif len(verified) == 2:
        factors["evidence"] = 0.80
    else:
        factors["evidence"] = 0.90

    best_score = trace.best_score if trace else 0.0
    factors["retrieval"] = max(0.0, min(1.0, best_score))

    factors["transcription"] = max(0.0, min(1.0, claim.transcript_confidence))

    factors["agreement"] = _agreement(verified)

    score = sum(WEIGHTS[key] * value for key, value in factors.items())

    # Hard caps: some situations must never be reported as high confidence.
    if evidence_status in (EvidenceStatus.INSUFFICIENT, EvidenceStatus.UNKNOWN):
        score = min(score, 0.44)
    if evidence_status is EvidenceStatus.CONFLICTING:
        score = min(score, 0.60)
    if not verified:
        score = min(score, 0.44)
    if claim.low_confidence_transcript:
        score = min(score, 0.64)

    factors["final"] = round(score, 3)
    return ConfidenceLevel.from_score(score), round(score, 3), factors


def _agreement(evidence: Sequence[Evidence]) -> float:
    """How consistently the verified excerpts point in the same direction."""
    if not evidence:
        return 0.0
    relations: List[EvidenceRelation] = [item.relation for item in evidence]
    supports = sum(1 for relation in relations if relation is EvidenceRelation.SUPPORTS)
    contradicts = sum(1 for relation in relations if relation is EvidenceRelation.CONTRADICTS)
    if supports and contradicts:
        return 0.25  # sources disagree
    dominant = max(supports, contradicts)
    if dominant == 0:
        return 0.45  # only neutral/partial excerpts
    return min(1.0, 0.6 + 0.2 * (dominant - 1))


def apply_adjustment(score: float, adjustment: float, *, max_drop: float = 0.35) -> tuple[ConfidenceLevel, float]:
    """Apply the devil's advocate adjustment, bounded by ``max_drop``."""
    bounded = max(-abs(max_drop), min(0.1, float(adjustment)))
    new_score = max(0.0, min(1.0, score + bounded))
    return ConfidenceLevel.from_score(new_score), round(new_score, 3)
