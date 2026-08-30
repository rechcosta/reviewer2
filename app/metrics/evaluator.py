"""Evaluation harness (project specification, section 31).

Computes the metrics the project cares about, in the order it cares about
them — critique precision first, quantity last:

* ``claim_extraction_recall`` / ``claim_extraction_precision``
* ``retrieval_hit_rate``
* ``critique_precision``  ← the metric that matters most
* ``false_positive_rate``
* ``evidence_coverage``
* ``confidence_calibration_error``
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..logging_utils import get_logger
from ..models import Classification, Critique, ReviewReport
from ..text_utils import overlap, strip_accents
from .models import EvaluationReport, GoldClaim, GoldStandard, MetricResult

logger = get_logger(__name__)

#: Two claims are considered the same when their content words overlap this much.
MATCH_THRESHOLD = 0.6

#: Confidence buckets used for calibration.
_CALIBRATION_BINS: Tuple[Tuple[str, float, float], ...] = (
    ("BAIXA", 0.0, 0.45),
    ("MEDIA", 0.45, 0.65),
    ("ALTA", 0.65, 0.85),
    ("MUITO_ALTA", 0.85, 1.01),
)


def evaluate(report: ReviewReport, gold: GoldStandard) -> EvaluationReport:
    """Score a review against its human annotation."""
    evaluation = EvaluationReport(video=report.video)
    # Extraction is measured over every claim the pipeline produced; the
    # critique metrics only over the ones that reached the report. Mixing the
    # two hides claims that were extracted and then dropped.
    critiques = report.active_critiques
    all_critiques = report.critiques

    pairs = _match_claims(all_critiques, gold.claims)
    matched = [(critique, gold_claim) for critique, gold_claim in pairs if gold_claim is not None]

    evaluation.matched_claims = [critique.claim.text for critique, _ in matched]
    evaluation.missed_claims = [
        gold_claim.text
        for gold_claim in gold.claims
        if not any(gold_claim is paired for _, paired in matched)
    ]

    evaluation.metrics["claim_extraction_recall"] = _ratio(
        "claim_extraction_recall",
        len(matched),
        len(gold.claims),
        "gold claims that Reviewer2 also extracted",
    )
    evaluation.metrics["claim_extraction_precision"] = _ratio(
        "claim_extraction_precision",
        len(matched),
        len(all_critiques),
        "extracted claims that correspond to a gold claim",
    )
    evaluation.metrics["retrieval_hit_rate"] = _retrieval_hit_rate(matched)

    reported = [(c, g) for c, g in matched if not c.dropped]
    evaluation.metrics["critique_precision"] = _critique_precision(reported)
    evaluation.metrics["dropped_correct_critiques"] = _dropped_correct(matched)
    evaluation.metrics["false_positive_rate"], evaluation.spurious_critiques = _false_positive_rate(
        critiques, gold
    )
    evaluation.metrics["evidence_coverage"] = _evidence_coverage(critiques)

    calibration, bins = _calibration_error(matched)
    evaluation.metrics["confidence_calibration_error"] = calibration
    evaluation.calibration_bins = bins

    return evaluation


# --------------------------------------------------------------------------- #
# Individual metrics
# --------------------------------------------------------------------------- #
def _match_claims(
    critiques: Sequence[Critique], gold_claims: Sequence[GoldClaim]
) -> List[Tuple[Critique, Optional[GoldClaim]]]:
    """Align extracted claims with gold claims by content-word overlap."""
    used: set[int] = set()
    pairs: List[Tuple[Critique, Optional[GoldClaim]]] = []

    for critique in critiques:
        best_index: Optional[int] = None
        best_score = MATCH_THRESHOLD
        for index, gold_claim in enumerate(gold_claims):
            if index in used:
                continue
            score = max(
                overlap(critique.claim.text, gold_claim.text),
                overlap(gold_claim.text, critique.claim.text),
            )
            if score >= best_score:
                best_index, best_score = index, score
        if best_index is None:
            pairs.append((critique, None))
        else:
            used.add(best_index)
            pairs.append((critique, gold_claims[best_index]))
    return pairs


def _critique_precision(matched: Sequence[Tuple[Critique, GoldClaim]]) -> MetricResult:
    """Share of matched claims whose classification is the expected one.

    This is the project's primary metric: are the verdicts correct, not how
    many were produced.
    """
    correct = sum(
        1 for critique, gold_claim in matched if critique.classification is gold_claim.classification
    )
    return _ratio(
        "critique_precision", correct, len(matched), "verdicts matching the human annotation"
    )


def _dropped_correct(matched: Sequence[Tuple[Critique, GoldClaim]]) -> MetricResult:
    """Share of dropped critiques whose verdict was actually right.

    A high value means the verification pass is deleting correct findings —
    the failure mode that silently makes the reviewer useless.
    """
    dropped = [(c, g) for c, g in matched if c.dropped]
    correct = sum(1 for c, g in dropped if c.classification is g.classification)
    return _ratio(
        "dropped_correct_critiques", correct, len(dropped),
        "dropped critiques whose verdict matched the annotation (lower is better)",
    )


def _false_positive_rate(
    critiques: Sequence[Critique], gold: GoldStandard
) -> Tuple[MetricResult, List[str]]:
    """Share of reported problems that should not have been reported."""
    problems = [critique for critique in critiques if critique.is_problem]
    spurious: List[str] = []

    for critique in problems:
        if _is_expected_problem(critique, gold):
            continue
        spurious.append(critique.claim.text)

    metric = _ratio(
        "false_positive_rate",
        len(spurious),
        len(problems),
        "reported problems that the annotation does not consider problems",
    )
    return metric, spurious


def _is_expected_problem(critique: Critique, gold: GoldStandard) -> bool:
    """True when the annotation also treats this claim as a problem."""
    for text in gold.should_not_flag:
        if _same_claim(critique.claim.text, text):
            return False
    for gold_claim in gold.claims:
        if _same_claim(critique.claim.text, gold_claim.text):
            return gold_claim.classification.is_problem
    # A problem on a claim the annotation never mentions cannot be confirmed;
    # NAO_SUSTENTADA is an explicit "undetermined", so it is not a false positive.
    return critique.classification is Classification.NOT_SUPPORTED


def _retrieval_hit_rate(matched: Sequence[Tuple[Critique, GoldClaim]]) -> MetricResult:
    """Share of claims whose expected evidence was actually retrieved."""
    expected = [(critique, gold) for critique, gold in matched if gold.expected_evidence]
    if not expected:
        return MetricResult(
            name="retrieval_hit_rate", value=0.0, detail="no expected_evidence in the annotation"
        )

    hits = 0
    for critique, gold_claim in expected:
        retrieved = " ".join(
            strip_accents(item.chunk.text)
            for item in (critique.retrieval_trace.results if critique.retrieval_trace else [])
        )
        retrieved = " ".join(retrieved.split())
        if any(" ".join(strip_accents(snippet).split()) in retrieved for snippet in gold_claim.expected_evidence):
            hits += 1
    return _ratio(
        "retrieval_hit_rate", hits, len(expected), "claims whose expected excerpt was retrieved"
    )


def _evidence_coverage(critiques: Sequence[Critique]) -> MetricResult:
    """Share of evidence-based critiques carrying a verifiable quote."""
    evidence_based = [
        critique
        for critique in critiques
        if critique.is_problem and critique.classification is not Classification.NOT_SUPPORTED
    ]
    covered = sum(1 for critique in evidence_based if critique.has_verified_evidence)
    return _ratio(
        "evidence_coverage", covered, len(evidence_based), "critiques with a verifiable quote"
    )


def _calibration_error(
    matched: Sequence[Tuple[Critique, GoldClaim]]
) -> Tuple[MetricResult, Dict[str, Dict[str, float]]]:
    """Expected calibration error: declared confidence vs. observed accuracy.

    A well-calibrated reviewer that says ``ALTA`` is right about as often as
    the numeric confidence it printed.
    """
    bins: Dict[str, Dict[str, float]] = {}
    total = 0
    weighted_error = 0.0

    for label, low, high in _CALIBRATION_BINS:
        members = [
            (critique, gold)
            for critique, gold in matched
            if low <= critique.confidence_score < high
        ]
        if not members:
            continue
        accuracy = sum(
            1 for critique, gold in members if critique.classification is gold.classification
        ) / len(members)
        mean_confidence = sum(critique.confidence_score for critique, _ in members) / len(members)
        bins[label] = {
            "count": float(len(members)),
            "mean_confidence": round(mean_confidence, 3),
            "accuracy": round(accuracy, 3),
            "gap": round(mean_confidence - accuracy, 3),
        }
        weighted_error += len(members) * abs(mean_confidence - accuracy)
        total += len(members)

    value = (weighted_error / total) if total else 0.0
    metric = MetricResult(
        name="confidence_calibration_error",
        value=round(value, 3),
        numerator=round(weighted_error, 3),
        denominator=float(total),
        detail="mean |declared confidence − observed accuracy| (lower is better)",
    )
    return metric, bins


def _ratio(name: str, numerator: float, denominator: float, detail: str) -> MetricResult:
    """Build a ratio metric, treating an empty denominator as 0.0."""
    value = (numerator / denominator) if denominator else 0.0
    return MetricResult(
        name=name,
        value=round(value, 3),
        numerator=float(numerator),
        denominator=float(denominator),
        detail=detail,
    )


def _same_claim(a: str, b: str) -> bool:
    """Whether two claim texts refer to the same assertion."""
    return max(overlap(a, b), overlap(b, a)) >= MATCH_THRESHOLD


def render_markdown(evaluation: EvaluationReport) -> str:
    """Render an evaluation as a small Markdown report."""
    lines = [
        "# Reviewer2 — Evaluation",
        "",
        f"- Video: `{evaluation.video}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Detail |",
        "|---|---|---|",
    ]
    order = [
        "critique_precision",
        "dropped_correct_critiques",
        "false_positive_rate",
        "evidence_coverage",
        "claim_extraction_recall",
        "claim_extraction_precision",
        "retrieval_hit_rate",
        "confidence_calibration_error",
    ]
    for name in order:
        metric = evaluation.metrics.get(name)
        if metric is None:
            continue
        shown = f"{metric.value:.3f}" if name.endswith("error") else metric.as_percentage
        counts = (
            f" ({metric.numerator:.0f}/{metric.denominator:.0f})" if metric.denominator else ""
        )
        lines.append(f"| `{name}` | {shown}{counts} | {metric.detail} |")

    if evaluation.calibration_bins:
        lines += ["", "## Confidence calibration", "", "| Bucket | n | Mean confidence | Accuracy | Gap |", "|---|---|---|---|---|"]
        for label, values in evaluation.calibration_bins.items():
            lines.append(
                f"| `{label}` | {values['count']:.0f} | {values['mean_confidence']:.2f} | "
                f"{values['accuracy']:.2f} | {values['gap']:+.2f} |"
            )

    if evaluation.missed_claims:
        lines += ["", "## Gold claims not extracted", ""]
        lines += [f"- {text}" for text in evaluation.missed_claims]
    if evaluation.spurious_critiques:
        lines += ["", "## Problems reported without support in the annotation", ""]
        lines += [f"- {text}" for text in evaluation.spurious_critiques]
    return "\n".join(lines) + "\n"
