"""Markdown report generation.

The report is the product of the whole pipeline. It is written so that a
reader can walk from any critique down to the video timestamp and up to the
page of the source that justifies it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..config import ReportConfig
from ..errors import ReportError
from ..logging_utils import get_logger
from ..models import (
    Classification,
    Critique,
    Evidence,
    InternalContradiction,
    ReviewReport,
    ReviewStatistics,
    Severity,
)
from ..analysis.omissions import collect_omissions
from .i18n import strings

logger = get_logger(__name__)


class ReportGenerator:
    """Renders a :class:`ReviewReport` as Markdown (and an audit JSON)."""

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        self.config = config or ReportConfig()
        self.strings = strings(self.config.language)

    # ------------------------------------------------------------------ #
    def render(self, report: ReviewReport) -> str:
        """Return the full Markdown document."""
        s = self.strings
        parts: List[str] = [f"# {s['title']}", ""]
        parts += self._executive_summary(report)
        parts += self._analysis_info(report)
        parts += self._statistics(report)
        parts += self._main_problems(report)
        parts += self._claims(report)
        parts += self._internal_contradictions(report)
        parts += self._source_contradictions(report)
        parts += self._omissions(report)
        parts += self._simplifications(report)
        parts += self._correct_points(report)
        parts += self._corrections(report)
        parts += self._final_assessment(report)
        parts += self._limitations(report)
        return "\n".join(parts).rstrip() + "\n"

    def write(self, report: ReviewReport, path: Path) -> Path:
        """Write the Markdown report (and optionally the audit JSON)."""
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.render(report), encoding="utf-8")
        except OSError as exc:
            raise ReportError(
                f"Could not write the report to {path}.",
                module="reports.generator",
                stage="write",
                cause=str(exc),
                action="Check the directory permissions and the available disk space.",
            ) from exc

        if self.config.include_audit_json:
            audit_path = path.with_name(f"{path.stem}_audit.json")
            audit_path.write_text(
                json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug("Audit JSON written to %s", audit_path)
        return path

    # ------------------------------------------------------------------ #
    # Sections
    # ------------------------------------------------------------------ #
    def _executive_summary(self, report: ReviewReport) -> List[str]:
        """The verdict first: what the author should do about this recording."""
        s = self.strings
        stats = report.statistics
        defects = report.defects
        verdict = report.quality

        lines = [
            f"## {s['s1']}",
            "",
            f"### {s['verdict']}: `{verdict.label(s)}`",
            "",
            verdict.advice(s),
            "",
        ]

        if defects:
            lines.append(f"**{s['defects']}:**")
            lines.append("")
            for critique in defects[:8]:
                lines.append(
                    f"- **[{critique.claim.timestamp}]** {_one_line(critique.claim.text)}"
                    f"  \n  `{critique.classification.value}` · `{critique.severity.value}`"
                    f" · {s['confidence'].lower()} `{critique.confidence.value}`"
                )
            if len(defects) > 8:
                lines.append(f"- … {len(defects) - 8} more")
            lines.append("")

        lines += [
            f"- {s['total_claims']}: **{stats.total_claims}**",
            f"- {s['defects']}: **{len(defects)}**",
            f"- {s['evidence_coverage']}: **{stats.evidence_coverage:.0%}**",
        ]
        if report.undetermined:
            lines.append(
                f"- {s['undetermined_count']}: **{len(report.undetermined)}** "
                f"— {s['undetermined_note']}"
            )
        lines.append("")
        return lines

    def _analysis_info(self, report: ReviewReport) -> List[str]:
        s = self.strings
        references = "\n".join(f"  - `{reference}`" for reference in report.references) or "  - —"
        return [
            f"## {s['s2']}",
            "",
            f"- {s['video']}: `{report.video}`",
            f"- {s['references']}:",
            references,
            f"- {s['llm']}: `{report.llm_model}`",
            f"- {s['asr']}: `{report.asr_model}`",
            f"- {s['embeddings']}: `{report.embedding_model}`",
            f"- {s['vector_store']}: `{report.vector_store}`",
            f"- {s['generated_at']}: {report.created_at.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}",
            f"- {s['duration']}: {report.duration_seconds:.1f}s",
            "",
        ]

    def _statistics(self, report: ReviewReport) -> List[str]:
        s = self.strings
        stats = report.statistics
        lines = [
            f"## {s['s3']}",
            "",
            f"| {s['metric']} | {s['value']} |",
            "|---|---|",
            f"| {s['segments']} | {stats.total_segments} |",
            f"| {s['total_claims']} | {stats.total_claims} |",
            f"| {s['analysed_claims']} | {stats.analysed_claims} |",
            f"| {s['documents']} | {stats.total_documents} |",
            f"| {s['chunks']} | {stats.total_chunks} |",
            f"| {s['defects']} | {stats.total_defects} |",
            f"| {s['undetermined_count']} | {stats.total_undetermined} |",
            f"| {s['omissions_count']} | {stats.total_omissions} |",
            f"| {s['contradictions_count']} | {stats.total_internal_contradictions} |",
            f"| {s['dropped']} | {stats.dropped_critiques} |",
            f"| {s['evidence_coverage']} | {stats.evidence_coverage:.0%} |",
            f"| {s['transcript_confidence']} | {stats.mean_transcript_confidence:.2f} |",
            "",
        ]
        lines += _histogram(f"**{s['by_classification']}**", stats.by_classification)
        lines += _histogram(f"**{s['by_severity']}**", stats.by_severity)
        lines += _histogram(f"**{s['by_confidence']}**", stats.by_confidence)
        return lines

    def _main_problems(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s4']}", ""]
        problems = [c for c in report.defects if c.severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM)]
        if not problems:
            return lines + [s["no_problems"], ""]
        for critique in problems:
            lines.append(
                f"- **{critique.claim.claim_id}** [{critique.claim.timestamp}] "
                f"`{critique.severity.value}` — {_one_line(critique.problem or critique.conclusion)}"
            )
        lines.append("")
        return lines

    def _claims(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s5']}", ""]
        critiques = report.active_critiques
        if not self.config.include_correct_claims:
            critiques = [c for c in critiques if c.is_problem]
        if not critiques:
            return lines + [s["none_found"], ""]
        for critique in critiques:
            lines += self._claim_block(critique)
        return lines

    def _claim_block(self, critique: Critique) -> List[str]:
        s = self.strings
        claim = critique.claim
        lines = [
            f"### Claim #{claim.claim_id.split('_')[-1]}",
            "",
            f"**{s['timestamp']}:** {claim.timestamp}",
            "",
            f"**{s['claim']}:**",
            f"> {_one_line(claim.text)}",
            "",
            f"**{s['classification']}:**",
            f"`{critique.classification.value}`",
            "",
            f"**{s['severity']}:**",
            f"`{critique.severity.value}`",
            "",
            f"**{s['confidence']}:**",
            f"`{critique.confidence.value}` ({critique.confidence_score:.2f})",
            "",
            f"**{s['statement_type']}:** `{critique.statement_type.value}` · "
            f"**{s['compatibility']}:** `{critique.compatibility.value}` · "
            f"**{s['evidence_status']}:** `{critique.evidence_status.value}`",
            "",
            f"**{s['evidence']}:**",
            "",
        ]
        lines += self._evidence_block(critique.evidence)
        lines += [
            f"**{s['analysis']}:**",
            "",
            critique.analysis or "—",
            "",
        ]
        if critique.problem:
            lines += [f"**{s['problem']}:**", "", critique.problem, ""]
        if critique.suggested_correction:
            lines += [
                f"**{s['correction']}:**",
                "",
                f"> {_one_line(critique.suggested_correction)}",
                "",
            ]
        if critique.conclusion:
            lines += [f"**{s['conclusion']}:**", "", critique.conclusion, ""]
        if critique.verification is not None and critique.classification.is_problem:
            lines += self._verification_block(critique)
        lines.append("---")
        lines.append("")
        return lines

    def _evidence_block(self, evidence: Sequence[Evidence]) -> List[str]:
        s = self.strings
        verified = [item for item in evidence if item.verified and item.quote]
        if not verified:
            return [s["no_evidence"], ""]
        lines: List[str] = []
        for item in verified:
            locator = [f"{s['source']}: `{item.document}`"]
            if item.page is not None:
                locator.append(f"{s['page']}: {item.page}")
            if item.section:
                locator.append(f"{s['section']}: {item.section}")
            locator.append(f"{s['similarity']}: {item.score:.2f}")
            lines.append(" · ".join(locator))
            lines.append("")
            lines.append(f"> {_quote(item.quote, self.config.max_evidence_quote_chars)}")
            lines.append("")
        return lines

    def _verification_block(self, critique: Critique) -> List[str]:
        s = self.strings
        result = critique.verification
        if result is None:
            return []
        lines = [f"**{s['verification']}:**", ""]
        if result.notes:
            lines.append(f"- {result.notes}")
        if result.alternative_interpretation:
            lines.append(f"- {s['alternative']}: {result.alternative_interpretation}")
        if result.assumptions:
            lines.append(f"- {s['assumptions']}: " + "; ".join(result.assumptions))
        if result.conditions_that_would_validate_claim:
            lines.append(f"- {result.conditions_that_would_validate_claim}")
        if result.transcription_risk:
            lines.append(f"- {result.transcription_risk}")
        lines.append("")
        return lines

    def _internal_contradictions(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s6']}", ""]
        contradictions = [c for c in report.internal_contradictions if c.is_contradiction]
        if not contradictions:
            return lines + [s["none_found"], ""]
        for item in contradictions:
            lines += self._contradiction_block(item)
        return lines

    def _contradiction_block(self, item: InternalContradiction) -> List[str]:
        s = self.strings
        return [
            f"### {item.contradiction_id}",
            "",
            f"**{s['claim_a']}** [{item.claim_a.timestamp}]:",
            f"> {_one_line(item.claim_a.text)}",
            "",
            f"**{s['claim_b']}** [{item.claim_b.timestamp}]:",
            f"> {_one_line(item.claim_b.text)}",
            "",
            f"**{s['explanation']}:** {item.explanation or '—'}",
            "",
            f"**{s['severity']}:** `{item.severity.value}` · "
            f"**{s['confidence']}:** `{item.confidence.value}` · "
            f"**{s['similarity']}:** {item.similarity:.2f}",
            "",
            "---",
            "",
        ]

    def _source_contradictions(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s7']}", ""]
        conflicting = [
            critique
            for critique in report.active_critiques
            if critique.classification in (Classification.INCORRECT, Classification.CONTRADICTORY)
        ]
        if not conflicting:
            return lines + [s["none_found"], ""]
        for critique in conflicting:
            lines += [
                f"### {critique.claim.claim_id} [{critique.claim.timestamp}]",
                "",
                f"> {_one_line(critique.claim.text)}",
                "",
                f"**{s['classification']}:** `{critique.classification.value}` · "
                f"**{s['compatibility']}:** `{critique.compatibility.value}`",
                "",
            ]
            lines += self._evidence_block(critique.evidence)
            lines += [critique.analysis or "—", "", "---", ""]
        return lines

    def _omissions(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s8']}", ""]
        omissions = collect_omissions(report.active_critiques)
        if not omissions:
            return lines + [s["none_found"], ""]
        for omission in omissions:
            critique = next(
                (c for c in report.active_critiques if c.claim.claim_id == omission.claim_id), None
            )
            timestamp = critique.claim.timestamp if critique else "—"
            lines += [
                f"### {omission.claim_id} [{timestamp}] — `{omission.impact.value}`",
                "",
                f"**{s['missing']}:** {omission.missing_information}",
                "",
            ]
            if omission.explanation:
                lines += [f"**{s['why']}:** {omission.explanation}", ""]
            lines += self._evidence_block(omission.evidence)
            lines += ["---", ""]
        return lines

    def _simplifications(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s9']}", ""]
        items = [
            critique
            for critique in report.active_critiques
            if critique.classification is Classification.PEDAGOGICAL_SIMPLIFICATION
        ]
        if not items:
            return lines + [s["none_found"], ""]
        for critique in items:
            lines += [
                f"- **{critique.claim.claim_id}** [{critique.claim.timestamp}] "
                f"{_one_line(critique.claim.text)}",
                f"  - {critique.analysis or '—'}",
            ]
        lines.append("")
        return lines

    def _correct_points(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s10']}", ""]
        items = [
            critique
            for critique in report.active_critiques
            if critique.classification is Classification.CORRECT
        ]
        if not items:
            return lines + [s["none_found"], ""]
        for critique in items:
            evidence = next((e for e in critique.evidence if e.verified), None)
            locator = f" — `{evidence.locator}`" if evidence else ""
            lines.append(
                f"- **{critique.claim.claim_id}** [{critique.claim.timestamp}] "
                f"{_one_line(critique.claim.text)}{locator}"
            )
        lines.append("")
        return lines

    def _corrections(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s11']}", ""]
        items = [c for c in report.problems if c.suggested_correction]
        if not items:
            return lines + [s["none_found"], ""]
        for critique in items:
            lines += [
                f"### {critique.claim.claim_id} [{critique.claim.timestamp}]",
                "",
                f"- {s['claim']}: “{_one_line(critique.claim.text)}”",
                f"- {s['correction']}: **“{_one_line(critique.suggested_correction)}”**",
                "",
            ]
        return lines

    def _final_assessment(self, report: ReviewReport) -> List[str]:
        s = self.strings
        stats = report.statistics
        defects = report.defects
        verdict = report.quality

        lines = [
            f"## {s['s12']}", "",
            f"**{s['verdict']}: `{verdict.label(s)}`** — {verdict.advice(s)}", "",
        ]
        if defects:
            for severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW):
                count = sum(1 for c in defects if c.severity is severity)
                if count:
                    lines.append(f"- `{severity.value}`: {count}")
            lines.append(f"- {s['evidence_coverage']}: {stats.evidence_coverage:.0%}")
        if report.undetermined:
            lines.append(f"- {s['undetermined_count']}: {len(report.undetermined)}")
        lines.append("")
        return lines

    def _limitations(self, report: ReviewReport) -> List[str]:
        s = self.strings
        lines = [f"## {s['s13']}", ""]
        for limitation in report.limitations:
            lines.append(f"- {limitation}")
        if not report.limitations:
            lines.append(s["none_found"])
        lines.append("")
        return lines


# --------------------------------------------------------------------------- #
def build_statistics(report: ReviewReport) -> ReviewStatistics:
    """Compute the aggregate numbers shown in section 3 of the report."""
    active = report.active_critiques
    by_classification: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    by_confidence: Dict[str, int] = {}

    for critique in active:
        by_classification[critique.classification.value] = (
            by_classification.get(critique.classification.value, 0) + 1
        )
        by_confidence[critique.confidence.value] = by_confidence.get(critique.confidence.value, 0) + 1
        if critique.is_problem:
            by_severity[critique.severity.value] = by_severity.get(critique.severity.value, 0) + 1

    # Evidence coverage measures how many *evidence-based* critiques carry a
    # verifiable quote. NAO_SUSTENTADA is excluded: it states that no evidence
    # exists, which is a valid result rather than a missing citation.
    evidence_based = [
        critique
        for critique in active
        if critique.is_problem and critique.classification is not Classification.NOT_SUPPORTED
    ]
    with_evidence = [critique for critique in evidence_based if critique.has_verified_evidence]
    coverage = (len(with_evidence) / len(evidence_based)) if evidence_based else 1.0

    return ReviewStatistics(
        total_segments=len(report.transcript.segments) if report.transcript else 0,
        total_claims=len(report.claims),
        analysed_claims=len(active),
        dropped_critiques=len(report.critiques) - len(active),
        by_classification=by_classification,
        by_severity=by_severity,
        by_confidence=by_confidence,
        total_documents=len(report.references),
        total_chunks=report.statistics.total_chunks,
        total_defects=len(report.defects),
        total_undetermined=len(report.undetermined),
        total_omissions=len(collect_omissions(active)),
        total_internal_contradictions=sum(1 for c in report.internal_contradictions if c.is_contradiction),
        evidence_coverage=round(coverage, 3),
        mean_transcript_confidence=round(report.transcript.mean_confidence, 3) if report.transcript else 0.0,
    )


def _histogram(title: str, counts: Dict[str, int]) -> List[str]:
    if not counts:
        return []
    lines = [title, ""]
    for key, value in sorted(counts.items(), key=lambda item: -item[1]):
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return lines


def _one_line(text: str) -> str:
    """Collapse a string into a single Markdown-safe line."""
    return " ".join(str(text or "").split())


def _quote(text: str, limit: int) -> str:
    """Truncate a quote, marking the truncation explicitly."""
    flat = _one_line(text)
    if len(flat) <= limit:
        return flat
    return flat[:limit].rstrip() + " […]"
