"""Second verification layer — the devil's advocate.

Every critique is attacked before it is allowed into the report. A critique
that cannot survive the attack is either downgraded or dropped, and the
reason is recorded. This is the mechanism that implements
*precision over quantity*.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

from ..config import VerificationConfig
from ..errors import LLMError
from ..llm import LLMProvider, as_dict
from ..logging_utils import get_logger
from ..models import (
    Classification,
    ConfidenceLevel,
    Critique,
    EvidenceStatus,
    Severity,
    VerificationResult,
    parse_optional,
)
from ..prompts import SYSTEM_PROMPT, devil_advocate_prompt
from ..reports.i18n import strings
from ..analysis.confidence import apply_adjustment

logger = get_logger(__name__)


class DevilsAdvocate:
    """Challenges critiques and keeps only those that hold up."""

    def __init__(
        self,
        llm: LLMProvider,
        config: Optional[VerificationConfig] = None,
        *,
        concurrency: int = 1,
        language: str = "pt",
    ) -> None:
        self.llm = llm
        self.config = config or VerificationConfig()
        self.concurrency = max(1, int(concurrency))
        self.strings = strings(language)
        self.language = language

    def verify_all(self, critiques: Sequence[Critique]) -> List[Critique]:
        """Run the verification pass over every critique."""
        if not self.config.enabled:
            logger.info("Verification stage disabled by configuration")
            return list(critiques)

        critiques = list(critiques)
        checkable = sum(1 for c in critiques if c.classification.is_problem)
        logger.info("Challenging %d critique(s) that report a problem", checkable)

        if self.concurrency == 1 or len(critiques) <= 1:
            verified = [self.verify(critique) for critique in critiques]
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="devil") as pool:
                verified = list(pool.map(self.verify, critiques))

        dropped = sum(1 for critique in verified if critique.dropped)
        logger.info("Verification complete: %d critique(s) dropped", dropped)
        return verified

    # ------------------------------------------------------------------ #
    def verify(self, critique: Critique) -> Critique:
        """Challenge a single critique, adjusting or dropping it."""
        if self.config.skip_correct_claims and not critique.classification.is_problem:
            return critique

        if critique.classification is Classification.NOT_SUPPORTED:
            # "No evidence" is already the humble verdict; nothing to attack.
            critique.verification = VerificationResult(
                sustained=True,
                notes=self.strings["dv_no_evidence"],
            )
            return critique

        source_texts = _source_texts(critique)
        prompt = devil_advocate_prompt(
            critique.claim.text,
            critique.classification.value,
            critique.analysis,
            critique.problem,
            [item.quote for item in critique.evidence if item.verified and item.quote],
            source_texts,
            transcript_note=(
                self.strings["dv_transcript_confidence"].format(
                    confidence=critique.claim.transcript_confidence
                )
                + (self.strings["dv_low"] if critique.claim.low_confidence_transcript else "")
            ),
            language=self.language,
        )
        try:
            raw = as_dict(self.llm.generate_json(prompt, system=SYSTEM_PROMPT, expect="object"))
        except LLMError as exc:
            logger.debug("Devil's advocate failed for %s: %s", critique.claim.claim_id, exc.message)
            # A failed verification must not silently promote an unchecked critique.
            critique.verification = VerificationResult(
                sustained=True,
                notes=self.strings["dv_failed"],
                confidence_adjustment=-0.1,
            )
            critique.confidence, critique.confidence_score = apply_adjustment(
                critique.confidence_score, -0.1, max_drop=self.config.max_confidence_drop
            )
            return critique

        result = VerificationResult(
            sustained=bool(raw.get("sustained", True)),
            alternative_interpretation=str(raw.get("alternative_interpretation", "")).strip(),
            assumptions=[str(a).strip() for a in (raw.get("assumptions") or []) if str(a).strip()],
            context_considered=bool(raw.get("context_considered", True)),
            source_supports_critique=bool(raw.get("source_supports_critique", True)),
            conditions_that_would_validate_claim=str(raw.get("conditions_that_would_validate_claim", "")).strip(),
            transcription_risk=str(raw.get("transcription_risk", "")).strip(),
            conflicting_sources=str(raw.get("conflicting_sources", "")).strip(),
            notes=str(raw.get("notes", "")).strip(),
            confidence_adjustment=_as_float(raw.get("confidence_adjustment"), 0.0),
            revised_classification=parse_optional(Classification, raw.get("revised_classification")),
            revised_severity=parse_optional(Severity, raw.get("revised_severity")),
        )
        critique.verification = result
        return self._apply(critique, result)

    # ------------------------------------------------------------------ #
    def _apply(self, critique: Critique, result: VerificationResult) -> Critique:
        """Apply the verification outcome to the critique.

        A critique is deleted only when the evidence genuinely fails to back
        it. Models — small ones especially — answer "not sustained" whenever
        they can imagine a more charitable reading of the claim, and deleting
        on that signal alone silently throws away correct findings. When the
        objection is a nuance rather than a refutation, the critique is kept,
        its confidence drops, and the objection is shown in the report.
        """
        if not result.sustained:
            baseless = not result.source_supports_critique
            if baseless and self.config.drop_unsustained:
                critique.dropped = True
                critique.dropped_reason = (
                    result.notes or self.strings["dv_unsustained"]
                )
                logger.debug("Dropped critique for %s: %s", critique.claim.claim_id, critique.dropped_reason)
                return critique

            if baseless:
                critique.classification = Classification.NOT_SUPPORTED
                critique.severity = Severity.LOW
                critique.evidence_status = EvidenceStatus.INSUFFICIENT
                critique.confidence = ConfidenceLevel.LOW
                critique.confidence_score = min(critique.confidence_score, 0.3)
                return critique

            # Contested but not refuted: keep the finding, lower the confidence.
            critique.confidence, critique.confidence_score = apply_adjustment(
                critique.confidence_score,
                min(result.confidence_adjustment, -0.15),
                max_drop=self.config.max_confidence_drop,
            )
            if result.revised_classification is not None:
                critique.classification = result.revised_classification
            if result.revised_severity is not None:
                critique.severity = result.revised_severity
            return critique

        if result.revised_classification is not None:
            critique.classification = result.revised_classification
        if result.revised_severity is not None:
            critique.severity = result.revised_severity
        if result.conflicting_sources:
            critique.evidence_status = EvidenceStatus.CONFLICTING

        adjustment = result.confidence_adjustment
        if result.alternative_interpretation and adjustment == 0.0:
            adjustment = -0.05  # a plausible alternative reading always costs confidence
        if adjustment:
            critique.confidence, critique.confidence_score = apply_adjustment(
                critique.confidence_score, adjustment, max_drop=self.config.max_confidence_drop
            )
        return critique


def _source_texts(critique: Critique) -> List[str]:
    """The retrieved excerpts, which the devil's advocate may re-read."""
    if critique.retrieval_trace is None:
        return []
    return [item.chunk.text for item in critique.retrieval_trace.results]


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
