"""Devil's advocate verification pass and confidence calibration."""

from __future__ import annotations

from app.analysis.confidence import apply_adjustment, calibrate
from app.config import VerificationConfig
from app.llm import LLMProvider
from app.models import (
    Claim,
    Classification,
    ConfidenceLevel,
    Critique,
    Evidence,
    EvidenceRelation,
    EvidenceStatus,
    RetrievalTrace,
    RetrievedChunk,
    DocumentChunk,
    Severity,
)
from app.verification import DevilsAdvocate


class _FixedProvider(LLMProvider):
    name = "fixed"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.model = "fixed"

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None, stop=None):
        return self.answer


def _critique(classification: Classification = Classification.INCORRECT) -> Critique:
    chunk = DocumentChunk(chunk_id="c1", document="d.md", text="A cache é volátil.", page=3)
    return Critique(
        claim=Claim(claim_id="claim_001", text="A cache é não volátil.", timestamp="00:00:24"),
        classification=classification,
        severity=Severity.HIGH,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.75,
        evidence_status=EvidenceStatus.OK,
        evidence=[
            Evidence(
                chunk_id="c1", document="d.md", page=3, quote="A cache é volátil.",
                score=0.8, relation=EvidenceRelation.CONTRADICTS, verified=True,
            )
        ],
        analysis="A fonte afirma o contrário.",
        problem="Definição incorreta.",
        retrieval_trace=RetrievalTrace(
            query="cache volátil", results=[RetrievedChunk(chunk=chunk, score=0.8)]
        ),
    )


# --------------------------------------------------------------------------- #
# Confidence calibration
# --------------------------------------------------------------------------- #
def test_confidence_is_capped_without_evidence() -> None:
    claim = Claim(claim_id="c", text="x", transcript_confidence=1.0)
    level, score, _ = calibrate(
        claim=claim,
        model_confidence=ConfidenceLevel.VERY_HIGH,
        evidence=[],
        trace=None,
        evidence_status=EvidenceStatus.INSUFFICIENT,
    )
    assert score <= 0.44
    assert level in (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM)


def test_more_verified_evidence_raises_confidence() -> None:
    claim = Claim(claim_id="c", text="x", transcript_confidence=1.0)
    chunk = DocumentChunk(chunk_id="c1", document="d", text="t")
    trace = RetrievalTrace(query="q", results=[RetrievedChunk(chunk=chunk, score=0.9)])
    one = Evidence(chunk_id="c1", document="d", quote="q", relation=EvidenceRelation.SUPPORTS, verified=True)
    two = Evidence(chunk_id="c2", document="d", quote="q", relation=EvidenceRelation.SUPPORTS, verified=True)

    _, single, _ = calibrate(
        claim=claim, model_confidence=ConfidenceLevel.HIGH, evidence=[one], trace=trace,
        evidence_status=EvidenceStatus.OK,
    )
    _, double, _ = calibrate(
        claim=claim, model_confidence=ConfidenceLevel.HIGH, evidence=[one, two], trace=trace,
        evidence_status=EvidenceStatus.OK,
    )
    assert double > single


def test_disagreeing_sources_lower_confidence() -> None:
    claim = Claim(claim_id="c", text="x", transcript_confidence=1.0)
    chunk = DocumentChunk(chunk_id="c1", document="d", text="t")
    trace = RetrievalTrace(query="q", results=[RetrievedChunk(chunk=chunk, score=0.9)])
    agreeing = [
        Evidence(chunk_id="c1", document="d", quote="q", relation=EvidenceRelation.SUPPORTS, verified=True),
        Evidence(chunk_id="c2", document="d", quote="q", relation=EvidenceRelation.SUPPORTS, verified=True),
    ]
    disagreeing = [
        agreeing[0],
        Evidence(chunk_id="c2", document="d", quote="q", relation=EvidenceRelation.CONTRADICTS, verified=True),
    ]
    _, high, _ = calibrate(
        claim=claim, model_confidence=ConfidenceLevel.HIGH, evidence=agreeing, trace=trace,
        evidence_status=EvidenceStatus.OK,
    )
    _, low, _ = calibrate(
        claim=claim, model_confidence=ConfidenceLevel.HIGH, evidence=disagreeing, trace=trace,
        evidence_status=EvidenceStatus.OK,
    )
    assert low < high


def test_bad_transcription_caps_confidence() -> None:
    claim = Claim(claim_id="c", text="x", transcript_confidence=0.4, low_confidence_transcript=True)
    chunk = DocumentChunk(chunk_id="c1", document="d", text="t")
    trace = RetrievalTrace(query="q", results=[RetrievedChunk(chunk=chunk, score=0.95)])
    evidence = [
        Evidence(chunk_id="c1", document="d", quote="q", relation=EvidenceRelation.SUPPORTS, verified=True)
        for _ in range(3)
    ]
    level, score, _ = calibrate(
        claim=claim, model_confidence=ConfidenceLevel.VERY_HIGH, evidence=evidence, trace=trace,
        evidence_status=EvidenceStatus.OK,
    )
    assert score <= 0.64
    assert level is not ConfidenceLevel.VERY_HIGH


def test_adjustment_is_bounded() -> None:
    _, score = apply_adjustment(0.8, -5.0, max_drop=0.35)
    assert score == 0.45
    _, raised = apply_adjustment(0.5, 2.0)
    assert raised == 0.6


# --------------------------------------------------------------------------- #
# Devil's advocate
# --------------------------------------------------------------------------- #
def test_baseless_critique_is_dropped() -> None:
    """Only a critique the source does not back is deleted."""
    answer = (
        '{"sustained": false, "source_supports_critique": false, '
        '"notes": "a fonte não sustenta a crítica", "confidence_adjustment": -0.3}'
    )
    critique = DevilsAdvocate(_FixedProvider(answer), VerificationConfig()).verify(_critique())
    assert critique.dropped
    assert "não sustenta" in critique.dropped_reason
    assert critique.is_problem is False


def test_contested_but_grounded_critique_survives_with_lower_confidence() -> None:
    """A mere nuance must not delete a correct finding.

    Small models answer "not sustained" whenever a more charitable reading of
    the claim exists. Deleting on that alone throws away real errors, so the
    critique is kept and its confidence drops instead.
    """
    answer = (
        '{"sustained": false, "source_supports_critique": true, '
        '"notes": "a afirmação pode ser lida como uma meta teórica", "confidence_adjustment": -0.1}'
    )
    critique = DevilsAdvocate(_FixedProvider(answer), VerificationConfig()).verify(_critique())
    assert not critique.dropped
    assert critique.classification is Classification.INCORRECT   # o achado é preservado
    assert critique.confidence_score <= 0.6                      # mas com menos confiança
    assert critique.verification is not None
    assert "meta teórica" in critique.verification.notes


def test_baseless_critique_can_be_downgraded_instead_of_dropped() -> None:
    answer = (
        '{"sustained": false, "source_supports_critique": false, '
        '"notes": "frágil", "confidence_adjustment": -0.3}'
    )
    config = VerificationConfig(drop_unsustained=False)
    critique = DevilsAdvocate(_FixedProvider(answer), config).verify(_critique())
    assert not critique.dropped
    assert critique.classification is Classification.NOT_SUPPORTED
    assert critique.confidence is ConfidenceLevel.LOW


def test_sustained_critique_survives() -> None:
    answer = '{"sustained": true, "notes": "mantida", "confidence_adjustment": 0.0, "assumptions": []}'
    critique = DevilsAdvocate(_FixedProvider(answer), VerificationConfig()).verify(_critique())
    assert not critique.dropped
    assert critique.classification is Classification.INCORRECT
    assert critique.verification is not None and critique.verification.sustained


def test_alternative_interpretation_costs_confidence() -> None:
    answer = (
        '{"sustained": true, "alternative_interpretation": "pode referir-se a outro contexto", '
        '"confidence_adjustment": 0.0}'
    )
    critique = DevilsAdvocate(_FixedProvider(answer), VerificationConfig()).verify(_critique())
    assert critique.confidence_score < 0.75


def test_revised_classification_is_applied() -> None:
    answer = (
        '{"sustained": true, "revised_classification": "PARCIALMENTE_CORRETA", '
        '"revised_severity": "BAIXO", "confidence_adjustment": 0.0}'
    )
    critique = DevilsAdvocate(_FixedProvider(answer), VerificationConfig()).verify(_critique())
    assert critique.classification is Classification.PARTIALLY_CORRECT
    assert critique.severity is Severity.LOW


def test_conflicting_sources_change_evidence_status() -> None:
    answer = '{"sustained": true, "conflicting_sources": "outra fonte diz o oposto", "confidence_adjustment": 0.0}'
    critique = DevilsAdvocate(_FixedProvider(answer), VerificationConfig()).verify(_critique())
    assert critique.evidence_status is EvidenceStatus.CONFLICTING


def test_not_supported_verdicts_are_not_attacked() -> None:
    advocate = DevilsAdvocate(_FixedProvider('{"sustained": false}'), VerificationConfig())
    critique = advocate.verify(_critique(Classification.NOT_SUPPORTED))
    assert not critique.dropped
    assert critique.verification is not None and critique.verification.sustained


def test_correct_claims_are_skipped() -> None:
    advocate = DevilsAdvocate(_FixedProvider('{"sustained": false}'), VerificationConfig())
    critique = advocate.verify(_critique(Classification.CORRECT))
    assert critique.verification is None


def test_verification_failure_reduces_confidence() -> None:
    critique = DevilsAdvocate(_FixedProvider("modelo fora do ar"), VerificationConfig()).verify(_critique())
    assert not critique.dropped
    assert critique.confidence_score < 0.75


def test_disabled_verification_is_a_no_op() -> None:
    advocate = DevilsAdvocate(_FixedProvider('{"sustained": false}'), VerificationConfig(enabled=False))
    critiques = advocate.verify_all([_critique()])
    assert not critiques[0].dropped
