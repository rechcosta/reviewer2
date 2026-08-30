"""Segmentation, claim extraction, markers and contradiction detection."""

from __future__ import annotations

from app.config import AnalysisConfig
from app.analysis import (
    ClaimExtractor,
    ContradictionDetector,
    build_windows,
    find_causal_markers,
    find_condition_markers,
    find_universal_markers,
)
from app.embeddings import HashingEmbeddings
from app.llm import HeuristicProvider, LLMProvider
from app.models import Importance, Transcript, TranscriptSegment


class _FixedProvider(LLMProvider):
    """Returns the same JSON answer for every call."""

    name = "fixed"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.model = "fixed"
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None, stop=None):
        self.prompts.append(prompt)
        return self.answer


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def test_windows_group_segments(transcript: Transcript) -> None:
    windows = build_windows(transcript, AnalysisConfig(window_max_chars=200, window_max_seconds=60))
    assert windows
    assert windows[0].segment_ids[0] == 0
    assert windows[0].start == 0.0
    assert all(window.text for window in windows)


def test_windows_split_on_long_pause() -> None:
    segments = [
        TranscriptSegment(segment_id=0, start=0, end=5, text="Primeira ideia.", confidence=1.0),
        TranscriptSegment(segment_id=1, start=20, end=25, text="Segunda ideia.", confidence=1.0),
    ]
    transcript = Transcript(source="x", segments=segments)
    windows = build_windows(transcript, AnalysisConfig(window_pause_seconds=1.0))
    assert len(windows) == 2


def test_windows_respect_max_chars() -> None:
    segments = [
        TranscriptSegment(segment_id=i, start=i, end=i + 0.9, text="Uma frase técnica média. ", confidence=1.0)
        for i in range(20)
    ]
    windows = build_windows(Transcript(source="x", segments=segments), AnalysisConfig(window_max_chars=60))
    assert len(windows) > 1


def test_window_mean_confidence() -> None:
    segments = [
        TranscriptSegment(segment_id=0, start=0, end=1, text="a.", confidence=0.4),
        TranscriptSegment(segment_id=1, start=1, end=2, text="b.", confidence=0.8),
    ]
    windows = build_windows(Transcript(source="x", segments=segments))
    assert windows[0].mean_confidence == 0.6000000000000001 or round(windows[0].mean_confidence, 2) == 0.6


# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #
def test_universal_markers_detected() -> None:
    assert "sempre" in find_universal_markers("O desempenho sempre aumenta.")
    assert "garante" in find_universal_markers("O pipeline garante aceleração.")
    assert find_universal_markers("O desempenho normalmente aumenta.") == []


def test_causal_markers_detected() -> None:
    assert "porque" in find_causal_markers("Aumenta porque executa mais instruções.")
    assert find_causal_markers("Aumenta o desempenho.") == []


def test_condition_markers_are_accent_aware() -> None:
    assert "só" in find_condition_markers("A aceleração só ocorre sem dependências.")
    # English "so" must not be mistaken for Portuguese "só".
    assert "só" not in find_condition_markers("so the pipeline stalls")


def test_markers_ignore_substrings() -> None:
    assert find_universal_markers("o valor de allocation") == []


# --------------------------------------------------------------------------- #
# Claim extraction
# --------------------------------------------------------------------------- #
def test_claims_have_ids_timestamps_and_markers(transcript: Transcript, llm: HeuristicProvider) -> None:
    claims = ClaimExtractor(llm, AnalysisConfig()).extract(transcript)
    assert claims
    assert [claim.claim_id for claim in claims] == [f"claim_{i + 1:03d}" for i in range(len(claims))]
    universal = [claim for claim in claims if claim.universal_markers]
    assert universal and universal[0].timestamp == "00:00:12"
    assert all(claim.transcript_confidence > 0 for claim in claims)


def test_claim_timestamps_come_from_segments(transcript: Transcript, llm: HeuristicProvider) -> None:
    claims = ClaimExtractor(llm, AnalysisConfig()).extract(transcript)
    starts = {segment.start for segment in transcript.segments}
    assert all(claim.start in starts or claim.start == 0.0 for claim in claims)


def test_duplicate_claims_are_removed(transcript: Transcript) -> None:
    answer = (
        '{"claims": [{"text": "A cache é volátil.", "topic": "cache", "context": "", '
        '"importance": "high", "verbatim": ""}, {"text": "a cache e volatil.", "topic": "cache", '
        '"context": "", "importance": "high", "verbatim": ""}]}'
    )
    claims = ClaimExtractor(_FixedProvider(answer), AnalysisConfig()).extract(transcript)
    assert len(claims) == 1
    assert claims[0].importance is Importance.HIGH


def test_short_fragments_are_ignored(transcript: Transcript) -> None:
    answer = '{"claims": [{"text": "sim", "topic": "", "context": "", "importance": "low"}]}'
    assert ClaimExtractor(_FixedProvider(answer), AnalysisConfig(min_claim_chars=15)).extract(transcript) == []


def test_extraction_survives_a_broken_window(transcript: Transcript) -> None:
    claims = ClaimExtractor(_FixedProvider("modelo fora do ar"), AnalysisConfig()).extract(transcript)
    assert claims == []  # failures are logged, never raised


# --------------------------------------------------------------------------- #
# Internal contradictions
# --------------------------------------------------------------------------- #
def test_candidate_pairs_include_shared_topic(transcript: Transcript, llm: HeuristicProvider) -> None:
    claims = ClaimExtractor(llm, AnalysisConfig()).extract(transcript)
    detector = ContradictionDetector(llm, HashingEmbeddings(256), AnalysisConfig())
    pairs = {(a.claim_id, b.claim_id) for a, b, _ in detector.candidate_pairs(claims)}
    cache_claims = [claim.claim_id for claim in claims if "cache" in claim.text.lower()]
    assert len(cache_claims) >= 2
    assert (cache_claims[0], cache_claims[1]) in pairs


def test_resolving_condition_cancels_a_contradiction(transcript: Transcript) -> None:
    answer = (
        '{"is_contradiction": true, "explanation": "conflito aparente", "same_sense_of_terms": true, '
        '"context_difference": "", "temporal_difference": "", '
        '"resolving_condition": "uma delas vale apenas com energia ligada", '
        '"confidence": "ALTA", "severity": "MEDIO"}'
    )
    provider = _FixedProvider(answer)
    claims = ClaimExtractor(HeuristicProvider(), AnalysisConfig()).extract(transcript)
    detector = ContradictionDetector(provider, HashingEmbeddings(256), AnalysisConfig())
    assert detector.detect(claims) == []


def test_different_sense_of_terms_cancels_a_contradiction(transcript: Transcript) -> None:
    answer = (
        '{"is_contradiction": true, "explanation": "x", "same_sense_of_terms": false, '
        '"resolving_condition": "", "confidence": "ALTA", "severity": "ALTO"}'
    )
    claims = ClaimExtractor(HeuristicProvider(), AnalysisConfig()).extract(transcript)
    detector = ContradictionDetector(_FixedProvider(answer), HashingEmbeddings(256), AnalysisConfig())
    assert detector.detect(claims) == []


def test_confirmed_contradiction_is_returned(transcript: Transcript) -> None:
    answer = (
        '{"is_contradiction": true, "explanation": "não volátil vs. dados perdidos", '
        '"same_sense_of_terms": true, "resolving_condition": "", "confidence": "ALTA", "severity": "ALTO"}'
    )
    claims = ClaimExtractor(HeuristicProvider(), AnalysisConfig()).extract(transcript)
    detector = ContradictionDetector(_FixedProvider(answer), HashingEmbeddings(256), AnalysisConfig())
    contradictions = detector.detect(claims)
    assert contradictions
    assert contradictions[0].is_contradiction
    assert contradictions[0].contradiction_id.startswith("contradiction_")


def test_single_claim_has_no_contradictions(llm: HeuristicProvider) -> None:
    detector = ContradictionDetector(llm, HashingEmbeddings(64), AnalysisConfig())
    assert detector.detect([]) == []
