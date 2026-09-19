"""Critique engine: RAG + evidence-bound judgement of each claim.

Two guarantees are enforced here, in code rather than in the prompt:

* a critique may only cite chunks that were actually retrieved for it;
* every quote must appear verbatim in the corresponding chunk, otherwise the
  evidence is discarded and the verdict is downgraded.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..config import Config
from ..errors import LLMError
from ..llm import LLMProvider, as_dict, as_list
from ..logging_utils import get_logger
from ..models import (
    Claim,
    Classification,
    CompatibilityVerdict,
    ConfidenceLevel,
    Critique,
    DocumentChunk,
    Evidence,
    EvidenceRelation,
    EvidenceStatus,
    ImpactLevel,
    Omission,
    RetrievalTrace,
    RetrievedChunk,
    Severity,
    StatementType,
)
from ..prompts import SYSTEM_PROMPT, critique_prompt
from ..reports.i18n import strings
from ..retrieval import EvidenceRetriever
from .confidence import calibrate
from ..text_utils import split_sentences
from .markers import strip_accents

logger = get_logger(__name__)



class CritiqueEngine:
    """Produces one :class:`Critique` per claim, always backed by evidence."""

    def __init__(self, llm: LLMProvider, retriever: EvidenceRetriever, config: Config) -> None:
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.strings = strings(config.report.language)

    # ------------------------------------------------------------------ #
    def review_claims(self, claims: Sequence[Claim]) -> List[Critique]:
        """Review every claim against the indexed reference material.

        This is by far the longest stage — one model call per claim — so it
        reports progress per claim, and runs several claims at once when
        ``llm.concurrency`` is above one. Results keep the original order
        regardless of the order they finish in.
        """
        claims = list(claims)
        total = len(claims)
        workers = max(1, int(self.config.llm.concurrency))

        if workers == 1 or total <= 1:
            critiques: List[Critique] = []
            for index, claim in enumerate(claims, start=1):
                logger.info("Reviewing claim %d/%d (%s)", index, total, claim.claim_id)
                critiques.append(self.review_claim(claim))
            return critiques

        logger.info("Reviewing %d claims with %d parallel model calls", total, workers)
        done = itertools.count(1)
        results: List[Optional[Critique]] = [None] * total

        def work(position: int) -> None:
            claim = claims[position]
            try:
                results[position] = self.review_claim(claim)
            except Exception:
                # One claim that blows up must not throw away the critiques of
                # every other claim — on CPU those cost minutes each. The claim
                # is dropped from the report, which the coverage statistic and
                # the limitations section already account for.
                logger.exception("Review failed for claim %s; it is left out", claim.claim_id)
            logger.info("Reviewed claim %d/%d (%s)", next(done), total, claim.claim_id)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="critique") as pool:
            list(pool.map(work, range(total)))

        return [critique for critique in results if critique is not None]

    def review_claim(self, claim: Claim) -> Critique:
        """Retrieve evidence for one claim and judge it, reusing the cache."""
        trace = self.retriever.retrieve(_query_for(claim), claim_id=claim.claim_id)
        scores = {item.chunk.chunk_id: item.score for item in trace.results}

        if not trace.results:
            return self._no_evidence(claim, trace)

        excerpts, sentences = _build_excerpts(trace.results)

        prompt = critique_prompt(
            claim.text,
            excerpts,
            claim_context=claim.context,
            transcript_note=_transcript_note(claim, self.strings),
            universal_markers=claim.universal_markers,
            causal_markers=claim.causal_markers,
            language=self.config.analysis.language,
        )
        try:
            raw = as_dict(self.llm.generate_json(prompt, system=SYSTEM_PROMPT, expect="object"))
        except LLMError as exc:
            logger.warning("Critique failed for %s: %s", claim.claim_id, exc.message)
            return self._no_evidence(claim, trace, reason=self.strings["cr_llm_failed"])

        evidence = _build_evidence(
            raw.get("evidence"), sentences, scores, limit=self.config.retrieval.max_evidence_per_claim
        )
        verified = [item for item in evidence if item.verified]

        classification = Classification.parse(raw.get("classification"), Classification.NOT_SUPPORTED)
        severity = Severity.parse(raw.get("severity"), Severity.LOW)
        model_confidence = ConfidenceLevel.parse(raw.get("confidence"), ConfidenceLevel.LOW)
        evidence_status = EvidenceStatus.parse(raw.get("evidence_status"), EvidenceStatus.UNKNOWN)
        statement_type = StatementType.parse(raw.get("statement_type"), StatementType.INFERENCE)
        compatibility = CompatibilityVerdict.parse(raw.get("compatibility"), CompatibilityVerdict.UNDETERMINED)

        analysis = str(raw.get("analysis", "")).strip()
        problem = str(raw.get("problem", "")).strip()

        # Anti-hallucination guard: a verdict about the sources requires at
        # least one quote that really exists in the retrieved excerpts.
        if classification.is_problem and not verified:
            logger.debug("Downgrading %s: no verifiable quote in the critique", claim.claim_id)
            analysis = (
                f"{analysis}\n\n{self.strings['cr_downgraded']}"
            ).strip()
            classification = Classification.NOT_SUPPORTED
            severity = Severity.LOW
            evidence_status = EvidenceStatus.INSUFFICIENT
            compatibility = CompatibilityVerdict.UNDETERMINED
            problem = ""

        if classification is Classification.CORRECT and not verified:
            evidence_status = EvidenceStatus.INSUFFICIENT

        confidence, score, factors = calibrate(
            claim=claim,
            model_confidence=model_confidence,
            evidence=evidence,
            trace=trace,
            evidence_status=evidence_status,
        )

        critique = Critique(
            claim=claim,
            classification=classification,
            severity=severity if classification.is_problem else _neutral_severity(classification),
            confidence=confidence,
            confidence_score=score,
            evidence_status=evidence_status,
            statement_type=statement_type,
            compatibility=compatibility,
            evidence=evidence,
            analysis=analysis or self.strings["cr_undetermined"],
            problem=problem,
            suggested_correction=str(raw.get("suggested_correction", "")).strip(),
            conclusion=str(raw.get("conclusion", "")).strip(),
            retrieval_trace=trace,
            confidence_factors=factors,
        )

        if self.config.analysis.detect_omissions and verified:
            # Omissions come back in the same answer as the critique: on CPU an
            # extra round trip per claim costs more than the whole analysis.
            critique.omissions = self._read_omissions(claim, raw.get("omissions"), sentences, scores)

        return critique

    # ------------------------------------------------------------------ #
    def _read_omissions(
        self,
        claim: Claim,
        raw_omissions: object,
        sentences: Dict[str, "EvidenceSentence"],
        scores: Dict[str, float],
    ) -> List[Omission]:
        """Turn the critique's omission entries into verified :class:`Omission` objects."""
        omissions: List[Omission] = []
        for item in as_list(raw_omissions):
            if not isinstance(item, dict):
                continue
            evidence = _build_evidence(
                [{"id": item.get("id"), "relation": "PARTIAL"}], sentences, scores
            )
            if not any(entry.verified for entry in evidence):
                continue  # an omission without a resolvable source sentence is not reported
            missing = str(item.get("missing_information", "")).strip() or evidence[0].quote
            omissions.append(
                Omission(
                    claim_id=claim.claim_id,
                    missing_information=missing,
                    impact=ImpactLevel.parse(item.get("impact")),
                    evidence=evidence,
                    explanation=str(item.get("explanation", "")).strip(),
                )
            )
        return omissions

    def _no_evidence(self, claim: Claim, trace: RetrievalTrace, *, reason: str = "") -> Critique:
        """Deterministic verdict when retrieval found nothing usable."""
        confidence, score, factors = calibrate(
            claim=claim,
            model_confidence=ConfidenceLevel.LOW,
            evidence=[],
            trace=trace,
            evidence_status=EvidenceStatus.INSUFFICIENT,
        )
        analysis = reason or self.strings["cr_no_retrieval"].format(threshold=trace.threshold)
        return Critique(
            claim=claim,
            classification=Classification.NOT_SUPPORTED,
            severity=Severity.LOW,
            confidence=confidence,
            confidence_score=score,
            evidence_status=EvidenceStatus.INSUFFICIENT,
            statement_type=StatementType.INFERENCE,
            compatibility=CompatibilityVerdict.UNDETERMINED,
            evidence=[],
            analysis=analysis,
            problem="",
            suggested_correction="",
            conclusion=self.strings["cr_no_evidence_conclusion"],
            retrieval_trace=trace,
            confidence_factors=factors,
        )


# --------------------------------------------------------------------------- #
# Evidence handling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvidenceSentence:
    """One citable sentence of a retrieved chunk, addressed by a short id."""

    id: str
    text: str
    chunk: DocumentChunk


def _build_excerpts(
    results: Sequence[RetrievedChunk],
) -> tuple[List[Dict[str, object]], Dict[str, EvidenceSentence]]:
    """Split retrieved chunks into identified sentences the model can cite.

    Returns the payload for the prompt and the id → sentence map used to
    resolve the model's answer back to the exact source text.
    """
    excerpts: List[Dict[str, object]] = []
    sentences: Dict[str, EvidenceSentence] = {}
    counter = 0

    for item in results:
        chunk = item.chunk
        chunk_sentences: List[Dict[str, str]] = []
        for sentence in split_sentences(chunk.text) or [chunk.text]:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue  # too short to carry evidence on its own
            counter += 1
            identifier = f"e{counter}"
            sentences[identifier] = EvidenceSentence(identifier, sentence, chunk)
            chunk_sentences.append({"id": identifier, "text": sentence})
        if not chunk_sentences:
            continue
        excerpts.append(
            {
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "page": chunk.page,
                "section": chunk.section,
                "score": round(item.score, 3),
                "sentences": chunk_sentences,
            }
        )
    return excerpts, sentences


def _build_evidence(
    raw_evidence: object,
    sentences: Dict[str, EvidenceSentence],
    scores: Dict[str, float],
    *,
    limit: int = 0,
) -> List[Evidence]:
    """Resolve the ids the model cited back into verified evidence.

    The model never writes a quote: it points at a sentence that was given to
    it, and the text is taken from the source here. A fabricated citation is
    therefore impossible — an unknown id is simply dropped. A quote written
    out anyway is still accepted, but only if it occurs verbatim in a
    retrieved sentence.
    """
    evidence: List[Evidence] = []
    seen: set[str] = set()

    for item in as_list(raw_evidence):
        if not isinstance(item, dict):
            continue
        sentence = _resolve_sentence(item, sentences)
        if sentence is None:
            logger.debug("Discarding evidence that cites no known sentence: %r", item)
            continue
        if sentence.id in seen:
            continue
        seen.add(sentence.id)
        chunk = sentence.chunk
        evidence.append(
            Evidence(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                page=chunk.page,
                section=chunk.section,
                quote=sentence.text,
                score=scores.get(chunk.chunk_id, 0.0),
                relation=EvidenceRelation.parse(item.get("relation"), EvidenceRelation.NEUTRAL),
                verified=True,
            )
        )

    if limit and len(evidence) > limit:
        evidence.sort(key=lambda entry: -entry.score)
        evidence = evidence[:limit]
    return evidence


def _resolve_sentence(
    item: Dict[str, object], sentences: Dict[str, EvidenceSentence]
) -> Optional[EvidenceSentence]:
    """Find the sentence an evidence entry refers to, by id or by quote."""
    for key in ("id", "sentence_id", "evidence_id"):
        identifier = str(item.get(key, "")).strip()
        if identifier in sentences:
            return sentences[identifier]

    # Some models write the sentence out instead of citing its id; accept it
    # only when it really is one of the sentences we provided.
    quote = str(item.get("quote", "")).strip().strip('"')
    if len(quote) >= 12:
        needle = _flatten(quote)
        for sentence in sentences.values():
            haystack = _flatten(sentence.text)
            if needle in haystack or haystack in needle:
                return sentence
    return None


def verify_quote(quote: str, chunk_text: str) -> tuple[bool, str]:
    """Check that ``quote`` really occurs in ``chunk_text``.

    Comparison ignores case, accents and whitespace differences, because ASR
    and PDF extraction introduce those routinely. Returns the verification
    flag and the quote to display (the original chunk wording when found).
    """
    if not quote:
        return False, ""
    needle = _flatten(quote)
    haystack = _flatten(chunk_text)
    if len(needle) < 12:
        return False, quote
    if needle in haystack:
        return True, quote
    # Tolerate a truncated or slightly reworded tail: require a long prefix match.
    prefix = needle[: max(40, int(len(needle) * 0.6))]
    if prefix and prefix in haystack:
        return True, quote
    return False, quote


def _flatten(text: str) -> str:
    return " ".join(strip_accents(text).split())


def _query_for(claim: Claim) -> str:
    """Build the retrieval query for a claim (claim plus its topic)."""
    topic = claim.topic.strip()
    return f"{claim.text} {topic}".strip() if topic else claim.text


def _transcript_note(claim: Claim, text: Dict[str, str]) -> str:
    """Warn the model when the transcription of this claim is unreliable."""
    if claim.low_confidence_transcript:
        return text["cr_transcript_note"].format(confidence=claim.transcript_confidence)
    return ""


def _neutral_severity(classification: Classification) -> Severity:
    """Severity used for verdicts that are not problems."""
    if classification is Classification.PEDAGOGICAL_SIMPLIFICATION:
        return Severity.PEDAGOGICAL
    return Severity.LOW
