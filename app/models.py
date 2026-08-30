"""Shared domain models.

These Pydantic models are the vocabulary of Reviewer2: transcripts, claims,
evidence, critiques and the final report. Enum *members* are English (the
code is English), while their *values* are the canonical Portuguese labels
used by the report and by the LLM prompts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Classification(str, Enum):
    """Verdict assigned to a claim."""

    CORRECT = "CORRETA"
    PARTIALLY_CORRECT = "PARCIALMENTE_CORRETA"
    INCORRECT = "INCORRETA"
    IMPRECISE = "IMPRECISA"
    CONTRADICTORY = "CONTRADITORIA"
    NOT_SUPPORTED = "NAO_SUSTENTADA"
    PEDAGOGICAL_SIMPLIFICATION = "SIMPLIFICACAO_PEDAGOGICA"

    @classmethod
    def parse(cls, raw: Any, default: "Classification" = None) -> "Classification":
        """Parse a (possibly noisy) LLM string into a Classification."""
        return _parse_enum(cls, raw, default or cls.NOT_SUPPORTED)

    @property
    def is_problem(self) -> bool:
        """True when the classification denotes something worth reporting."""
        return self not in (Classification.CORRECT, Classification.PEDAGOGICAL_SIMPLIFICATION)


class Severity(str, Enum):
    """How damaging a problem is."""

    CRITICAL = "CRITICO"
    HIGH = "ALTO"
    MEDIUM = "MEDIO"
    LOW = "BAIXO"
    PEDAGOGICAL = "PEDAGOGICO"

    @classmethod
    def parse(cls, raw: Any, default: "Severity" = None) -> "Severity":
        return _parse_enum(cls, raw, default or cls.LOW)

    @property
    def rank(self) -> int:
        """Sort key: lower is more severe."""
        return _SEVERITY_RANK[self]


class ConfidenceLevel(str, Enum):
    """How much trust the system places in its own analysis."""

    VERY_HIGH = "MUITO_ALTA"
    HIGH = "ALTA"
    MEDIUM = "MEDIA"
    LOW = "BAIXA"

    @classmethod
    def parse(cls, raw: Any, default: "ConfidenceLevel" = None) -> "ConfidenceLevel":
        return _parse_enum(cls, raw, default or cls.LOW)

    @property
    def score(self) -> float:
        """Numeric value in [0, 1] used by the confidence calibrator."""
        return _CONFIDENCE_SCORE[self]

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Map a numeric score back to a level."""
        if score >= 0.85:
            return cls.VERY_HIGH
        if score >= 0.65:
            return cls.HIGH
        if score >= 0.45:
            return cls.MEDIUM
        return cls.LOW


class EvidenceStatus(str, Enum):
    """Anti-hallucination markers required by the project specification."""

    OK = "OK"
    UNKNOWN = "UNKNOWN"
    CONFLICTING = "CONFLICTING_EVIDENCE"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

    @classmethod
    def parse(cls, raw: Any, default: "EvidenceStatus" = None) -> "EvidenceStatus":
        return _parse_enum(cls, raw, default or cls.UNKNOWN)


class StatementType(str, Enum):
    """Separation between fact, inference, interpretation and opinion."""

    FACT = "FATO"
    INFERENCE = "INFERENCIA"
    INTERPRETATION = "INTERPRETACAO"
    OPINION = "OPINIAO"

    @classmethod
    def parse(cls, raw: Any, default: "StatementType" = None) -> "StatementType":
        return _parse_enum(cls, raw, default or cls.INFERENCE)


class Importance(str, Enum):
    """How central a claim is to the explanation."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def parse(cls, raw: Any, default: "Importance" = None) -> "Importance":
        return _parse_enum(cls, raw, default or cls.MEDIUM)


class ImpactLevel(str, Enum):
    """Impact of an omission."""

    HIGH = "ALTA"
    MEDIUM = "MEDIA"
    LOW = "BAIXA"

    @classmethod
    def parse(cls, raw: Any, default: "ImpactLevel" = None) -> "ImpactLevel":
        return _parse_enum(cls, raw, default or cls.MEDIUM)


class EvidenceRelation(str, Enum):
    """Relation between one evidence excerpt and the claim."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PARTIAL = "PARTIAL"
    NEUTRAL = "NEUTRAL"

    @classmethod
    def parse(cls, raw: Any, default: "EvidenceRelation" = None) -> "EvidenceRelation":
        return _parse_enum(cls, raw, default or cls.NEUTRAL)


class CompatibilityVerdict(str, Enum):
    """Result of comparing a claim against a source (section 14)."""

    COMPATIBLE = "SIM"
    INCOMPATIBLE = "NAO"
    PARTIAL = "PARCIAL"
    UNDETERMINED = "INDETERMINADO"

    @classmethod
    def parse(cls, raw: Any, default: "CompatibilityVerdict" = None) -> "CompatibilityVerdict":
        return _parse_enum(cls, raw, default or cls.UNDETERMINED)


_SEVERITY_RANK: Dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.PEDAGOGICAL: 4,
}

_CONFIDENCE_SCORE: Dict[ConfidenceLevel, float] = {
    ConfidenceLevel.VERY_HIGH: 0.92,
    ConfidenceLevel.HIGH: 0.75,
    ConfidenceLevel.MEDIUM: 0.55,
    ConfidenceLevel.LOW: 0.30,
}

# Accent-insensitive aliases per enum, so that model output such as
# "partially correct", "MÉDIO" or "very high" is still understood. The tables
# are scoped per enum because the same English word maps to different tokens
# in different enums (e.g. "HIGH" is ALTO for severity but "high" for
# importance).
_ALIASES: Dict[str, Dict[str, str]] = {
    "Classification": {
        "CORRECT": "CORRETA",
        "PARTIALLY_CORRECT": "PARCIALMENTE_CORRETA",
        "PARTLY_CORRECT": "PARCIALMENTE_CORRETA",
        "INCORRECT": "INCORRETA",
        "WRONG": "INCORRETA",
        "IMPRECISE": "IMPRECISA",
        "INACCURATE": "IMPRECISA",
        "CONTRADICTORY": "CONTRADITORIA",
        "NOT_SUPPORTED": "NAO_SUSTENTADA",
        "UNSUPPORTED": "NAO_SUSTENTADA",
        "PEDAGOGICAL_SIMPLIFICATION": "SIMPLIFICACAO_PEDAGOGICA",
        "SIMPLIFICATION": "SIMPLIFICACAO_PEDAGOGICA",
    },
    "Severity": {
        "CRITICAL": "CRITICO",
        "HIGH": "ALTO",
        "MEDIUM": "MEDIO",
        "LOW": "BAIXO",
        "PEDAGOGICAL": "PEDAGOGICO",
    },
    "ConfidenceLevel": {
        "VERY_HIGH": "MUITO_ALTA",
        "HIGH": "ALTA",
        "MEDIUM": "MEDIA",
        "LOW": "BAIXA",
    },
    "StatementType": {
        "FACT": "FATO",
        "INFERENCE": "INFERENCIA",
        "INTERPRETATION": "INTERPRETACAO",
        "OPINION": "OPINIAO",
    },
    "ImpactLevel": {
        "HIGH": "ALTA",
        "MEDIUM": "MEDIA",
        "LOW": "BAIXA",
    },
    "Importance": {
        "ALTA": "HIGH",
        "MEDIA": "MEDIUM",
        "BAIXA": "LOW",
        "ALTO": "HIGH",
        "MEDIO": "MEDIUM",
        "BAIXO": "LOW",
    },
    "CompatibilityVerdict": {
        "YES": "SIM",
        "COMPATIBLE": "SIM",
        "NO": "NAO",
        "INCOMPATIBLE": "NAO",
        "PARTIAL": "PARCIAL",
        "PARTIALLY": "PARCIAL",
        "UNDETERMINED": "INDETERMINADO",
        "UNKNOWN": "INDETERMINADO",
    },
    "EvidenceRelation": {
        "SUPPORT": "SUPPORTS",
        "SUPPORTED": "SUPPORTS",
        "CONTRADICT": "CONTRADICTS",
        "REFUTES": "CONTRADICTS",
        "PARTIALLY": "PARTIAL",
    },
    "EvidenceStatus": {
        "SUFFICIENT": "OK",
        "CONFLICTING": "CONFLICTING_EVIDENCE",
        "INSUFFICIENT": "INSUFFICIENT_EVIDENCE",
    },
}


def normalize_token(raw: Any) -> str:
    """Uppercase, strip accents and collapse separators of an enum-ish token."""
    import unicodedata

    text = str(raw or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.upper().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def parse_optional(enum_cls, raw: Any):
    """Parse ``raw`` into an enum member, or return ``None`` when unrecognised.

    Used where an unparseable value must not silently become a default —
    for instance a revised classification proposed by the verification pass.
    """
    if raw in (None, ""):
        return None
    sentinel = object()
    parsed = _parse_enum(enum_cls, raw, sentinel)
    return None if parsed is sentinel else parsed


def _parse_enum(enum_cls, raw: Any, default):
    """Best-effort conversion of arbitrary model output into an enum member."""
    if isinstance(raw, enum_cls):
        return raw
    token = normalize_token(raw)
    if not token:
        return default
    token = _ALIASES.get(enum_cls.__name__, {}).get(token, token)
    for member in enum_cls:
        if normalize_token(member.value) == token or member.name == token:
            return member
    # Last resort: unique prefix match (e.g. "PARCIALMENTE").
    matches = [m for m in enum_cls if normalize_token(m.value).startswith(token[:6])]
    if len(matches) == 1:
        return matches[0]
    return default


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS``."""
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Word(BaseModel):
    """A single timed word, when the ASR backend provides word timestamps."""

    word: str
    start: float
    end: float
    probability: float = 1.0


class TranscriptSegment(BaseModel):
    """A timed chunk of speech produced by the ASR stage."""

    segment_id: int = 0
    start: float
    end: float
    text: str
    confidence: float = 1.0
    low_confidence: bool = False
    words: List[Word] = Field(default_factory=list)

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    """Full transcription of the analysed media."""

    source: str
    language: Optional[str] = None
    duration: float = 0.0
    model: str = "unknown"
    segments: List[TranscriptSegment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def text(self) -> str:
        """The whole transcript as plain text."""
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())

    @property
    def mean_confidence(self) -> float:
        if not self.segments:
            return 0.0
        return sum(s.confidence for s in self.segments) / len(self.segments)

    @property
    def low_confidence_ratio(self) -> float:
        """Share of segments flagged as low confidence by the ASR stage."""
        if not self.segments:
            return 0.0
        return sum(1 for s in self.segments if s.low_confidence) / len(self.segments)

    def segment_by_id(self, segment_id: int) -> Optional[TranscriptSegment]:
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        return None

    def window(self, start: float, end: float) -> List[TranscriptSegment]:
        """Segments overlapping the ``[start, end]`` interval."""
        return [s for s in self.segments if s.end >= start and s.start <= end]


class TranscriptWindow(BaseModel):
    """A semantically grouped set of consecutive segments."""

    window_id: int
    start: float
    end: float
    text: str
    segment_ids: List[int] = Field(default_factory=list)
    mean_confidence: float = 1.0

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)


# --------------------------------------------------------------------------- #
# Documents / retrieval
# --------------------------------------------------------------------------- #
class DocumentChunk(BaseModel):
    """A retrievable excerpt of a reference document, with full provenance."""

    chunk_id: str
    document: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    source_path: str = ""

    @property
    def locator(self) -> str:
        """Human readable ``document, page N, section`` locator."""
        parts = [self.document]
        if self.page is not None:
            parts.append(f"página {self.page}")
        if self.section:
            parts.append(str(self.section))
        return ", ".join(parts)


class RetrievedChunk(BaseModel):
    """A chunk returned by the vector store, together with its score."""

    chunk: DocumentChunk
    score: float


class RetrievalTrace(BaseModel):
    """Audit record explaining why a given conclusion was produced."""

    query: str
    claim_id: Optional[str] = None
    results: List[RetrievedChunk] = Field(default_factory=list)
    threshold: float = 0.0
    top_k: int = 0

    @property
    def best_score(self) -> float:
        return max((r.score for r in self.results), default=0.0)


# --------------------------------------------------------------------------- #
# Claims, evidence and critiques
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    """An analysable technical assertion extracted from the transcript."""

    claim_id: str
    text: str
    timestamp: str = "00:00:00"
    start: float = 0.0
    end: float = 0.0
    topic: str = ""
    context: str = ""
    importance: Importance = Importance.MEDIUM
    segment_ids: List[int] = Field(default_factory=list)
    verbatim: str = ""
    transcript_confidence: float = 1.0
    low_confidence_transcript: bool = False
    universal_markers: List[str] = Field(default_factory=list)
    causal_markers: List[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """One excerpt of a source used to judge a claim."""

    chunk_id: str
    document: str
    page: Optional[int] = None
    section: Optional[str] = None
    quote: str
    score: float = 0.0
    relation: EvidenceRelation = EvidenceRelation.NEUTRAL
    verified: bool = False

    @property
    def locator(self) -> str:
        parts = [self.document]
        if self.page is not None:
            parts.append(f"página {self.page}")
        if self.section:
            parts.append(str(self.section))
        return ", ".join(parts)


class Omission(BaseModel):
    """A relevant condition present in the sources but missing in the video."""

    claim_id: str
    missing_information: str
    impact: ImpactLevel = ImpactLevel.MEDIUM
    evidence: List[Evidence] = Field(default_factory=list)
    explanation: str = ""


class VerificationResult(BaseModel):
    """Outcome of the devil's advocate pass over a critique."""

    sustained: bool = True
    alternative_interpretation: str = ""
    assumptions: List[str] = Field(default_factory=list)
    context_considered: bool = True
    source_supports_critique: bool = True
    conditions_that_would_validate_claim: str = ""
    transcription_risk: str = ""
    conflicting_sources: str = ""
    notes: str = ""
    confidence_adjustment: float = 0.0
    revised_classification: Optional[Classification] = None
    revised_severity: Optional[Severity] = None


class Critique(BaseModel):
    """The full evidence chain for a single claim."""

    model_config = ConfigDict(use_enum_values=False)

    claim: Claim
    classification: Classification = Classification.NOT_SUPPORTED
    severity: Severity = Severity.LOW
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    confidence_score: float = 0.0
    evidence_status: EvidenceStatus = EvidenceStatus.INSUFFICIENT
    statement_type: StatementType = StatementType.INFERENCE
    compatibility: CompatibilityVerdict = CompatibilityVerdict.UNDETERMINED
    evidence: List[Evidence] = Field(default_factory=list)
    analysis: str = ""
    problem: str = ""
    suggested_correction: str = ""
    conclusion: str = ""
    omissions: List[Omission] = Field(default_factory=list)
    verification: Optional[VerificationResult] = None
    retrieval_trace: Optional[RetrievalTrace] = None
    dropped: bool = False
    dropped_reason: str = ""
    confidence_factors: Dict[str, float] = Field(default_factory=dict)

    @property
    def is_problem(self) -> bool:
        return (not self.dropped) and self.classification.is_problem

    @property
    def has_verified_evidence(self) -> bool:
        return any(e.verified for e in self.evidence)


class InternalContradiction(BaseModel):
    """A possible contradiction between two claims of the same video."""

    contradiction_id: str
    claim_a: Claim
    claim_b: Claim
    is_contradiction: bool = False
    explanation: str = ""
    same_sense_of_terms: bool = True
    context_difference: str = ""
    temporal_difference: str = ""
    resolving_condition: str = ""
    confidence: ConfidenceLevel = ConfidenceLevel.LOW
    severity: Severity = Severity.MEDIUM
    similarity: float = 0.0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
class QualityVerdict(str, Enum):
    """Actionable verdict on the recording as a whole."""

    PUBLISHABLE = "PODE_PUBLICAR"
    MINOR_FIXES = "AJUSTES_MENORES"
    NEEDS_CORRECTION = "PRECISA_CORRECAO"
    NEEDS_RERECORDING = "REGRAVAR_TRECHO"

    @property
    def advice(self) -> str:
        """One line telling the author what to do next."""
        return {
            QualityVerdict.PUBLISHABLE:
                "Nenhum erro técnico sustentado por evidência. Pode publicar.",
            QualityVerdict.MINOR_FIXES:
                "Sem erros graves. Considere ajustar a formulação dos pontos abaixo, "
                "ou mencioná-los na descrição do vídeo.",
            QualityVerdict.NEEDS_CORRECTION:
                "Há afirmações que podem levar a compreensão técnica incorreta. "
                "Corrija-as com uma errata ou uma anotação no vídeo.",
            QualityVerdict.NEEDS_RERECORDING:
                "Há erro que compromete a explicação. Recomenda-se regravar o trecho indicado.",
        }[self]


class ReviewStatistics(BaseModel):
    """Aggregated numbers shown in the report and used by future metrics."""

    total_segments: int = 0
    total_claims: int = 0
    analysed_claims: int = 0
    dropped_critiques: int = 0
    by_classification: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)
    by_confidence: Dict[str, int] = Field(default_factory=dict)
    total_documents: int = 0
    total_chunks: int = 0
    total_defects: int = 0
    total_undetermined: int = 0
    total_omissions: int = 0
    total_internal_contradictions: int = 0
    evidence_coverage: float = 0.0
    mean_transcript_confidence: float = 0.0


class ReviewReport(BaseModel):
    """Everything produced by one run of the pipeline."""

    video: str
    references: List[str] = Field(default_factory=list)
    transcript: Optional[Transcript] = None
    claims: List[Claim] = Field(default_factory=list)
    critiques: List[Critique] = Field(default_factory=list)
    internal_contradictions: List[InternalContradiction] = Field(default_factory=list)
    statistics: ReviewStatistics = Field(default_factory=ReviewStatistics)
    limitations: List[str] = Field(default_factory=list)
    llm_model: str = "unknown"
    asr_model: str = "unknown"
    embedding_model: str = "unknown"
    vector_store: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = 0.0

    @property
    def active_critiques(self) -> List[Critique]:
        """Critiques that survived the devil's advocate pass."""
        return [c for c in self.critiques if not c.dropped]

    @property
    def problems(self) -> List[Critique]:
        """Critiques that report an actual problem, most severe first."""
        problems = [c for c in self.active_critiques if c.is_problem]
        return sorted(problems, key=lambda c: (c.severity.rank, -c.confidence_score))

    @property
    def defects(self) -> List[Critique]:
        """Actual defects in the explanation, most severe first.

        Excludes NAO_SUSTENTADA: "the sources do not cover this" is a limit of
        the reference material, not something the presenter got wrong. Mixing
        the two sends the author chasing problems that do not exist.
        """
        return [c for c in self.problems if c.classification is not Classification.NOT_SUPPORTED]

    @property
    def undetermined(self) -> List[Critique]:
        """Claims the provided sources could not settle either way."""
        return [
            c for c in self.active_critiques
            if c.classification is Classification.NOT_SUPPORTED
        ]

    @property
    def quality(self) -> "QualityVerdict":
        """Overall verdict on the recording, from the defects found."""
        critical = sum(1 for c in self.defects if c.severity is Severity.CRITICAL)
        high = sum(1 for c in self.defects if c.severity is Severity.HIGH)
        medium = sum(1 for c in self.defects if c.severity is Severity.MEDIUM)

        if critical:
            return QualityVerdict.NEEDS_RERECORDING
        if high:
            return QualityVerdict.NEEDS_CORRECTION
        if medium:
            return QualityVerdict.MINOR_FIXES
        return QualityVerdict.PUBLISHABLE
