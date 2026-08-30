"""Report rendering."""

from __future__ import annotations

from pathlib import Path

from app.config import ReportConfig
from app.models import (
    Claim,
    Classification,
    ConfidenceLevel,
    Critique,
    DocumentChunk,
    Evidence,
    EvidenceRelation,
    EvidenceStatus,
    InternalContradiction,
    Omission,
    ImpactLevel,
    ReviewReport,
    RetrievalTrace,
    RetrievedChunk,
    Severity,
    Transcript,
    TranscriptSegment,
)
from app.reports import ReportGenerator, build_statistics

SECTION_TITLES_PT = [
    "1. Resumo Executivo",
    "2. Informações da Análise",
    "3. Estatísticas",
    "4. Principais Problemas",
    "5. Análise das Afirmações",
    "6. Contradições Internas",
    "7. Contradições com as Fontes",
    "8. Omissões Relevantes",
    "9. Simplificações Pedagógicas",
    "10. Pontos Tecnicamente Corretos",
    "11. Correções Sugeridas",
    "12. Avaliação Final",
    "13. Limitações da Análise",
]


def _report() -> ReviewReport:
    claim = Claim(
        claim_id="claim_001",
        text="Aumentar a frequência sempre aumenta o desempenho.",
        timestamp="00:04:31",
        transcript_confidence=0.9,
    )
    correct_claim = Claim(claim_id="claim_002", text="A cache é volátil.", timestamp="00:05:00")
    chunk = DocumentChunk(chunk_id="p_0007_00", document="paper.pdf", text="Performance becomes limited.", page=7)
    evidence = Evidence(
        chunk_id="p_0007_00", document="paper.pdf", page=7, section="Results",
        quote="Performance becomes limited.", score=0.71,
        relation=EvidenceRelation.PARTIAL, verified=True,
    )
    critique = Critique(
        claim=claim,
        classification=Classification.PARTIALLY_CORRECT,
        severity=Severity.MEDIUM,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.7,
        evidence_status=EvidenceStatus.OK,
        evidence=[evidence],
        analysis="A afirmação é universal; a fonte apresenta uma condição.",
        problem="Generalização indevida.",
        suggested_correction="Aumentar a frequência aumenta o desempenho em cargas limitadas por computação.",
        conclusion="Fundamento correto, formulação universal.",
        omissions=[
            Omission(
                claim_id="claim_001",
                missing_information="O aumento de frequência eleva o consumo de energia.",
                impact=ImpactLevel.HIGH,
                evidence=[evidence],
                explanation="A fonte cita o custo energético.",
            )
        ],
        retrieval_trace=RetrievalTrace(query="frequência", results=[RetrievedChunk(chunk=chunk, score=0.71)]),
    )
    correct = Critique(
        claim=correct_claim,
        classification=Classification.CORRECT,
        severity=Severity.LOW,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.7,
        evidence_status=EvidenceStatus.OK,
        evidence=[evidence],
        analysis="Consistente com a fonte.",
        conclusion="Correta.",
    )
    transcript = Transcript(
        source="aula.mp4",
        segments=[TranscriptSegment(segment_id=0, start=0, end=5, text="olá", confidence=0.9)],
    )
    report = ReviewReport(
        video="data/videos/aula.mp4",
        references=["data/documents/paper.pdf"],
        transcript=transcript,
        claims=[claim, correct_claim],
        critiques=[critique, correct],
        internal_contradictions=[
            InternalContradiction(
                contradiction_id="contradiction_001",
                claim_a=claim,
                claim_b=correct_claim,
                is_contradiction=True,
                explanation="Conflito entre os trechos.",
                similarity=0.62,
            )
        ],
        llm_model="ollama/qwen2.5:7b-instruct",
        asr_model="faster-whisper/small",
        embedding_model="hashing-512",
        vector_store="numpy",
        limitations=["Apenas as fontes fornecidas foram consideradas."],
    )
    report.statistics = build_statistics(report)
    return report


def test_all_thirteen_sections_are_present() -> None:
    markdown = ReportGenerator(ReportConfig()).render(_report())
    for title in SECTION_TITLES_PT:
        assert f"## {title}" in markdown


def test_english_report_has_english_sections() -> None:
    markdown = ReportGenerator(ReportConfig(language="en")).render(_report())
    assert "## 1. Executive Summary" in markdown
    assert "## 13. Limitations of the Analysis" in markdown


def test_claim_block_is_traceable() -> None:
    markdown = ReportGenerator(ReportConfig()).render(_report())
    assert "### Claim #001" in markdown
    assert "**Timestamp:** 00:04:31" in markdown
    assert "`PARCIALMENTE_CORRETA`" in markdown
    assert "Fonte: `paper.pdf`" in markdown
    assert "Página: 7" in markdown
    assert "Performance becomes limited." in markdown


def test_unverified_evidence_is_never_quoted() -> None:
    report = _report()
    report.critiques[0].evidence[0].verified = False
    markdown = ReportGenerator(ReportConfig()).render(report)
    assert "Nenhuma evidência verificável" in markdown


def test_dropped_critiques_are_excluded() -> None:
    report = _report()
    report.critiques[0].dropped = True
    report.critiques[0].dropped_reason = "não se sustentou"
    report.statistics = build_statistics(report)
    markdown = ReportGenerator(ReportConfig()).render(report)
    assert "### Claim #001" not in markdown
    assert report.statistics.dropped_critiques == 1


def test_statistics_counters() -> None:
    stats = build_statistics(_report())
    assert stats.total_claims == 2
    assert stats.analysed_claims == 2
    assert stats.by_classification["PARCIALMENTE_CORRETA"] == 1
    assert stats.by_classification["CORRETA"] == 1
    assert stats.total_omissions == 1
    assert stats.total_internal_contradictions == 1
    assert stats.evidence_coverage == 1.0


def test_evidence_coverage_ignores_not_supported_verdicts() -> None:
    report = _report()
    report.critiques[0].classification = Classification.NOT_SUPPORTED
    report.critiques[0].evidence = []
    stats = build_statistics(report)
    assert stats.evidence_coverage == 1.0


def test_report_is_written_with_audit_json(tmp_path: Path) -> None:
    path = ReportGenerator(ReportConfig()).write(_report(), tmp_path / "review.md")
    assert path.exists()
    audit = tmp_path / "review_audit.json"
    assert audit.exists()

    import json

    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["critiques"][0]["retrieval_trace"]["results"][0]["chunk"]["page"] == 7


def test_correct_claims_can_be_hidden() -> None:
    markdown = ReportGenerator(ReportConfig(include_correct_claims=False)).render(_report())
    assert "### Claim #002" not in markdown
    assert "### Claim #001" in markdown


def test_long_quotes_are_truncated_explicitly() -> None:
    report = _report()
    report.critiques[0].evidence[0].quote = "palavra " * 200
    report.critiques[0].evidence[0].verified = True
    markdown = ReportGenerator(ReportConfig(max_evidence_quote_chars=100)).render(report)
    assert "[…]" in markdown
