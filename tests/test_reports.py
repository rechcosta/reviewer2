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
from app.reports import ReportGenerator, build_statistics, strings

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


# --------------------------------------------------------------------------- #
# Language: nothing user-facing may leak Portuguese into an English report
# --------------------------------------------------------------------------- #
def test_english_report_has_no_portuguese_prose() -> None:
    """Regression: the verdict, its advice and the limitations were hardcoded.

    Classification/severity/confidence tokens stay in Portuguese by design —
    they are the canonical vocabulary documented in both READMEs — but every
    sentence written for the reader must follow the report language.
    """
    report = _report()
    report.limitations = [
        strings("en")["lim_sources_only"],
        strings("en")["lim_coverage"].format(ratio=0.25),
    ]
    markdown = ReportGenerator(ReportConfig(language="en")).render(report)

    assert "MINOR_FIXES" in markdown or "READY_TO_PUBLISH" in markdown
    for portuguese in (
        "Sem erros graves", "Pode publicar", "Há erro que compromete",
        "A análise considera apenas", "não puderam ser acompanhadas",
        "Nenhum item identificado", "Erros encontrados",
    ):
        assert portuguese not in markdown, f"vazou português: {portuguese!r}"


def test_verdict_label_and_advice_follow_the_language() -> None:
    from app.models import QualityVerdict

    pt, en = strings("pt"), strings("en")
    verdict = QualityVerdict.NEEDS_RERECORDING
    assert verdict.label(pt) == "REGRAVAR_TRECHO"
    assert verdict.label(en) == "RERECORD_SEGMENT"
    assert "regravar" in verdict.advice(pt).lower()
    assert "re-record" in verdict.advice(en).lower()


def test_defects_exclude_undetermined_claims() -> None:
    """NAO_SUSTENTADA is a limit of the sources, never an error by the author."""
    report = _report()
    report.critiques[1].classification = Classification.NOT_SUPPORTED
    report.critiques[1].evidence = []

    assert len(report.problems) == 2          # both count as "not correct"
    assert len(report.defects) == 1           # but only one is an actual error
    assert len(report.undetermined) == 1
    assert report.defects[0].claim.claim_id == "claim_001"


def test_readmes_document_the_verdict_tokens_the_code_emits() -> None:
    """Regression: the English README listed the Portuguese verdict tokens.

    The verdict names are part of the documented contract, so a rename in
    the code must not leave the READMEs promising something else.
    """
    from pathlib import Path as _Path

    from app.models import QualityVerdict

    english = _Path("README.md").read_text(encoding="utf-8")
    portuguese = _Path("README.pt-BR.md").read_text(encoding="utf-8")

    for verdict in QualityVerdict:
        en_token = verdict.label(strings("en"))
        pt_token = verdict.label(strings("pt"))
        assert f"`{en_token}`" in english, f"README.md não documenta {en_token}"
        assert f"`{pt_token}`" in portuguese, f"README.pt-BR.md não documenta {pt_token}"


def test_english_readme_example_uses_english_field_labels() -> None:
    """The sample report in README.md must match what the generator writes."""
    from pathlib import Path as _Path

    english = _Path("README.md").read_text(encoding="utf-8")
    sample = english.split("```markdown", 1)[1].split("```", 1)[0]

    for label in ("**Claim:**", "**Classification:**", "**Severity:**", "**Confidence:**"):
        assert label in sample, f"faltou {label} no exemplo do README inglês"
    for portuguese_label in ("**Afirmação:**", "**Classificação:**", "**Gravidade:**"):
        assert portuguese_label not in sample, f"rótulo em português no exemplo inglês: {portuguese_label}"
