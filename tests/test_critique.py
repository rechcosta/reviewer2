"""Critique engine: evidence binding, anti-hallucination guards, omissions."""

from __future__ import annotations

from pathlib import Path

from app.analysis import CritiqueEngine, verify_quote
from app.analysis.critique import EvidenceSentence, _build_evidence, _build_excerpts
from app.config import Config
from app.llm import LLMProvider
from app.models import (
    Claim,
    Classification,
    DocumentChunk,
    EvidenceStatus,
    Importance,
)
from app.retrieval import EvidenceRetriever


class _FixedProvider(LLMProvider):
    name = "fixed"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.model = "fixed"

    def generate(self, prompt, *, system=None, temperature=None, max_tokens=None, stop=None):
        return self.answer


def _claim(text: str = "Aumentar a frequência do processador sempre aumenta o desempenho.") -> Claim:
    return Claim(
        claim_id="claim_001",
        text=text,
        timestamp="00:00:12",
        start=12.4,
        end=22.0,
        topic="frequência",
        importance=Importance.HIGH,
        transcript_confidence=0.93,
        universal_markers=["sempre"],
    )


def _retriever(config: Config, reference_file: Path) -> EvidenceRetriever:
    retriever = EvidenceRetriever(config)
    retriever.index_references([str(reference_file)])
    return retriever


# --------------------------------------------------------------------------- #
# Quote verification
# --------------------------------------------------------------------------- #
def test_verify_quote_accepts_exact_match() -> None:
    ok, _ = verify_quote("a cache é volátil", "Sabemos que a cache é volátil e rápida.")
    assert ok


def test_verify_quote_ignores_accents_and_spacing() -> None:
    ok, _ = verify_quote("a  cache  e VOLATIL", "Sabemos que a cache é volátil.")
    assert ok


def test_verify_quote_rejects_invented_text() -> None:
    ok, _ = verify_quote("a cache é permanente e nunca perde dados", "A cache é volátil.")
    assert not ok


def test_verify_quote_rejects_tiny_quotes() -> None:
    ok, _ = verify_quote("sim", "sim, é verdade")
    assert not ok


# --------------------------------------------------------------------------- #
# Evidence binding
# --------------------------------------------------------------------------- #
def _sentences() -> dict:
    chunk = DocumentChunk(chunk_id="c1", document="d.md", text="A cache é volátil.", page=3)
    return {"e1": EvidenceSentence("e1", "A cache é volátil.", chunk)}


def test_unknown_sentence_ids_are_discarded() -> None:
    evidence = _build_evidence(
        [
            {"id": "e1", "relation": "SUPPORTS"},
            {"id": "e99", "relation": "SUPPORTS"},          # id que não existe
            {"id": "inventado", "relation": "CONTRADICTS"},
        ],
        _sentences(),
        {"c1": 0.8},
    )
    assert len(evidence) == 1
    assert evidence[0].chunk_id == "c1"
    assert evidence[0].verified


def test_quote_is_taken_from_the_source_not_from_the_model() -> None:
    """The model cannot fabricate a quote: the text comes from the chunk."""
    evidence = _build_evidence([{"id": "e1", "relation": "SUPPORTS"}], _sentences(), {"c1": 0.8})
    assert evidence[0].quote == "A cache é volátil."
    assert evidence[0].page == 3


def test_a_written_quote_is_accepted_only_if_it_matches_a_real_sentence() -> None:
    real = _build_evidence(
        [{"quote": "A cache é volátil.", "relation": "SUPPORTS"}], _sentences(), {"c1": 0.8}
    )
    assert len(real) == 1 and real[0].verified

    invented = _build_evidence(
        [{"quote": "A cache nunca perde dados quando desligada.", "relation": "SUPPORTS"}],
        _sentences(),
        {"c1": 0.8},
    )
    assert invented == []


def test_duplicate_citations_are_collapsed() -> None:
    evidence = _build_evidence(
        [{"id": "e1", "relation": "SUPPORTS"}, {"id": "e1", "relation": "PARTIAL"}],
        _sentences(),
        {"c1": 0.8},
    )
    assert len(evidence) == 1


def test_excerpts_number_every_citable_sentence() -> None:
    from app.models import RetrievedChunk

    chunk = DocumentChunk(
        chunk_id="c1", document="d.md", page=2,
        text="A cache é uma memória volátil. Seu conteúdo é perdido sem energia. Ok.",
    )
    excerpts, sentences = _build_excerpts([RetrievedChunk(chunk=chunk, score=0.7)])
    assert excerpts[0]["chunk_id"] == "c1"
    assert excerpts[0]["page"] == 2
    ids = [s["id"] for s in excerpts[0]["sentences"]]
    assert ids == ["e1", "e2"]          # "Ok." é curta demais para servir de evidência
    assert sentences["e1"].text.startswith("A cache")
    assert sentences["e2"].chunk is chunk


# --------------------------------------------------------------------------- #
# Critique behaviour
# --------------------------------------------------------------------------- #
def test_claim_without_evidence_is_not_supported(config: Config, reference_file: Path, llm) -> None:
    engine = CritiqueEngine(llm, _retriever(config, reference_file), config)
    critique = engine.review_claim(_claim("O campeonato de 1998 foi vencido pela França."))
    assert critique.classification is Classification.NOT_SUPPORTED
    assert critique.evidence_status is EvidenceStatus.INSUFFICIENT
    assert critique.evidence == []
    assert "Não foi possível determinar" in critique.analysis
    assert critique.confidence.value in {"BAIXA", "MEDIA"}


def test_universal_claim_against_conditional_source(config: Config, reference_file: Path, llm) -> None:
    engine = CritiqueEngine(llm, _retriever(config, reference_file), config)
    critique = engine.review_claim(_claim())
    assert critique.classification is Classification.PARTIALLY_CORRECT
    assert critique.has_verified_evidence
    assert critique.evidence[0].document == "referencia.md"
    assert critique.problem


def test_critique_without_verifiable_quote_is_downgraded(config: Config, reference_file: Path) -> None:
    answer = (
        '{"classification": "INCORRETA", "severity": "CRITICO", "confidence": "MUITO_ALTA", '
        '"evidence_status": "OK", "statement_type": "FATO", "compatibility": "NAO", '
        '"evidence": [{"id": "e999", "relation": "CONTRADICTS"}], '
        '"analysis": "análise", "problem": "problema grave", "suggested_correction": "corrigir", '
        '"conclusion": "errado"}'
    )
    engine = CritiqueEngine(_FixedProvider(answer), _retriever(config, reference_file), config)
    critique = engine.review_claim(_claim())
    assert critique.classification is Classification.NOT_SUPPORTED
    assert critique.evidence_status is EvidenceStatus.INSUFFICIENT
    assert critique.problem == ""
    assert "não apresentou citação verificável" in critique.analysis


def test_retrieval_trace_is_attached(config: Config, reference_file: Path, llm) -> None:
    engine = CritiqueEngine(llm, _retriever(config, reference_file), config)
    critique = engine.review_claim(_claim())
    assert critique.retrieval_trace is not None
    assert critique.retrieval_trace.claim_id == "claim_001"
    assert critique.retrieval_trace.results


def test_llm_failure_does_not_abort_the_claim(config: Config, reference_file: Path) -> None:
    engine = CritiqueEngine(_FixedProvider("o modelo caiu"), _retriever(config, reference_file), config)
    critique = engine.review_claim(_claim())
    assert critique.classification is Classification.NOT_SUPPORTED
    assert critique.evidence == []


def test_omissions_require_a_verifiable_quote(config: Config, reference_file: Path, llm) -> None:
    engine = CritiqueEngine(llm, _retriever(config, reference_file), config)
    critique = engine.review_claim(_claim("O prefetching reduz a latência de acesso à memória."))
    for omission in critique.omissions:
        assert any(evidence.verified for evidence in omission.evidence)
        assert omission.claim_id == "claim_001"


def test_confidence_factors_are_recorded(config: Config, reference_file: Path, llm) -> None:
    engine = CritiqueEngine(llm, _retriever(config, reference_file), config)
    critique = engine.review_claim(_claim())
    assert set(critique.confidence_factors) >= {"model", "evidence", "retrieval", "transcription", "final"}
