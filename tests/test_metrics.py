"""Evaluation harness (specification section 31)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.errors import Reviewer2Error
from app.evaluate import main as eval_main
from app.metrics import GoldStandard, evaluate, render_markdown
from app.metrics.models import GoldClaim
from app.models import (
    Claim,
    Classification,
    ConfidenceLevel,
    Critique,
    DocumentChunk,
    Evidence,
    EvidenceRelation,
    RetrievalTrace,
    RetrievedChunk,
    ReviewReport,
    Severity,
)


def _critique(
    text: str,
    classification: Classification,
    *,
    confidence: float = 0.7,
    verified: bool = True,
    evidence_text: str = "A memória cache é volátil.",
) -> Critique:
    chunk = DocumentChunk(chunk_id="c1", document="ref.md", text=evidence_text, page=1)
    return Critique(
        claim=Claim(claim_id="claim_x", text=text),
        classification=classification,
        severity=Severity.HIGH if classification.is_problem else Severity.LOW,
        confidence=ConfidenceLevel.from_score(confidence),
        confidence_score=confidence,
        evidence=[
            Evidence(
                chunk_id="c1", document="ref.md", page=1, quote=evidence_text,
                relation=EvidenceRelation.SUPPORTS, verified=verified,
            )
        ],
        retrieval_trace=RetrievalTrace(query=text, results=[RetrievedChunk(chunk=chunk, score=0.8)]),
    )


def _report(critiques) -> ReviewReport:
    return ReviewReport(video="aula.mp4", references=["ref.md"], critiques=critiques)


# --------------------------------------------------------------------------- #
def test_perfect_run_scores_one() -> None:
    text = "A memória cache é uma memória não volátil."
    report = _report([_critique(text, Classification.INCORRECT)])
    gold = GoldStandard(
        video="aula.mp4",
        claims=[GoldClaim(text=text, classification=Classification.INCORRECT)],
    )
    evaluation = evaluate(report, gold)
    assert evaluation.get("critique_precision") == 1.0
    assert evaluation.get("claim_extraction_recall") == 1.0
    assert evaluation.get("false_positive_rate") == 0.0
    assert evaluation.get("evidence_coverage") == 1.0


def test_wrong_classification_lowers_precision() -> None:
    text = "A memória cache é uma memória não volátil."
    report = _report([_critique(text, Classification.CORRECT)])
    gold = GoldStandard(video="v", claims=[GoldClaim(text=text, classification=Classification.INCORRECT)])
    assert evaluate(report, gold).get("critique_precision") == 0.0


def test_flagging_a_correct_claim_is_a_false_positive() -> None:
    text = "Os dados da cache são perdidos quando o computador é desligado."
    report = _report([_critique(text, Classification.INCORRECT)])
    gold = GoldStandard(
        video="v",
        claims=[GoldClaim(text=text, classification=Classification.CORRECT)],
        should_not_flag=[text],
    )
    evaluation = evaluate(report, gold)
    assert evaluation.get("false_positive_rate") == 1.0
    assert evaluation.spurious_critiques == [text]


def test_not_supported_is_not_a_false_positive() -> None:
    """Declaring "undetermined" is an honest result, not a wrong critique."""
    report = _report([_critique("Uma afirmação fora do escopo das fontes.", Classification.NOT_SUPPORTED)])
    gold = GoldStandard(video="v", claims=[])
    assert evaluate(report, gold).get("false_positive_rate") == 0.0


def test_missed_claims_are_listed() -> None:
    report = _report([_critique("Afirmação A sobre pipeline e estágios.", Classification.CORRECT)])
    gold = GoldStandard(
        video="v",
        claims=[
            GoldClaim(text="Afirmação A sobre pipeline e estágios.", classification=Classification.CORRECT),
            GoldClaim(text="Afirmação B sobre prefetching e latência.", classification=Classification.INCORRECT),
        ],
    )
    evaluation = evaluate(report, gold)
    assert evaluation.get("claim_extraction_recall") == 0.5
    assert evaluation.missed_claims == ["Afirmação B sobre prefetching e latência."]


def test_retrieval_hit_rate_checks_the_expected_excerpt() -> None:
    text = "A memória cache é uma memória não volátil."
    hit = _report([_critique(text, Classification.INCORRECT, evidence_text="A memória cache é volátil.")])
    gold = GoldStandard(
        video="v",
        claims=[
            GoldClaim(
                text=text,
                classification=Classification.INCORRECT,
                expected_evidence=["A memória cache é volátil."],
            )
        ],
    )
    assert evaluate(hit, gold).get("retrieval_hit_rate") == 1.0

    miss = _report([_critique(text, Classification.INCORRECT, evidence_text="Um texto completamente diferente.")])
    assert evaluate(miss, gold).get("retrieval_hit_rate") == 0.0


def test_evidence_coverage_counts_verifiable_quotes() -> None:
    text = "A memória cache é uma memória não volátil."
    report = _report([_critique(text, Classification.INCORRECT, verified=False)])
    gold = GoldStandard(video="v", claims=[GoldClaim(text=text, classification=Classification.INCORRECT)])
    assert evaluate(report, gold).get("evidence_coverage") == 0.0


def test_calibration_error_detects_overconfidence() -> None:
    """A reviewer that is confident and wrong must score a large error."""
    gold_claims, critiques = [], []
    for index in range(4):
        text = f"Afirmação número {index} sobre arquitetura de computadores e memória."
        gold_claims.append(GoldClaim(text=text, classification=Classification.CORRECT))
        critiques.append(_critique(text, Classification.INCORRECT, confidence=0.92))
    evaluation = evaluate(_report(critiques), GoldStandard(video="v", claims=gold_claims))
    assert evaluation.get("confidence_calibration_error") > 0.8
    assert evaluation.calibration_bins["MUITO_ALTA"]["accuracy"] == 0.0


def test_calibration_error_is_small_when_honest() -> None:
    gold_claims, critiques = [], []
    for index in range(4):
        text = f"Afirmação número {index} sobre arquitetura de computadores e memória."
        gold_claims.append(GoldClaim(text=text, classification=Classification.CORRECT))
        critiques.append(_critique(text, Classification.CORRECT, confidence=0.92))
    evaluation = evaluate(_report(critiques), GoldStandard(video="v", claims=gold_claims))
    assert evaluation.get("confidence_calibration_error") < 0.1


def test_markdown_rendering_lists_the_primary_metric_first() -> None:
    text = "A memória cache é uma memória não volátil."
    report = _report([_critique(text, Classification.INCORRECT)])
    gold = GoldStandard(video="v", claims=[GoldClaim(text=text, classification=Classification.INCORRECT)])
    markdown = render_markdown(evaluate(report, gold))
    assert markdown.index("critique_precision") < markdown.index("false_positive_rate")
    assert "Confidence calibration" in markdown


def test_gold_standard_example_file_is_valid() -> None:
    gold = GoldStandard.load(Path("examples/gold_standard.json"))
    assert len(gold.claims) >= 5
    assert any(claim.classification is Classification.INCORRECT for claim in gold.claims)
    assert gold.should_not_flag


def test_missing_gold_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(Reviewer2Error) as excinfo:
        GoldStandard.load(tmp_path / "nope.json")
    assert "recommended action" in excinfo.value.format()


def test_eval_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    text = "A memória cache é uma memória não volátil."
    report = _report([_critique(text, Classification.INCORRECT)])
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")

    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps({"video": "v", "claims": [{"text": text, "classification": "INCORRETA"}]}),
        encoding="utf-8",
    )

    assert eval_main(["--report", str(audit), "--gold", str(gold_path)]) == 0
    assert "critique_precision" in capsys.readouterr().out


def test_eval_cli_reports_missing_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    gold_path = tmp_path / "gold.json"
    gold_path.write_text('{"video": "v", "claims": []}', encoding="utf-8")
    assert eval_main(["--report", str(tmp_path / "nope.json"), "--gold", str(gold_path)]) == 1
    assert "recommended action" in capsys.readouterr().out
