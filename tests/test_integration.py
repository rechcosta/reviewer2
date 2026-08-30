"""End-to-end integration test.

    transcript → claims → RAG → critique → verification → report

Runs entirely offline with the heuristic provider, hashing embeddings and
the NumPy vector store.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import Config
from app.errors import Reviewer2Error
from app.main import main
from app.models import Classification
from app.pipeline import build_pipeline


def test_full_pipeline(config: Config, transcript_file: Path, reference_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "review.md"
    pipeline = build_pipeline(config, offline=True)
    report = pipeline.run(
        "aula.mp4",
        [str(reference_file)],
        transcript_path=transcript_file,
        output=output,
    )

    # Every stage produced something.
    assert report.transcript is not None and report.transcript.segments
    assert report.claims
    assert report.critiques
    assert report.statistics.total_chunks > 0
    assert report.statistics.analysed_claims == len(report.active_critiques)

    # The report exists and is traceable.
    markdown = output.read_text(encoding="utf-8")
    assert markdown.startswith("# Reviewer2")
    assert "## 5. Análise das Afirmações" in markdown
    assert report.claims[0].timestamp in markdown

    # Audit artefacts.
    assert (tmp_path / "review_audit.json").exists()
    traces = json.loads((tmp_path / "review_retrieval.json").read_text(encoding="utf-8"))
    assert traces and traces[0]["query"]


def test_every_citation_exists_in_the_corpus(
    config: Config, transcript_file: Path, reference_file: Path, tmp_path: Path
) -> None:
    """No critique may cite a document, page or quote that was not retrieved."""
    pipeline = build_pipeline(config, offline=True)
    report = pipeline.run(
        "aula.mp4", [str(reference_file)], transcript_path=transcript_file, output=tmp_path / "r.md"
    )
    # Whitespace is normalised during parsing, so compare against a flattened
    # version of the original file: the words must come from the real source.
    corpus = " ".join(reference_file.read_text(encoding="utf-8").split())
    indexed_ids = {chunk.chunk_id for chunk in pipeline.retriever.store.chunks}

    for critique in report.critiques:
        for evidence in critique.evidence:
            assert evidence.chunk_id in indexed_ids
            assert evidence.document == "referencia.md"
            if evidence.verified:
                assert " ".join(evidence.quote.split()) in corpus


def test_problems_carry_verifiable_evidence(
    config: Config, transcript_file: Path, reference_file: Path, tmp_path: Path
) -> None:
    pipeline = build_pipeline(config, offline=True)
    report = pipeline.run(
        "aula.mp4", [str(reference_file)], transcript_path=transcript_file, output=tmp_path / "r.md"
    )
    for critique in report.problems:
        if critique.classification is Classification.NOT_SUPPORTED:
            assert critique.evidence == []  # explicitly declared as undetermined
        else:
            assert critique.has_verified_evidence


def test_limitations_are_always_reported(
    config: Config, transcript_file: Path, reference_file: Path, tmp_path: Path
) -> None:
    pipeline = build_pipeline(config, offline=True)
    report = pipeline.run(
        "aula.mp4", [str(reference_file)], transcript_path=transcript_file, output=tmp_path / "r.md"
    )
    joined = " ".join(report.limitations)
    assert "modo heurístico offline" in joined
    assert "Embeddings lexicais" in joined


def test_pipeline_requires_references(config: Config, transcript_file: Path) -> None:
    pipeline = build_pipeline(config, offline=True)
    with pytest.raises(Reviewer2Error) as excinfo:
        pipeline.run("aula.mp4", [], transcript_path=transcript_file)
    assert excinfo.value.stage == "index"


def test_cli_offline_run(transcript_file: Path, reference_file: Path, tmp_path: Path) -> None:
    output = tmp_path / "cli_review.md"
    code = main(
        [
            "--transcript", str(transcript_file),
            "--reference", str(reference_file),
            "--offline",
            "--output", str(output),
            "--log-level", "WARNING",
        ]
    )
    assert code == 0
    assert output.exists()
    assert "# Reviewer2" in output.read_text(encoding="utf-8")


def test_cli_reports_missing_reference(transcript_file: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--transcript", str(transcript_file),
            "--reference", str(tmp_path / "inexistente.pdf"),
            "--offline",
            "--log-level", "ERROR",
        ]
    )
    assert code == 1


def test_cli_requires_video_or_transcript(reference_file: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--reference", str(reference_file)])


def test_cli_print_config(capsys: pytest.CaptureFixture) -> None:
    assert main(["--print-config"]) == 0
    assert "provider" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The Ollama path, end to end
# --------------------------------------------------------------------------- #
def test_full_pipeline_over_the_ollama_protocol(
    config: Config, transcript_file: Path, reference_file: Path, tmp_path: Path
) -> None:
    """Run the whole pipeline against a server speaking Ollama's real API.

    This exercises the production path a user gets with `ollama serve` —
    request shape, response parsing and the evidence guards — without
    needing a model to be installed.
    """
    import json as _json
    import re
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from app.pipeline import ReviewPipeline
    from app.llm import OllamaProvider
    from app.retrieval import EvidenceRetriever

    def answer(prompt: str) -> str:
        task = (re.search(r"REVIEWER2_TASK:\s*([a-z_]+)", prompt) or [None, ""])[1]
        payload = _json.loads(prompt[prompt.rindex("INPUT:") + 6 :].strip())
        if task == "claim_extraction":
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", payload.get("text", ""))]
            sentences = [s for s in sentences if len(s) > 20][:2]
            return _json.dumps(
                {"claims": [{"text": s, "topic": "t", "context": "", "importance": "high", "verbatim": s} for s in sentences]}
            )
        if task == "claim_critique":
            excerpt = (payload.get("excerpts") or [{}])[0]
            first = (excerpt.get("sentences") or [{}])[0]
            return _json.dumps(
                {
                    "classification": "PARCIALMENTE_CORRETA", "severity": "MEDIO",
                    "confidence": "ALTA", "evidence_status": "OK", "statement_type": "FATO",
                    "compatibility": "PARCIAL",
                    "evidence": [{"id": first.get("id", ""), "relation": "PARTIAL"}],
                    "analysis": "a fonte condiciona o fenômeno", "problem": "condição omitida",
                    "suggested_correction": "explicitar a condição", "conclusion": "incompleta",
                    "omissions": [],
                }
            )
        if task == "devil_advocate":
            return _json.dumps({"sustained": True, "notes": "mantida", "confidence_adjustment": 0.0})
        if task == "internal_contradiction":
            return _json.dumps({"is_contradiction": False, "explanation": "", "same_sense_of_terms": True})
        return _json.dumps({"omissions": []})

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, payload):
            body = _json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": "qwen2.5:7b-instruct"}]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            request = _json.loads(self.rfile.read(length) or b"{}")
            self._send({"response": answer(request.get("prompt", "")), "done": True})

    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        llm = OllamaProvider(model="qwen2.5:7b-instruct", base_url=f"http://127.0.0.1:{httpd.server_port}")
        assert llm.health_check() is True

        pipeline = ReviewPipeline(config, llm=llm, retriever=EvidenceRetriever(config))
        report = pipeline.run(
            "aula.mp4", [str(reference_file)], transcript_path=transcript_file, output=tmp_path / "r.md"
        )
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert report.llm_model == "ollama/qwen2.5:7b-instruct"
    assert report.claims and report.critiques
    problems = report.problems
    assert problems, "the stub always reports a problem, so some must survive verification"
    # The guards still apply over the Ollama transport: every quote is real.
    corpus = " ".join(reference_file.read_text(encoding="utf-8").split())
    for critique in problems:
        assert critique.has_verified_evidence
        for evidence in critique.evidence:
            if evidence.verified:
                assert " ".join(evidence.quote.split()) in corpus


def test_console_summary_matches_the_report(
    config: Config, transcript_file: Path, reference_file: Path, tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Regression: the console counted NAO_SUSTENTADA as an error, the report did not.

    A number about the user's own work must not differ between the two
    places they read it.
    """
    from app.main import main as cli_main

    output = tmp_path / "r.md"
    assert cli_main([
        "--transcript", str(transcript_file), "--reference", str(reference_file),
        "--offline", "--output", str(output), "--log-level", "ERROR",
    ]) == 0

    console = capsys.readouterr().out
    markdown = output.read_text(encoding="utf-8")

    errors = int(re.search(r"errors found\s+:\s+(\d+)", console).group(1))
    reported = int(re.search(r"Erros encontrados: \*\*(\d+)\*\*", markdown).group(1))
    assert errors == reported

    if "undetermined" in console:
        undetermined = int(re.search(r"undetermined\s+:\s+(\d+)", console).group(1))
        assert f"não verificáveis com estas fontes: **{undetermined}**" in markdown


#: Tokens that stay Portuguese by design: the canonical vocabulary of the
#: specification, documented in both READMEs.
CANONICAL_TOKENS = {
    "CORRETA", "INCORRETA", "PARCIALMENTE_CORRETA", "IMPRECISA", "CONTRADITORIA",
    "NAO_SUSTENTADA", "SIMPLIFICACAO_PEDAGOGICA", "CRITICO", "ALTO", "MEDIO", "BAIXO",
    "PEDAGOGICO", "MUITO_ALTA", "ALTA", "MEDIA", "BAIXA", "FATO", "INFERENCIA",
    "INTERPRETACAO", "OPINIAO", "SIM", "NAO", "PARCIAL", "INDETERMINADO",
}

#: Words that only appear in Portuguese prose written by the code itself.
PORTUGUESE_PROSE = (
    "Não foi possível", "Nenhum item", "Sem evidência", "A crítica", "Crítica mantida",
    "A afirmação", "A análise", "A transcrição", "Ausência de", "Há erro", "Há afirmações",
    "Sem erros graves", "Pode publicar", "não puderam", "verificáveis com estas fontes",
    "A fonte condiciona", "Generalização indevida", "Reformular a afirmação",
)


def test_english_report_leaks_no_portuguese_prose(
    config: Config, transcript_file: Path, reference_file: Path, tmp_path: Path
) -> None:
    """Every sentence the code writes must follow --report-language.

    Regression for a whole class of defect: the verdict, its advice, the
    limitations, the verification notes and the critique engine's own
    sentences were each hardcoded in Portuguese at some point, so an English
    report came out half translated.

    Claim text and source quotes are excluded: those are the user's own
    Portuguese content, not strings written by Reviewer2.
    """
    config.report.language = "en"
    config.analysis.language = "en"
    report = build_pipeline(config, offline=True).run(
        "aula.mp4", [str(reference_file)], transcript_path=transcript_file, output=tmp_path / "en.md"
    )
    assert report.critiques, "the run must produce critiques for this to prove anything"

    markdown = (tmp_path / "en.md").read_text(encoding="utf-8")
    offenders = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith(">") or stripped.startswith("###"):
            continue  # quotes and claim headings carry the video's own language
        if any(token in stripped for token in CANONICAL_TOKENS):
            continue
        for phrase in PORTUGUESE_PROSE:
            if phrase in stripped:
                offenders.append(stripped[:100])
                break
    assert not offenders, "Portuguese prose in an English report:\n  " + "\n  ".join(offenders)


def test_portuguese_report_stays_portuguese(
    config: Config, transcript_file: Path, reference_file: Path, tmp_path: Path
) -> None:
    """The opposite direction: the default must not drift into English."""
    report = build_pipeline(config, offline=True).run(
        "aula.mp4", [str(reference_file)], transcript_path=transcript_file, output=tmp_path / "pt.md"
    )
    markdown = (tmp_path / "pt.md").read_text(encoding="utf-8")
    assert "## 1. Resumo Executivo" in markdown
    assert "## 13. Limitações da Análise" in markdown
    for english in ("Executive Summary", "Ready to publish", "Could not be determined"):
        assert english not in markdown
    assert report.quality.value.isupper()
