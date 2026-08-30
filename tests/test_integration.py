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
