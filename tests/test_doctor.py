"""Environment check (`reviewer2 --check`)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from app.config import Config
from app.doctor import FAIL, OK, _estimated_ram_gb, main, render, run_checks


class _OllamaStub(BaseHTTPRequestHandler):
    """Speaks the real Ollama protocol; `installed` controls the model list."""

    installed: list = ["qwen2.5:7b-instruct"]

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        body = json.dumps({"models": [{"name": name} for name in self.installed]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def ollama(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    _OllamaStub.installed = ["qwen2.5:7b-instruct"]
    httpd = HTTPServer(("127.0.0.1", 0), _OllamaStub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _config(tmp_path: Path, **llm) -> Config:
    return Config.load(
        use_env=False,
        overrides={
            "paths": {"data_dir": str(tmp_path), "reports_dir": str(tmp_path / "reports")},
            "llm": llm,
        },
    )


def _named(checks, prefix: str):
    return next(check for check in checks if check.name.startswith(prefix))


def test_reports_a_reachable_model(tmp_path: Path, ollama: str) -> None:
    checks = run_checks(_config(tmp_path, provider="ollama", model="qwen2.5:7b-instruct", base_url=ollama))
    assert _named(checks, "LLM").status == OK


def test_reports_a_missing_model_with_the_pull_command(tmp_path: Path, ollama: str) -> None:
    _OllamaStub.installed = ["llama3.1:8b"]
    checks = run_checks(_config(tmp_path, provider="ollama", model="qwen2.5:7b-instruct", base_url=ollama))
    llm = _named(checks, "LLM")
    assert llm.status == FAIL
    assert "ollama pull qwen2.5:7b-instruct" in llm.action
    assert "llama3.1:8b" in llm.detail  # tells the user what *is* installed


def test_reports_a_server_that_is_down(tmp_path: Path) -> None:
    checks = run_checks(_config(tmp_path, provider="ollama", base_url="http://127.0.0.1:1"))
    llm = _named(checks, "LLM")
    assert llm.status == FAIL
    assert "ollama serve" in llm.action


def test_core_checks_pass_in_this_environment(tmp_path: Path, ollama: str) -> None:
    checks = run_checks(_config(tmp_path, provider="ollama", base_url=ollama))
    for name in ("Python", "FFmpeg", "Diretórios"):
        assert _named(checks, name).status == OK


def test_data_directories_are_created_and_writable(tmp_path: Path, ollama: str) -> None:
    config = _config(tmp_path, provider="ollama", base_url=ollama)
    assert _named(run_checks(config), "Diretórios").status == OK
    assert Path(config.paths.reports_dir).is_dir()


def test_ram_estimate_scales_with_model_size() -> None:
    assert _estimated_ram_gb("qwen2.5:3b-instruct") < _estimated_ram_gb("qwen2.5:7b-instruct")
    assert _estimated_ram_gb("qwen2.5:7b-instruct") < _estimated_ram_gb("qwen2.5:14b-instruct")
    assert 6.0 < _estimated_ram_gb("qwen2.5:7b-instruct") < 9.0
    assert _estimated_ram_gb("modelo-sem-tamanho") > 0  # falls back to a 7B assumption


def test_render_lists_every_check_and_its_action(tmp_path: Path) -> None:
    text = render(run_checks(_config(tmp_path, provider="ollama", base_url="http://127.0.0.1:1")))
    assert "verificação do ambiente" in text
    assert "ollama serve" in text
    assert "problema(s) impedem" in text


def test_exit_code_signals_a_blocking_problem(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(_config(tmp_path, provider="ollama", base_url="http://127.0.0.1:1")) == 1
    assert "✗" in capsys.readouterr().out


def test_exit_code_is_zero_when_ready(tmp_path: Path, ollama: str, capsys: pytest.CaptureFixture) -> None:
    assert main(_config(tmp_path, provider="ollama", base_url=ollama)) == 0
    assert "pronto" in capsys.readouterr().out


def test_cli_check_flag(tmp_path: Path, ollama: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import main as cli_main

    monkeypatch.setenv("REVIEWER2_LLM__BASE_URL", ollama)
    assert cli_main(["--check"]) == 0
