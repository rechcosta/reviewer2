"""HTTP contract tests for the Ollama and OpenAI-compatible providers.

A tiny local HTTP server stands in for the real backends, so the request
payloads, the response parsing and the error handling are exercised for
real — without needing a model or a network connection.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterator

import pytest

from app.errors import LLMError
from app.llm import OllamaProvider, OpenAICompatibleProvider


class _Handler(BaseHTTPRequestHandler):
    """Serves canned Ollama and OpenAI-compatible responses."""

    routes: Dict[str, Any] = {}
    received: Dict[str, Any] = {}

    def log_message(self, *args) -> None:  # silence the test output
        pass

    def _send(self, status: int, payload: Any) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        status, payload = self.routes.get(self.path, (404, {"error": "not found"}))
        self._send(status, payload)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        _Handler.received[self.path] = json.loads(self.rfile.read(length) or b"{}")
        _Handler.received["headers"] = dict(self.headers)
        status, payload = self.routes.get(self.path, (404, {"error": "not found"}))
        self._send(status, payload)


@pytest.fixture
def server() -> Iterator[str]:
    """Start the stub server on a free port and yield its base URL."""
    _Handler.received = {}
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    # poll_interval also bounds how long shutdown() blocks; the default of
    # 0.5s is paid by every test that uses the fixture.
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
def test_ollama_generate(server: str) -> None:
    _Handler.routes = {"/api/generate": (200, {"response": '{"classification": "CORRETA"}'})}
    provider = OllamaProvider(model="qwen2.5:7b-instruct", base_url=server, temperature=0.1)

    assert provider.generate_json("prompt", system="sys")["classification"] == "CORRETA"

    sent = _Handler.received["/api/generate"]
    assert sent["model"] == "qwen2.5:7b-instruct"
    assert sent["stream"] is False
    assert sent["system"] == "sys"
    assert sent["options"]["temperature"] == 0.1
    assert sent["options"]["num_ctx"] > 0


def test_ollama_health_check_matches_installed_model(server: str) -> None:
    _Handler.routes = {"/api/tags": (200, {"models": [{"name": "qwen2.5:7b-instruct"}]})}
    assert OllamaProvider(model="qwen2.5:7b-instruct", base_url=server).health_check() is True
    assert OllamaProvider(model="qwen2.5", base_url=server).health_check() is True  # tag-insensitive
    assert OllamaProvider(model="llama3.1:8b", base_url=server).health_check() is False


def test_ollama_missing_model_reports_pull_command(server: str) -> None:
    _Handler.routes = {"/api/generate": (404, {"error": "model not found"})}
    with pytest.raises(LLMError) as excinfo:
        OllamaProvider(model="missing:7b", base_url=server).generate("prompt")
    assert "ollama pull missing:7b" in excinfo.value.format()


def test_ollama_empty_answer_is_an_error(server: str) -> None:
    _Handler.routes = {"/api/generate": (200, {"response": "   "})}
    with pytest.raises(LLMError):
        OllamaProvider(base_url=server).generate("prompt")


def test_ollama_server_error_is_actionable(server: str) -> None:
    _Handler.routes = {"/api/generate": (500, {"error": "boom"})}
    with pytest.raises(LLMError) as excinfo:
        OllamaProvider(base_url=server).generate("prompt")
    assert "HTTP 500" in excinfo.value.message


def test_ollama_unreachable_server_is_actionable() -> None:
    with pytest.raises(LLMError) as excinfo:
        OllamaProvider(base_url="http://127.0.0.1:1").generate("prompt")
    assert "ollama serve" in excinfo.value.format()


# --------------------------------------------------------------------------- #
# OpenAI-compatible (llama.cpp, LM Studio, vLLM…)
# --------------------------------------------------------------------------- #
def test_openai_compatible_generate(server: str) -> None:
    _Handler.routes = {
        "/v1/chat/completions": (200, {"choices": [{"message": {"content": '{"ok": true}'}}]})
    }
    provider = OpenAICompatibleProvider(model="local-model", base_url=server, api_key="secret")

    assert provider.generate_json("prompt", system="sys")["ok"] is True

    sent = _Handler.received["/v1/chat/completions"]
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["messages"][1]["content"] == "prompt"
    assert sent["stream"] is False
    assert _Handler.received["headers"]["Authorization"] == "Bearer secret"


def test_openai_compatible_without_system_prompt(server: str) -> None:
    _Handler.routes = {"/v1/chat/completions": (200, {"choices": [{"message": {"content": "hi"}}]})}
    OpenAICompatibleProvider(base_url=server).generate("prompt")
    assert _Handler.received["/v1/chat/completions"]["messages"][0]["role"] == "user"


def test_openai_compatible_health_check(server: str) -> None:
    _Handler.routes = {"/v1/models": (200, {"data": [{"id": "local-model"}]})}
    assert OpenAICompatibleProvider(base_url=server).health_check() is True


def test_openai_compatible_unexpected_shape_is_reported(server: str) -> None:
    _Handler.routes = {"/v1/chat/completions": (200, {"unexpected": "shape"})}
    with pytest.raises(LLMError) as excinfo:
        OpenAICompatibleProvider(base_url=server).generate("prompt")
    assert "/v1/chat/completions" in excinfo.value.format()
