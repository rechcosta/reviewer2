"""llama.cpp / OpenAI-compatible provider.

Works with ``llama-server`` (llama.cpp), LM Studio, vLLM, llama-cpp-python's
server mode and any other backend exposing ``/v1/chat/completions``.
It also accepts an API key for self-hosted gateways, but nothing here
requires a paid service.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..errors import LLMError
from ..logging_utils import get_logger
from .interface import LLMProvider

logger = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """LLM backend speaking the OpenAI chat-completions protocol."""

    name = "openai-compatible"

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:8080",
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        data = self._post("/v1/chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "Unexpected response shape from the OpenAI-compatible server.",
                module="llm.llamacpp",
                stage="response",
                cause=str(data)[:300],
                action="Confirm the endpoint implements /v1/chat/completions.",
            ) from exc
        return str(content).strip()

    def health_check(self) -> bool:
        try:
            import requests

            response = requests.get(f"{self.base_url}/v1/models", timeout=10, headers=self._headers())
            return response.status_code < 400
        except Exception as exc:
            logger.debug("OpenAI-compatible endpoint unreachable: %s", exc)
            return False

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "The 'requests' package is required by this provider.",
                module="llm.llamacpp",
                stage="setup",
                cause="requests is not installed.",
                action="Install it with: pip install requests",
            ) from exc
        try:
            response = requests.post(
                f"{self.base_url}{path}", json=payload, timeout=self.timeout, headers=self._headers()
            )
        except Exception as exc:
            raise LLMError(
                f"Could not reach the LLM server at {self.base_url}.",
                module="llm.llamacpp",
                stage="request",
                cause=str(exc),
                action="Start llama-server (or equivalent) and check llm.base_url.",
            ) from exc
        if response.status_code >= 400:
            raise LLMError(
                f"LLM server returned HTTP {response.status_code}.",
                module="llm.llamacpp",
                stage="request",
                cause=response.text[:300],
                action="Check the model name and the server logs.",
            )
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(
                "LLM server returned a non-JSON body.",
                module="llm.llamacpp",
                stage="response",
                cause=response.text[:300],
                action="Confirm the endpoint is OpenAI-compatible.",
            ) from exc


# Backwards-friendly alias: llama.cpp is the most common use of this provider.
LlamaCppProvider = OpenAICompatibleProvider
