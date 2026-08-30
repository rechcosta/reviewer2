"""Ollama provider.

Talks to a local Ollama server (``ollama serve``) through its HTTP API.
Nothing leaves the machine, which keeps the project at zero operational
cost and satisfies the privacy requirement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..errors import LLMError
from ..logging_utils import get_logger
from .interface import LLMProvider

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """LLM backend backed by a local Ollama instance."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        *,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        num_ctx: int = 8192,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "top_p": self.top_p,
                "num_predict": max_tokens or self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if stop:
            payload["options"]["stop"] = stop

        data = self._post("/api/generate", payload)
        response = str(data.get("response", "")).strip()
        if not response:
            raise LLMError(
                "Ollama returned an empty answer.",
                module="llm.ollama",
                stage="generate",
                cause=f"Model '{self.model}' produced no text.",
                action="Check that the model is installed: ollama pull " + self.model,
            )
        return response

    def health_check(self) -> bool:
        """Check the server is up and the configured model is available."""
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            names = {item.get("name", "") for item in response.json().get("models", [])}
        except Exception as exc:
            logger.debug("Ollama not reachable at %s: %s", self.base_url, exc)
            return False
        if not names:
            return False
        base = self.model.split(":")[0]
        return any(name == self.model or name.split(":")[0] == base for name in names)

    def list_models(self) -> List[str]:
        """Return the models installed on the Ollama server."""
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return [item.get("name", "") for item in response.json().get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "The 'requests' package is required by the Ollama provider.",
                module="llm.ollama",
                stage="setup",
                cause="requests is not installed.",
                action="Install it with: pip install requests",
            ) from exc

        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
        except Exception as exc:
            raise LLMError(
                f"Could not reach the Ollama server at {self.base_url}.",
                module="llm.ollama",
                stage="request",
                cause=str(exc),
                action="Start the server with 'ollama serve' or set llm.base_url / REVIEWER2_LLM__BASE_URL.",
            ) from exc

        if response.status_code == 404:
            raise LLMError(
                f"Model '{self.model}' is not available on the Ollama server.",
                module="llm.ollama",
                stage="request",
                cause=response.text[:300],
                action=f"Download it with: ollama pull {self.model}",
            )
        if response.status_code >= 400:
            raise LLMError(
                f"Ollama returned HTTP {response.status_code}.",
                module="llm.ollama",
                stage="request",
                cause=response.text[:300],
                action="Check the Ollama logs and the configured model name.",
            )
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError(
                "Ollama returned a non-JSON body.",
                module="llm.ollama",
                stage="response",
                cause=response.text[:300],
                action="Update Ollama to a recent version.",
            ) from exc
