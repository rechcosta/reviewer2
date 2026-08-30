"""Response cache for any LLM provider.

A review is dozens of model calls and tens of minutes. Repeating one because
a reference was added, or because the run was interrupted, is the most
expensive rework this tool can cause.

Caching at the provider level covers every stage at once — claim extraction,
critique, the devil's advocate and contradiction checks — instead of only the
one stage that happened to be wired for it. The key is the exact request
(model, system prompt, user prompt, temperature), so any change to the
transcript, the references or the prompts produces a different key and the
stale answer can never be served.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from ..logging_utils import get_logger
from .interface import LLMProvider

logger = get_logger(__name__)

#: Bump when prompt semantics change in a way that invalidates old answers.
CACHE_VERSION = "1"


class CachingProvider(LLMProvider):
    """Wraps a provider and serves repeated requests from disk."""

    def __init__(self, inner: LLMProvider, directory: Path, *, enabled: bool = True) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.max_retries = inner.max_retries
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

        if self.enabled:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # a read-only data dir must not break the review
                logger.warning("LLM cache disabled (%s)", exc)
                self.enabled = False

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
        if not self.enabled:
            return self.inner.generate(
                prompt, system=system, temperature=temperature, max_tokens=max_tokens, stop=stop
            )

        key = self._key(prompt, system, temperature, max_tokens)
        cached = self._read(key)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        answer = self.inner.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens, stop=stop
        )
        self._write(key, answer)
        return answer

    def health_check(self) -> bool:
        return self.inner.health_check()

    # ------------------------------------------------------------------ #
    def _key(
        self, prompt: str, system: Optional[str], temperature: Optional[float], max_tokens: Optional[int]
    ) -> str:
        material = json.dumps(
            {
                "version": CACHE_VERSION,
                "provider": self.inner.name,
                "model": self.inner.model,
                "system": system or "",
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def _read(self, key: str) -> Optional[str]:
        path = self.directory / f"{key}.txt"
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug("Ignoring unreadable cache entry %s: %s", path.name, exc)
            return None

    def _write(self, key: str, answer: str) -> None:
        path = self.directory / f"{key}.txt"
        try:
            # Written atomically so an interrupted run leaves no half file.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(answer, encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            logger.debug("Could not write cache entry %s: %s", path.name, exc)

    def summary(self) -> str:
        """Hit/miss summary for the logs."""
        total = self.hits + self.misses
        if not self.enabled or not total:
            return ""
        return f"{self.hits}/{total} model calls served from cache"
