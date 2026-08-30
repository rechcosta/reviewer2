"""Abstract LLM interface.

The rest of Reviewer2 only ever talks to :class:`LLMProvider`, so the
concrete model (Ollama, llama.cpp, any OpenAI-compatible server) can be
swapped through configuration without touching the analysis code.
"""

from __future__ import annotations

import abc
import json
import re
from typing import Any, Dict, List, Optional

from ..errors import LLMError
from ..logging_utils import get_logger

logger = get_logger(__name__)

_JSON_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class LLMProvider(abc.ABC):
    """Minimal contract every language-model backend must implement."""

    name: str = "llm"
    model: str = "unknown"
    #: How many JSON repair attempts are allowed before giving up.
    max_retries: int = 1

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Return the model completion for ``prompt``."""

    def health_check(self) -> bool:
        """Return True when the backend is reachable and the model is usable."""
        try:
            self.generate("ok", max_tokens=5)
            return True
        except Exception as exc:  # pragma: no cover - depends on environment
            logger.debug("Health check failed for %s: %s", self.name, exc)
            return False

    # ------------------------------------------------------------------ #
    # JSON helpers
    # ------------------------------------------------------------------ #
    def generate_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        expect: str = "object",
        retries: Optional[int] = None,
    ) -> Any:
        """Generate a completion and parse it as JSON.

        The model is asked once; if the answer is not valid JSON, a repair
        prompt is issued (up to ``retries`` times). Raises :class:`LLMError`
        when no valid JSON can be obtained — callers treat that as
        ``INSUFFICIENT_EVIDENCE`` rather than inventing content.
        """
        retries = self.max_retries if retries is None else retries
        raw = self.generate(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        for attempt in range(retries + 1):
            parsed = parse_json(raw, expect=expect)
            if parsed is not None:
                return parsed
            if attempt >= retries:
                break
            logger.debug("Invalid JSON from %s, requesting repair (attempt %d)", self.name, attempt + 1)
            raw = self.generate(
                _repair_prompt(raw, expect),
                system="You output JSON only. No prose, no markdown fences.",
                temperature=0.0,
                max_tokens=max_tokens,
            )
        raise LLMError(
            "The model did not return valid JSON.",
            module="llm",
            stage="generate_json",
            cause=f"Last answer (truncated): {raw[:400]!r}",
            action="Use a stronger instruct model, lower the temperature, or reduce the input size.",
        )


def _repair_prompt(broken: str, expect: str) -> str:
    return (
        f"The text below should be a single valid JSON {expect} but it is malformed.\n"
        "Rewrite it as valid JSON. Do not add, remove or invent any information.\n"
        "Answer with JSON only.\n\n"
        f"{broken[:6000]}"
    )


def parse_json(raw: str, *, expect: str = "object") -> Optional[Any]:
    """Extract the first valid JSON value of the requested kind from an answer.

    Handles code fences, leading prose and trailing text, and repairs answers
    truncated by the token limit (a very common failure with small models).

    The requested ``expect`` kind is enforced: asking for an object never
    returns an inner array. Without that guard a truncated critique object
    would silently degrade into its own ``evidence`` array, and the caller
    would read an evidence entry as if it were the verdict.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    fenced = _JSON_FENCE.search(text)
    candidates: List[str] = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text)

    opener = "{" if expect == "object" else "["

    for candidate in candidates:
        # 0. A bare array where an object was expected: some models answer with
        #    a plain list. Checked first, and only when the answer genuinely
        #    starts with one, so the whole list survives instead of collapsing
        #    into its first element.
        if expect == "object" and candidate.lstrip().startswith("["):
            value = _try_load(_extract_balanced(candidate, "["), "array")
            if value is not None:
                return value

        # 1. A complete, balanced block of the requested kind.
        value = _try_load(_extract_balanced(candidate, opener), expect)
        if value is not None:
            return value

        # 2. The same block, truncated by the token limit.
        repaired = _close_truncated(candidate, opener)
        if repaired is not None:
            value = _try_load(repaired, expect)
            if value is not None:
                logger.warning(
                    "The model's JSON answer was truncated and had to be repaired; "
                    "consider raising llm.max_tokens."
                )
                return value
    return None


def _try_load(text: Optional[str], expect: str) -> Optional[Any]:
    """Parse ``text`` as JSON and check it is of the expected kind."""
    if not text:
        return None
    wanted = dict if expect == "object" else list
    for attempt in (text, _relax(text)):
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(value, wanted):
            return value
    return None


def _close_truncated(text: str, opener: str) -> Optional[str]:
    """Close a JSON block that was cut off mid-answer.

    Terminates an unterminated string, drops a dangling ``"key":`` with no
    value, and appends the missing closing brackets. Returns ``None`` when
    the block is not actually truncated.
    """
    start = text.find(opener)
    if start < 0:
        return None

    stack: List[str] = []
    in_string = False
    escaped = False
    for char in text[start:]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()

    if not stack:
        return None  # nothing is open: the block was not truncated

    body = text[start:]
    if in_string:
        body += '"'
    # A key whose value never arrived, and any trailing comma.
    body = re.sub(r',\s*"[^"]*"\s*:\s*$', "", body)
    body = re.sub(r'(\{)\s*"[^"]*"\s*:\s*$', r"\1", body)
    body = re.sub(r",\s*$", "", body)
    return body + "".join(reversed(stack))


def _extract_balanced(text: str, opener: str) -> Optional[str]:
    """Return the first balanced ``{...}``/``[...]`` block, ignoring strings."""
    closer = "}" if opener == "{" else "]"
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _relax(text: str) -> str:
    """Fix the two most common LLM JSON defects: trailing commas and NaN."""
    relaxed = re.sub(r",\s*([}\]])", r"\1", text)
    relaxed = re.sub(r"\bNaN\b|\bInfinity\b", "null", relaxed)
    relaxed = relaxed.replace("“", '"').replace("”", '"')
    return relaxed


def as_dict(value: Any) -> Dict[str, Any]:
    """Coerce parsed JSON into a dictionary (defensive helper)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def as_list(value: Any, key: Optional[str] = None) -> List[Any]:
    """Coerce parsed JSON into a list, optionally unwrapping ``{key: [...]}``."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if key and isinstance(value.get(key), list):
            return value[key]
        for candidate in value.values():
            if isinstance(candidate, list):
                return candidate
    return []
