"""LLM layer: one abstract interface, several interchangeable backends."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import LLMConfig
from ..errors import ConfigurationError
from ..logging_utils import get_logger
from .cache import CachingProvider
from .heuristic import HeuristicProvider
from .interface import LLMProvider, as_dict, as_list, parse_json
from .llamacpp import LlamaCppProvider, OpenAICompatibleProvider
from .ollama import OllamaProvider

logger = get_logger(__name__)

__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "LlamaCppProvider",
    "HeuristicProvider",
    "CachingProvider",
    "build_llm",
    "parse_json",
    "as_dict",
    "as_list",
]


def build_llm(
    config: LLMConfig, *, check: bool = True, cache_dir: Optional[Path] = None
) -> LLMProvider:
    """Instantiate the provider described by ``config``.

    When ``check`` is true the backend is probed; if it is unreachable and
    ``fallback_to_heuristic`` is enabled, the offline rule engine is used
    instead (clearly logged, so the report can state the limitation).
    """
    provider = (config.provider or "ollama").strip().lower()

    if provider in {"heuristic", "mock", "offline", "none"}:
        return HeuristicProvider()  # deterministic already; caching adds nothing

    if provider == "ollama":
        llm: LLMProvider = OllamaProvider(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            num_ctx=config.num_ctx,
            timeout=config.timeout,
        )
    elif provider in {"llamacpp", "llama.cpp", "llama-cpp", "openai-compatible", "openai_compatible"}:
        llm = OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
    else:
        raise ConfigurationError(
            f"Unknown LLM provider: '{config.provider}'.",
            module="llm",
            stage="build",
            cause="llm.provider must be one of: ollama, llamacpp, openai-compatible, heuristic.",
            action="Fix llm.provider in config.yaml or REVIEWER2_LLM__PROVIDER.",
        )

    llm.max_retries = max(0, int(config.max_retries))

    if check and not llm.health_check():
        message = (
            f"LLM backend '{llm.name}' with model '{config.model}' is not reachable at {config.base_url}."
        )
        if config.fallback_to_heuristic:
            logger.warning("%s Falling back to the offline heuristic engine.", message)
            return HeuristicProvider()
        raise ConfigurationError(
            message,
            module="llm",
            stage="health_check",
            cause="The server is down, or the model is not installed.",
            action=(
                "Start the backend (e.g. 'ollama serve'), install the model "
                f"('ollama pull {config.model}'), or run with --llm-provider heuristic "
                "for an offline dry run."
            ),
        )

    if config.cache and cache_dir is not None:
        return CachingProvider(llm, Path(cache_dir), enabled=True)
    return llm
