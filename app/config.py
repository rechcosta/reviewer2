"""Configuration handling.

Configuration lives outside the code, in ``config.yaml`` and ``.env``.
Precedence (highest first): CLI overrides > environment variables >
``config.yaml`` > built-in defaults.

Environment variables use the ``REVIEWER2_`` prefix and ``__`` as the nesting
separator, e.g. ``REVIEWER2_LLM__MODEL=llama3.1:8b``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .errors import ConfigurationError

ENV_PREFIX = "REVIEWER2_"
DEFAULT_CONFIG_FILES = ("config.yaml", "config.yml")


class PathsConfig(BaseModel):
    """Where Reviewer2 reads and writes data."""

    data_dir: Path = Path("data")
    videos_dir: Path = Path("data/videos")
    documents_dir: Path = Path("data/documents")
    transcripts_dir: Path = Path("data/transcripts")
    reports_dir: Path = Path("data/reports")
    index_dir: Path = Path("data/index")
    cache_dir: Path = Path("data/cache")

    def ensure(self) -> None:
        """Create every configured directory."""
        for value in self.model_dump().values():
            Path(value).mkdir(parents=True, exist_ok=True)


class TranscriptionConfig(BaseModel):
    """faster-whisper / ASR settings."""

    backend: str = "faster-whisper"
    model: str = "small"
    device: str = "auto"
    compute_type: str = "int8"
    language: Optional[str] = None
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    low_confidence_threshold: float = 0.6
    audio_sample_rate: int = 16000
    reuse_existing: bool = True


class DocumentsConfig(BaseModel):
    """Parsing and chunking of reference material."""

    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_size: int = 120
    pdf_backend: str = "auto"  # auto | pymupdf | pypdf
    html_strip_tags: List[str] = Field(default_factory=lambda: ["script", "style", "nav", "footer", "header"])
    request_timeout: int = 30
    user_agent: str = "Reviewer2/0.1 (local technical reviewer)"
    download_dir: Path = Path("data/cache")


class EmbeddingsConfig(BaseModel):
    """Embedding backend settings."""

    backend: str = "auto"  # auto | sentence-transformers | hashing
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"
    batch_size: int = 32
    hashing_dimension: int = 512


class RetrievalConfig(BaseModel):
    """Vector store and retrieval settings."""

    backend: str = "auto"  # auto | faiss | numpy
    top_k: int = 6
    min_score: float = 0.25
    max_evidence_per_claim: int = 4
    lexical_boost: float = 0.15
    persist: bool = True


class LLMConfig(BaseModel):
    """Language model provider settings."""

    provider: str = "ollama"  # ollama | llamacpp | openai-compatible | heuristic
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 3072
    timeout: int = 600
    num_ctx: int = 8192
    max_retries: int = 2
    fallback_to_heuristic: bool = False
    concurrency: int = 2      # simultaneous model calls; batching raises CPU throughput
    cache: bool = True        # reuse identical model calls across runs


class AnalysisConfig(BaseModel):
    """Claim extraction and critique settings."""

    window_max_seconds: float = 45.0
    window_max_chars: int = 1200
    window_pause_seconds: float = 1.4
    max_claims_per_window: int = 6
    min_claim_chars: int = 15
    detect_omissions: bool = True
    detect_internal_contradictions: bool = True
    contradiction_similarity_threshold: float = 0.45
    contradiction_similarity_threshold_lexical: float = 0.28
    max_contradiction_pairs: int = 10
    language: str = "pt"  # language used for prompts and report text


class VerificationConfig(BaseModel):
    """Devil's advocate settings."""

    enabled: bool = True
    drop_unsustained: bool = True
    max_confidence_drop: float = 0.35
    skip_correct_claims: bool = True


class ReportConfig(BaseModel):
    """Report rendering settings."""

    language: str = "pt"  # pt | en
    include_correct_claims: bool = True
    include_audit_json: bool = True
    max_evidence_quote_chars: int = 700
    filename_suffix: str = "_review"


class Config(BaseModel):
    """Root configuration object."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        *,
        overrides: Optional[Dict[str, Any]] = None,
        use_env: bool = True,
    ) -> "Config":
        """Build a configuration from file, environment and explicit overrides."""
        data: Dict[str, Any] = {}

        config_path = _resolve_config_path(path)
        if config_path is not None:
            data = _read_yaml(config_path)

        if use_env:
            load_dotenv()
            data = deep_merge(data, _config_from_env())

        if overrides:
            data = deep_merge(data, _prune_none(overrides))

        try:
            return cls.model_validate(data)
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigurationError(
                "Invalid configuration.",
                module="config",
                stage="validation",
                cause=str(exc),
                action="Check config.yaml and the REVIEWER2_* environment variables.",
            ) from exc

    def dump_yaml(self) -> str:
        """Serialise the configuration back to YAML (for debugging)."""
        try:
            import yaml
        except ImportError:  # pragma: no cover
            import json

            return json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_config_path(path: Optional[Path]) -> Optional[Path]:
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise ConfigurationError(
                f"Configuration file not found: {candidate}",
                module="config",
                stage="load",
                cause="The path given with --config does not exist.",
                action="Point --config to an existing YAML file, or omit it to use defaults.",
            )
        return candidate
    env_path = os.environ.get(f"{ENV_PREFIX}CONFIG")
    if env_path:
        return Path(env_path)
    for name in DEFAULT_CONFIG_FILES:
        candidate = Path(name)
        if candidate.exists():
            return candidate
    return None


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigurationError(
            "PyYAML is required to read configuration files.",
            module="config",
            stage="load",
            cause="PyYAML is not installed.",
            action="Install it with: pip install pyyaml",
        ) from exc
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ConfigurationError(
            f"Could not parse configuration file: {path}",
            module="config",
            stage="load",
            cause=str(exc),
            action="Fix the YAML syntax of the file.",
        ) from exc
    if not isinstance(content, dict):
        raise ConfigurationError(
            f"Configuration file must contain a mapping: {path}",
            module="config",
            stage="load",
            cause=f"Top level object is {type(content).__name__}.",
            action="Rewrite the file as key/value pairs.",
        )
    return content


def load_dotenv(path: Optional[Path] = None) -> None:
    """Minimal ``.env`` loader (no external dependency).

    Existing environment variables always win over the file.
    """
    env_file = Path(path or os.environ.get(f"{ENV_PREFIX}ENV_FILE", ".env"))
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _coerce(value: str) -> Any:
    """Convert an environment string into bool/int/float when appropriate."""
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _config_from_env() -> Dict[str, Any]:
    """Translate ``REVIEWER2_SECTION__FIELD`` variables into nested dicts."""
    result: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX) or key in (f"{ENV_PREFIX}CONFIG", f"{ENV_PREFIX}ENV_FILE"):
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(value)
    return result


def deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``extra`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _prune_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``None`` values so that CLI defaults do not clobber the config."""
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested = _prune_none(value)
            if nested:
                cleaned[key] = nested
        elif value is not None:
            cleaned[key] = value
    return cleaned
