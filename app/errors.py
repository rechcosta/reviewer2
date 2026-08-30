"""Structured errors for Reviewer2.

Every error carries the module and stage where it happened, a probable
cause and a recommended action, so that failures are actionable instead of
being raw tracebacks (see the observability requirements of the project).
"""

from __future__ import annotations

from typing import Optional


class Reviewer2Error(Exception):
    """Base class for every error raised by Reviewer2."""

    def __init__(
        self,
        message: str,
        *,
        module: str = "reviewer2",
        stage: str = "unknown",
        cause: Optional[str] = None,
        action: Optional[str] = None,
    ) -> None:
        self.message = message
        self.module = module
        self.stage = stage
        self.cause = cause
        self.action = action
        super().__init__(self.format())

    def format(self) -> str:
        """Render a multi-line, human readable description of the failure."""
        lines = [
            f"{self.message}",
            f"  module: {self.module}",
            f"  stage:  {self.stage}",
        ]
        if self.cause:
            lines.append(f"  probable cause: {self.cause}")
        if self.action:
            lines.append(f"  recommended action: {self.action}")
        return "\n".join(lines)


class ConfigurationError(Reviewer2Error):
    """Invalid or missing configuration."""


class MissingDependencyError(Reviewer2Error):
    """An optional third-party dependency is required but not installed."""

    def __init__(self, package: str, *, module: str, stage: str, extra: str = "") -> None:
        super().__init__(
            f"Optional dependency '{package}' is not installed.",
            module=module,
            stage=stage,
            cause=f"'{package}' is needed for this feature but is missing from the environment.",
            action=f"Install it with: pip install {package}" + (f"  ({extra})" if extra else ""),
        )
        self.package = package


class TranscriptionError(Reviewer2Error):
    """Audio extraction or ASR failed."""


class DocumentError(Reviewer2Error):
    """A reference document could not be loaded or parsed."""


class EmbeddingError(Reviewer2Error):
    """Embedding generation failed."""


class RetrievalError(Reviewer2Error):
    """Vector store / retrieval failed."""


class LLMError(Reviewer2Error):
    """The language model backend failed or returned unusable output."""


class ReportError(Reviewer2Error):
    """The final report could not be produced."""
