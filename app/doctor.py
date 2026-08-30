"""Environment check: ``reviewer2 --check``.

Verifies everything a real review needs before you start one, and says
exactly what to do about each problem it finds. Runs in a few seconds and
touches nothing.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import Config

OK = "OK"
WARN = "AVISO"
FAIL = "FALHA"


@dataclass
class Check:
    """One verification and what to do if it failed."""

    name: str
    status: str
    detail: str = ""
    action: str = ""

    @property
    def blocking(self) -> bool:
        return self.status == FAIL


def run_checks(config: Config) -> List[Check]:
    """Run every environment check, in the order the pipeline needs them."""
    return [
        _python(),
        _ffmpeg(),
        _dependency("faster_whisper", "faster-whisper", "transcrição de áudio", blocking=True),
        _dependency("sentence_transformers", "sentence-transformers", "embeddings semânticos"),
        _dependency("faiss", "faiss-cpu", "banco vetorial rápido"),
        _dependency("pymupdf", "pymupdf", "leitura de PDF"),
        _dependency("docx", "python-docx", "leitura de DOCX"),
        _dependency("bs4", "beautifulsoup4", "leitura de HTML e URLs"),
        _llm(config),
        _memory(config),
        _disk(config),
        _directories(config),
    ]


# --------------------------------------------------------------------------- #
def _python() -> Check:
    version = sys.version_info
    if version >= (3, 10):
        return Check("Python", OK, f"{version.major}.{version.minor}.{version.micro}")
    return Check(
        "Python", FAIL, f"{version.major}.{version.minor}",
        "O Reviewer2 exige Python 3.10 ou superior.",
    )


def _ffmpeg() -> Check:
    if shutil.which("ffmpeg"):
        return Check("FFmpeg", OK, shutil.which("ffmpeg") or "")
    return Check(
        "FFmpeg", FAIL, "não encontrado no PATH",
        "Instale: sudo apt install ffmpeg  (Linux) · brew install ffmpeg  (macOS)",
    )


def _dependency(module: str, package: str, purpose: str, *, blocking: bool = False) -> Check:
    try:
        importlib.import_module(module)
        return Check(package, OK, purpose)
    except ImportError:
        return Check(
            package,
            FAIL if blocking else WARN,
            f"não instalado — {purpose}",
            f"pip install {package}",
        )


def _llm(config: Config) -> Check:
    """Check the configured model backend is up and the model is installed."""
    from .llm import OllamaProvider, build_llm

    label = f"LLM ({config.llm.provider}/{config.llm.model})"
    try:
        provider = build_llm(config.llm, check=False)
    except Exception as exc:
        return Check(label, FAIL, str(exc)[:120], "Revise llm.provider em config.yaml.")

    if provider.health_check():
        return Check(label, OK, f"respondendo em {config.llm.base_url}")

    if isinstance(provider, OllamaProvider):
        installed = provider.list_models()
        if installed:
            return Check(
                label, FAIL,
                f"servidor no ar, mas o modelo não está instalado. Disponíveis: {', '.join(installed)}",
                f"ollama pull {config.llm.model}",
            )
        return Check(
            label, FAIL, f"servidor não responde em {config.llm.base_url}",
            "Inicie o servidor com: ollama serve",
        )
    return Check(
        label, FAIL, f"não responde em {config.llm.base_url}",
        "Verifique se o servidor do modelo está no ar e se llm.base_url está correto.",
    )


def _memory(config: Config) -> Check:
    """Warn when the machine is likely too small for the configured model."""
    total = _total_ram_gb()
    if total is None:
        return Check("Memória", WARN, "não foi possível medir a RAM total")

    needed = _estimated_ram_gb(config.llm.model)
    detail = f"{total:.0f} GB no total · o modelo configurado precisa de ~{needed:.0f} GB durante a revisão"
    if total >= needed + 4:
        return Check("Memória", OK, detail)
    if total >= needed:
        return Check(
            "Memória", WARN, detail,
            "Vai rodar, mas com pouca folga. Feche aplicações pesadas durante a revisão.",
        )
    return Check(
        "Memória", WARN, detail,
        f"Use um modelo menor (ollama pull qwen2.5:3b-instruct) ou reduza llm.num_ctx "
        f"(atual: {config.llm.num_ctx}).",
    )


def _disk(config: Config) -> Check:
    try:
        usage = shutil.disk_usage(Path(config.paths.data_dir).resolve().anchor or "/")
        free_gb = usage.free / 1e9
    except OSError as exc:
        return Check("Disco", WARN, str(exc)[:80])
    detail = f"{free_gb:.0f} GB livres"
    if free_gb >= 10:
        return Check("Disco", OK, detail)
    return Check(
        "Disco", WARN, detail,
        "Modelos de Whisper e embeddings são baixados no primeiro uso (~1-2 GB).",
    )


def _directories(config: Config) -> Check:
    try:
        config.paths.ensure()
        probe = Path(config.paths.reports_dir) / ".reviewer2_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return Check("Diretórios de dados", OK, str(Path(config.paths.data_dir).resolve()))
    except OSError as exc:
        return Check(
            "Diretórios de dados", FAIL, str(exc)[:120],
            "Verifique as permissões da pasta data/.",
        )


# --------------------------------------------------------------------------- #
def _total_ram_gb() -> Optional[float]:
    """Total physical RAM in GB, or None when it cannot be determined."""
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:  # macOS and others
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def _estimated_ram_gb(model: str) -> float:
    """Rough RAM needed by a model name, from its parameter count.

    A 4-bit quantised model needs roughly 0.6 GB per billion parameters for
    the weights, plus context and runtime overhead.
    """
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model.lower())
    billions = float(match.group(1)) if match else 7.0
    return billions * 0.75 + 2.0


def render(checks: List[Check]) -> str:
    """Format the check results as an aligned report."""
    symbols = {OK: "✓", WARN: "!", FAIL: "✗"}
    width = max(len(check.name) for check in checks) + 2
    lines = ["", "Reviewer2 — verificação do ambiente", ""]
    for check in checks:
        lines.append(f"  {symbols[check.status]} {check.name.ljust(width)} {check.detail}")
        if check.action:
            lines.append(f"    → {check.action}")

    failures = [check for check in checks if check.blocking]
    warnings = [check for check in checks if check.status == WARN]
    lines.append("")
    if failures:
        lines.append(f"  {len(failures)} problema(s) impedem uma revisão completa.")
    elif warnings:
        lines.append("  Tudo pronto. Os avisos acima são opcionais ou degradam a qualidade.")
    else:
        lines.append("  Tudo pronto para revisar.")
    lines.append("")
    return "\n".join(lines)


def main(config: Config) -> int:
    """Entry point used by ``reviewer2 --check``. Returns the exit code."""
    checks = run_checks(config)
    print(render(checks))
    return 1 if any(check.blocking for check in checks) else 0
