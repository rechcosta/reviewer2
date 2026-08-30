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
from typing import Dict, List, Optional

from .config import Config

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"

#: Every user-facing string of the check, in both report languages.
TEXTS: Dict[str, Dict[str, str]] = {
    "pt": {
        "title": "Reviewer2 — verificação do ambiente",
        "python_min": "O Reviewer2 exige Python 3.10 ou superior.",
        "ffmpeg_missing": "não encontrado no PATH",
        "ffmpeg_action": "Instale: sudo apt install ffmpeg  (Linux) · brew install ffmpeg  (macOS)",
        "asr": "transcrição de áudio",
        "embeddings": "embeddings semânticos",
        "faiss": "banco vetorial rápido",
        "pdf": "leitura de PDF",
        "docx": "leitura de DOCX",
        "html": "leitura de HTML e URLs",
        "not_installed": "não instalado",
        "memory": "Memória",
        "disk": "Disco",
        "dirs": "Diretórios de dados",
        "llm_ok": "respondendo em {url}",
        "llm_no_model": "servidor no ar, mas o modelo não está instalado. Disponíveis: {models}",
        "llm_down": "servidor não responde em {url}",
        "llm_start": "Inicie o servidor com: ollama serve",
        "llm_check_url": "Verifique se o servidor do modelo está no ar e se llm.base_url está correto.",
        "llm_provider": "Revise llm.provider em config.yaml.",
        "mem_unknown": "não foi possível medir a RAM total",
        "mem_detail": "{total:.0f} GB no total · o modelo configurado precisa de ~{needed:.0f} GB durante a revisão",
        "mem_tight": "Vai rodar, mas com pouca folga. Feche aplicações pesadas durante a revisão.",
        "mem_small": "Use um modelo menor (ollama pull qwen2.5:3b-instruct) ou reduza llm.num_ctx (atual: {ctx}).",
        "disk_free": "{free:.0f} GB livres",
        "disk_action": "Modelos de Whisper e embeddings são baixados no primeiro uso (~1-2 GB).",
        "dirs_action": "Verifique as permissões da pasta data/.",
        "blocked": "  {n} problema(s) impedem uma revisão completa.",
        "warned": "  Tudo pronto. Os avisos acima são opcionais ou degradam a qualidade.",
        "ready": "  Tudo pronto para revisar.",
    },
    "en": {
        "title": "Reviewer2 — environment check",
        "python_min": "Reviewer2 requires Python 3.10 or newer.",
        "ffmpeg_missing": "not found on PATH",
        "ffmpeg_action": "Install it: sudo apt install ffmpeg  (Linux) · brew install ffmpeg  (macOS)",
        "asr": "audio transcription",
        "embeddings": "semantic embeddings",
        "faiss": "fast vector store",
        "pdf": "PDF reading",
        "docx": "DOCX reading",
        "html": "HTML and URL reading",
        "not_installed": "not installed",
        "memory": "Memory",
        "disk": "Disk",
        "dirs": "Data directories",
        "llm_ok": "responding at {url}",
        "llm_no_model": "server is up, but the model is not installed. Available: {models}",
        "llm_down": "server not responding at {url}",
        "llm_start": "Start the server with: ollama serve",
        "llm_check_url": "Check that the model server is running and that llm.base_url is correct.",
        "llm_provider": "Review llm.provider in config.yaml.",
        "mem_unknown": "could not measure total RAM",
        "mem_detail": "{total:.0f} GB total · the configured model needs ~{needed:.0f} GB during a review",
        "mem_tight": "It will run, but with little headroom. Close heavy applications during the review.",
        "mem_small": "Use a smaller model (ollama pull qwen2.5:3b-instruct) or lower llm.num_ctx (currently {ctx}).",
        "disk_free": "{free:.0f} GB free",
        "disk_action": "Whisper and embedding models are downloaded on first use (~1-2 GB).",
        "dirs_action": "Check the permissions of the data/ folder.",
        "blocked": "  {n} problem(s) prevent a complete review.",
        "warned": "  Ready to run. The warnings above are optional or reduce quality.",
        "ready": "  Ready to review.",
    },
}


def texts(language: str) -> Dict[str, str]:
    """Return the doctor's string table for ``language`` (defaults to Portuguese)."""
    return TEXTS["en"] if str(language).lower().startswith("en") else TEXTS["pt"]


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
    t = texts(config.report.language)
    return [
        _python(t),
        _ffmpeg(t),
        _dependency("faster_whisper", "faster-whisper", t["asr"], blocking=True),
        _dependency("sentence_transformers", "sentence-transformers", t["embeddings"]),
        _dependency("faiss", "faiss-cpu", t["faiss"]),
        _dependency("pymupdf", "pymupdf", t["pdf"]),
        _dependency("docx", "python-docx", t["docx"]),
        _dependency("bs4", "beautifulsoup4", t["html"]),
        _llm(config, t),
        _memory(config, t),
        _disk(config, t),
        _directories(config, t),
    ]


# --------------------------------------------------------------------------- #
def _python(t: Dict[str, str]) -> Check:
    version = sys.version_info
    if version >= (3, 10):
        return Check("Python", OK, f"{version.major}.{version.minor}.{version.micro}")
    return Check("Python", FAIL, f"{version.major}.{version.minor}", t["python_min"])


def _ffmpeg(t: Dict[str, str]) -> Check:
    if shutil.which("ffmpeg"):
        return Check("FFmpeg", OK, shutil.which("ffmpeg") or "")
    return Check("FFmpeg", FAIL, t["ffmpeg_missing"], t["ffmpeg_action"])


def _dependency(module: str, package: str, purpose: str, *, blocking: bool = False) -> Check:
    try:
        importlib.import_module(module)
        return Check(package, OK, purpose)
    except ImportError:
        return Check(
            package,
            FAIL if blocking else WARN,
            f"{purpose} — {package}",
            f"pip install {package}",
        )


def _llm(config: Config, t: Dict[str, str]) -> Check:
    """Check the configured model backend is up and the model is installed."""
    from .llm import OllamaProvider, build_llm

    label = f"LLM ({config.llm.provider}/{config.llm.model})"
    try:
        provider = build_llm(config.llm, check=False)
    except Exception as exc:
        return Check(label, FAIL, str(exc)[:120], t["llm_provider"])

    if provider.health_check():
        return Check(label, OK, t["llm_ok"].format(url=config.llm.base_url))

    inner = getattr(provider, "inner", provider)
    if isinstance(inner, OllamaProvider):
        installed = inner.list_models()
        if installed:
            return Check(
                label, FAIL,
                t["llm_no_model"].format(models=", ".join(installed)),
                f"ollama pull {config.llm.model}",
            )
        return Check(label, FAIL, t["llm_down"].format(url=config.llm.base_url), t["llm_start"])
    return Check(label, FAIL, t["llm_down"].format(url=config.llm.base_url), t["llm_check_url"])


def _memory(config: Config, t: Dict[str, str]) -> Check:
    """Warn when the machine is likely too small for the configured model."""
    total = _total_ram_gb()
    if total is None:
        return Check(t["memory"], WARN, t["mem_unknown"])

    needed = _estimated_ram_gb(config.llm.model)
    detail = t["mem_detail"].format(total=total, needed=needed)
    if total >= needed + 4:
        return Check(t["memory"], OK, detail)
    if total >= needed:
        return Check(t["memory"], WARN, detail, t["mem_tight"])
    return Check(t["memory"], WARN, detail, t["mem_small"].format(ctx=config.llm.num_ctx))


def _disk(config: Config, t: Dict[str, str]) -> Check:
    try:
        usage = shutil.disk_usage(Path(config.paths.data_dir).resolve().anchor or "/")
        free_gb = usage.free / 1e9
    except OSError as exc:
        return Check(t["disk"], WARN, str(exc)[:80])
    detail = t["disk_free"].format(free=free_gb)
    if free_gb >= 10:
        return Check(t["disk"], OK, detail)
    return Check(t["disk"], WARN, detail, t["disk_action"])


def _directories(config: Config, t: Dict[str, str]) -> Check:
    try:
        config.paths.ensure()
        probe = Path(config.paths.reports_dir) / ".reviewer2_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return Check(t["dirs"], OK, str(Path(config.paths.data_dir).resolve()))
    except OSError as exc:
        return Check(t["dirs"], FAIL, str(exc)[:120], t["dirs_action"])


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


def render(checks: List[Check], language: str = "pt") -> str:
    """Format the check results as an aligned report."""
    t = texts(language)
    symbols = {OK: "✓", WARN: "!", FAIL: "✗"}
    width = max(len(check.name) for check in checks) + 2
    lines = ["", t["title"], ""]
    for check in checks:
        lines.append(f"  {symbols[check.status]} {check.name.ljust(width)} {check.detail}")
        if check.action:
            lines.append(f"    → {check.action}")

    failures = [check for check in checks if check.blocking]
    warnings = [check for check in checks if check.status == WARN]
    lines.append("")
    if failures:
        lines.append(t["blocked"].format(n=len(failures)))
    elif warnings:
        lines.append(t["warned"])
    else:
        lines.append(t["ready"])
    lines.append("")
    return "\n".join(lines)


def main(config: Config) -> int:
    """Entry point used by ``reviewer2 --check``. Returns the exit code."""
    checks = run_checks(config)
    print(render(checks, config.report.language))
    return 1 if any(check.blocking for check in checks) else 0
