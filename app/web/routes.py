"""HTTP routes.

This module deliberately does **not** use ``from __future__ import
annotations``: FastAPI resolves the parameter types at import time, and
postponed (string) annotations would prevent it from seeing ``UploadFile``.

It is imported only from :func:`app.web.api.create_app`, after the FastAPI
dependency has been checked, so FastAPI stays optional for CLI users.
"""

import shutil
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from ..config import Config
from ..documents.loader import SUPPORTED_SUFFIXES, is_url
from ..logging_utils import get_logger
from .i18n import ui_strings
from .jobs import JobManager
from .ui import index_html

logger = get_logger(__name__)

#: Media the review accepts. ``.json`` is a transcript produced by a previous
#: run, which lets the web interface skip the ASR stage just like the CLI.
MEDIA_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
    ".json",
}


def register_routes(app: FastAPI, config: Config, manager: JobManager, *, offline: bool = False) -> None:
    """Attach every route to ``app``."""
    strings = ui_strings(config.report.language)

    @app.get("/", response_class=HTMLResponse)
    def index():
        """The single-page interface."""
        return index_html(config.report.language)

    @app.get("/api/health")
    def health():
        """Report which backends are available, so the UI can warn early."""
        return {
            "status": "ok",
            "llm": {
                "provider": config.llm.provider,
                "model": config.llm.model,
                "reachable": _llm_reachable(config),
            },
            "backends": _available_backends(),
        }

    @app.post("/api/reviews", status_code=202)
    def create_review(
        video: UploadFile = File(...),
        references: List[UploadFile] = File(default=[]),
        reference_urls: str = Form(default=""),
        offline_mode: str = Form(default="false"),
    ):
        """Upload a video (or a transcript) plus references and queue the review."""
        video_path = _store_upload(
            video, Path(config.paths.videos_dir), MEDIA_SUFFIXES, strings["label_video"], strings
        )

        stored: List[str] = []
        for upload in references:
            if upload.filename:
                stored.append(
                    str(
                        _store_upload(
                            upload,
                            Path(config.paths.documents_dir),
                            SUPPORTED_SUFFIXES,
                            strings["label_reference"],
                            strings,
                        )
                    )
                )
        for url in (line.strip() for line in reference_urls.splitlines()):
            if not url:
                continue
            if not is_url(url):
                raise HTTPException(status_code=400, detail=strings["err_invalid_url"].format(url=url))
            stored.append(url)

        if not stored:
            raise HTTPException(status_code=400, detail=strings["err_no_reference"])

        job = manager.submit(
            str(video_path), stored, offline=offline or _as_bool(offline_mode)
        )
        return JSONResponse(job.to_dict(), status_code=202)

    @app.get("/api/reviews")
    def list_reviews():
        """Every job submitted since the server started."""
        return {"jobs": [job.to_dict() for job in manager.list_jobs()]}

    @app.get("/api/reviews/{job_id}")
    def get_review(job_id: str):
        """Progress and result of one job."""
        return _job_or_404(manager, job_id, strings).to_dict()

    @app.get("/api/reviews/{job_id}/report", response_class=PlainTextResponse)
    def get_report(job_id: str):
        """The finished Markdown report."""
        job = _job_or_404(manager, job_id, strings)
        if not job.report_path or not Path(job.report_path).exists():
            raise HTTPException(status_code=409, detail=strings["err_report_not_ready"])
        return Path(job.report_path).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
def _job_or_404(manager: JobManager, job_id: str, strings: Dict[str, str]):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=strings["err_job_not_found"])
    return job


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _store_upload(
    upload: UploadFile, directory: Path, allowed: set, label: str, strings: Dict[str, str]
) -> Path:
    """Save an upload under ``directory``, rejecting unsupported extensions."""
    filename = Path(upload.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail=strings["err_unnamed_file"].format(label=label))
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=strings["err_unsupported"].format(
                label=label, suffix=suffix or filename, accepted=", ".join(sorted(allowed))
            ),
        )
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def _llm_reachable(config: Config) -> bool:
    """Whether the configured LLM backend answers right now."""
    try:
        from ..llm import build_llm

        return build_llm(config.llm, check=False).health_check()
    except Exception:
        return False


def _available_backends() -> dict:
    """Which optional backends are installed."""
    import importlib

    labels = {
        "faster_whisper": "asr",
        "sentence_transformers": "embeddings",
        "faiss": "faiss",
        "pymupdf": "pdf",
        "docx": "docx",
        "bs4": "html",
    }
    available = {}
    for module, label in labels.items():
        try:
            importlib.import_module(module)
            available[label] = True
        except ImportError:
            available[label] = False
    return available
