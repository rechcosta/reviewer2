"""In-process job queue for the web interface.

A review takes minutes, so the HTTP layer must not block on it. Jobs run in
a background thread pool, one at a time by default (transcription and the
LLM are already CPU/GPU bound), and the client polls for progress.

State is intentionally in memory: Reviewer2 is a local, single-user tool and
the specification asks that nothing be sent anywhere.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..errors import Reviewer2Error
from ..logging_utils import get_logger
from ..models import ReviewReport
from ..pipeline import STAGES as PIPELINE_STAGES, build_pipeline
from .i18n import ui_strings

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Lifecycle of a review job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def stages(language: str) -> List[str]:
    """The stage labels in the pipeline's own order — the progress bar walks this list."""
    strings = ui_strings(language)
    return [strings[f"stage_{name}"] for name in PIPELINE_STAGES]


#: Labels for a job built without a configured language (tests, direct use).
DEFAULT_STAGES: List[str] = stages("pt")


@dataclass
class Job:
    """One review request and everything known about its progress."""

    job_id: str
    video: str
    references: List[str]
    status: JobStatus = JobStatus.QUEUED
    stage: str = ""
    stage_index: int = 0
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    report_path: Optional[str] = None
    error: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)
    stages: List[str] = field(default_factory=lambda: list(DEFAULT_STAGES))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view used by the API."""
        return {
            "job_id": self.job_id,
            "video": Path(self.video).name,
            "references": [Path(reference).name for reference in self.references],
            "status": self.status.value,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "stage_count": len(self.stages),
            "created_at": self.created_at,
            "elapsed": round((self.finished_at or time.time()) - self.created_at, 1),
            "report_path": self.report_path,
            "error": self.error,
            "summary": self.summary,
            "log": self.log[-40:],
        }


class JobManager:
    """Runs review jobs in the background and keeps their state."""

    def __init__(self, config: Config, *, max_workers: int = 1) -> None:
        self.config = config
        self._strings = ui_strings(config.report.language)
        self._stages = stages(config.report.language)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="reviewer2")

    # ------------------------------------------------------------------ #
    def submit(self, video: str, references: List[str], *, offline: bool = False) -> Job:
        """Queue a review and return its job record immediately."""
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            video=str(video),
            references=[str(r) for r in references],
            stages=self._stages,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(self._run, job, offline)
        logger.info("Queued review job %s", job.job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Look up a job by id."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[Job]:
        """Every job, newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    # ------------------------------------------------------------------ #
    def _run(self, job: Job, offline: bool) -> None:
        """Execute the pipeline for one job, recording progress and errors."""
        job.status = JobStatus.RUNNING
        self._advance(job, 0)
        try:
            pipeline = build_pipeline(self.config, offline=offline)
            media = Path(job.video)
            output = Path(self.config.paths.reports_dir) / f"{media.stem}_review.md"
            # A .json upload is a transcript from a previous run: reuse it and
            # skip transcription, exactly like `reviewer2 --transcript`.
            transcript_path = media if media.suffix.lower() == ".json" else None

            # Mirror the pipeline's log lines into the job record so the
            # browser can show the same detail the CLI prints.
            handler = _JobLogHandler(job)
            app_logger = logging.getLogger("app")
            previous_level = app_logger.level
            # The UI shows the same lines the CLI prints, whatever the
            # configured global log level is.
            app_logger.setLevel(min(previous_level or logging.INFO, logging.INFO))
            app_logger.addHandler(handler)
            try:
                report: ReviewReport = pipeline.run(
                    job.video,
                    job.references,
                    transcript_path=transcript_path,
                    output=output,
                    on_stage=lambda index, name: self._advance(job, index),
                )
            finally:
                app_logger.removeHandler(handler)
                app_logger.setLevel(previous_level)

            job.report_path = str(output)
            job.summary = {
                "claims": report.statistics.total_claims,
                "analysed": report.statistics.analysed_claims,
                "problems": len(report.problems),
                "dropped": report.statistics.dropped_critiques,
                "evidence_coverage": report.statistics.evidence_coverage,
                "by_classification": report.statistics.by_classification,
                "llm_model": report.llm_model,
                "asr_model": report.asr_model,
                "embedding_model": report.embedding_model,
            }
            job.status = JobStatus.DONE
            self._advance(job, len(self._stages))
        except Reviewer2Error as exc:
            job.status = JobStatus.FAILED
            job.error = exc.format()
            logger.error("Job %s failed: %s", job.job_id, exc.message)
        except Exception as exc:  # pragma: no cover - unexpected failures
            job.status = JobStatus.FAILED
            job.error = self._strings["err_unexpected"].format(error=exc)
            logger.exception("Job %s crashed", job.job_id)
        finally:
            job.finished_at = time.time()

    def _advance(self, job: Job, index: int) -> None:
        job.stage_index = min(index, len(job.stages))
        job.stage = job.stages[min(index, len(job.stages) - 1)]


class _JobLogHandler(logging.Handler):
    """Logging handler that mirrors the pipeline's log lines into a job."""

    def __init__(self, job: Job) -> None:
        super().__init__(level=logging.INFO)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.job.log.append(record.getMessage())
        except Exception:  # pragma: no cover - never break the pipeline
            pass
