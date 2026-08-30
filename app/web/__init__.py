"""Web interface (MVP 7): a thin HTTP layer over the same pipeline."""

from .api import create_app
from .jobs import Job, JobManager, JobStatus

__all__ = ["create_app", "JobManager", "Job", "JobStatus", "app"]


def __getattr__(name: str):
    """Lazily build the default app, so ``uvicorn app.web:app`` works.

    Building it eagerly would make FastAPI a hard dependency of the package.
    """
    if name == "app":
        return create_app()
    raise AttributeError(name)
