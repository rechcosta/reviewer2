"""Web application factory.

FastAPI is an optional dependency, so it is imported only when the web
interface is actually requested; the CLI works without it.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config
from ..errors import Reviewer2Error
from ..logging_utils import get_logger
from .jobs import JobManager

logger = get_logger(__name__)


def create_app(config: Optional[Config] = None, *, offline: bool = False):
    """Build the FastAPI application serving the Reviewer2 interface."""
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise Reviewer2Error(
            "FastAPI is required by the web interface.",
            module="web",
            stage="startup",
            cause="fastapi / uvicorn are not installed.",
            action=(
                "Install them with: pip install 'reviewer2[api]'  "
                "(or: pip install fastapi uvicorn python-multipart)"
            ),
        ) from exc

    config = config or Config.load()
    config.paths.ensure()
    manager = JobManager(config)

    app = FastAPI(
        title="Reviewer2",
        description="Revisor técnico independente baseado em evidências",
        version="0.1.0",
    )

    from .routes import register_routes

    register_routes(app, config, manager, offline=offline)
    return app
