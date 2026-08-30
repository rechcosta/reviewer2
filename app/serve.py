"""``reviewer2-web`` entry point: serves the web interface with uvicorn."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import Config
from .errors import Reviewer2Error
from .logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Start the Reviewer2 web server."""
    parser = argparse.ArgumentParser(
        prog="reviewer2-web",
        description="Serve the Reviewer2 web interface on localhost.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("--offline", action="store_true", help="run every review in offline mode")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required to serve the web interface.\n"
            "  install with: pip install 'reviewer2[api]'",
            file=sys.stderr,
        )
        return 2

    try:
        config = Config.load(Path(args.config) if args.config else None)
        from .web import create_app

        app = create_app(config, offline=args.offline)
    except Reviewer2Error as exc:
        print(exc.format(), file=sys.stderr)
        return 1

    logger.info("Reviewer2 web interface on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
