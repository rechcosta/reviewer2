"""Logging helpers.

Reviewer2 logs one line per pipeline stage, e.g.::

    [INFO] Running transcription
    [INFO] Extracted 37 claims
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "[%(levelname)s] %(message)s"
_VERBOSE_FORMAT = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", *, verbose: bool = False, log_file: Optional[Path] = None) -> None:
    """Configure the root logger used by the whole application."""
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(_VERBOSE_FORMAT if verbose else _FORMAT))
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_VERBOSE_FORMAT))
        root.addHandler(file_handler)

    # Third-party libraries are noisy at INFO level.
    for noisy in ("httpx", "urllib3", "faiss", "sentence_transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
