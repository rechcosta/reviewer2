"""Entry point module, so that ``python -m reviewer2`` also works.

The implementation lives in :mod:`app`; this file only forwards to it.
"""

from app.main import build_parser, main

__all__ = ["main", "build_parser"]

if __name__ == "__main__":
    raise SystemExit(main())
