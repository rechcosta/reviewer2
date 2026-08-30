"""Allow ``python -m app`` (and ``python -m app.main``)."""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
