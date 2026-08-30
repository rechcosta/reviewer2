"""``reviewer2-eval`` entry point: scores a review against a gold standard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .errors import Reviewer2Error
from .logging_utils import setup_logging
from .metrics import GoldStandard, evaluate, render_markdown
from .models import ReviewReport


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Compare an audit JSON with a human annotation and print the metrics."""
    parser = argparse.ArgumentParser(
        prog="reviewer2-eval",
        description="Measure critique precision, false positives, evidence coverage and calibration.",
        epilog=(
            "Example:\n"
            "  reviewer2-eval --report data/reports/aula_review_audit.json "
            "--gold examples/gold_standard.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--report", required=True, help="the *_audit.json produced by a review")
    parser.add_argument("--gold", required=True, help="the human annotation (JSON)")
    parser.add_argument("--output", help="write the evaluation as Markdown to this path")
    parser.add_argument("--json", action="store_true", help="print the raw metrics as JSON")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    try:
        report = _load_report(Path(args.report))
        gold = GoldStandard.load(Path(args.gold))
    except Reviewer2Error as exc:
        print(exc.format())
        return 1

    evaluation = evaluate(report, gold)

    if args.json:
        print(json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print(render_markdown(evaluation))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(evaluation), encoding="utf-8")
        print(f"Evaluation written to {path}")
    return 0


def _load_report(path: Path) -> ReviewReport:
    """Load the audit JSON written next to a Markdown report."""
    if not path.exists():
        raise Reviewer2Error(
            f"Report file not found: {path}",
            module="metrics",
            stage="load",
            cause="Point --report at the *_audit.json produced by a review.",
            action="Run a review first; the audit JSON is written next to the Markdown report.",
        )
    try:
        return ReviewReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise Reviewer2Error(
            f"Could not read the report: {path}",
            module="metrics",
            stage="load",
            cause=str(exc)[:300],
            action="Use the *_audit.json file, not the Markdown report.",
        ) from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
