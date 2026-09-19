"""Command line interface.

    python -m app --video aula.mp4 --reference artigo.pdf
    reviewer2 --video aula.mp4 --reference artigo.pdf --reference notas.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .config import Config
from .errors import Reviewer2Error
from .logging_utils import get_logger, setup_logging
from .pipeline import ReviewPipeline, build_pipeline
from .reports import strings, token

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``reviewer2`` command."""
    parser = argparse.ArgumentParser(
        prog="reviewer2",
        description=(
            "Reviewer2 — an independent, evidence-based technical reviewer for video explanations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  reviewer2 --video data/videos/aula.mp4 --reference data/documents/artigo.pdf\n"
            "  reviewer2 --video aula.mp4 --reference docs/ --llm-model llama3.1:8b\n"
            "  reviewer2 --transcript data/transcripts/aula.json --reference artigo.pdf --offline\n"
        ),
    )
    parser.add_argument("--video", "-v", help="video or audio file to review")
    parser.add_argument(
        "--reference", "-r", action="append", default=[],
        help="reference material: file, directory or URL (repeatable)",
    )
    parser.add_argument("--output", "-o", help="path of the Markdown report to write")
    parser.add_argument("--config", "-c", help="path to config.yaml")
    parser.add_argument(
        "--transcript", help="use this transcript JSON instead of running the ASR stage"
    )

    models = parser.add_argument_group("models")
    models.add_argument("--llm-provider", help="ollama | llamacpp | openai-compatible | heuristic")
    models.add_argument("--llm-model", help="model name, e.g. qwen2.5:7b-instruct")
    models.add_argument("--llm-url", help="base URL of the LLM server")
    models.add_argument("--whisper-model", help="tiny | base | small | medium | large-v3")
    models.add_argument("--embedding-model", help="sentence-transformers model name")
    models.add_argument("--language", help="force the spoken language of the video (e.g. pt, en)")

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--top-k", type=int, help="number of chunks retrieved per claim")
    behaviour.add_argument("--min-score", type=float, help="minimum similarity for evidence")
    behaviour.add_argument("--temperature", type=float, help="LLM temperature")
    behaviour.add_argument("--report-language", choices=["pt", "en"], help="language of the report")
    behaviour.add_argument("--no-verification", action="store_true", help="skip the devil's advocate pass")
    behaviour.add_argument("--no-omissions", action="store_true", help="skip omission detection")
    behaviour.add_argument("--no-contradictions", action="store_true", help="skip internal contradiction checks")
    behaviour.add_argument("--force-transcription", action="store_true", help="ignore any cached transcript")
    behaviour.add_argument("--rebuild-index", action="store_true", help="ignore any cached vector index")
    behaviour.add_argument(
        "--offline", action="store_true",
        help="run without any model: heuristic LLM plus hashing embeddings (dry run)",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING | ERROR")
    misc.add_argument("--verbose", action="store_true", help="verbose log format")
    misc.add_argument("--log-file", help="also write logs to this file")
    misc.add_argument(
        "--check", action="store_true",
        help="verify the environment (ffmpeg, model, dependencies, RAM) and exit",
    )
    misc.add_argument("--print-config", action="store_true", help="print the effective configuration and exit")
    misc.add_argument("--version", action="version", version=f"Reviewer2 {__version__}")
    return parser


def build_config(args: argparse.Namespace) -> Config:
    """Merge CLI arguments over the file/environment configuration."""
    overrides: dict = {
        "llm": {
            "provider": "heuristic" if args.offline else args.llm_provider,
            "model": args.llm_model,
            "base_url": args.llm_url,
            "temperature": args.temperature,
        },
        "transcription": {"model": args.whisper_model, "language": args.language},
        "embeddings": {
            "backend": "hashing" if args.offline else None,
            "model": args.embedding_model,
        },
        "retrieval": {"top_k": args.top_k, "min_score": args.min_score},
        "analysis": {
            "detect_omissions": False if args.no_omissions else None,
            "detect_internal_contradictions": False if args.no_contradictions else None,
            # Asking for an English report also means the analysis prose should
            # be written in English; config.yaml can still set them separately.
            "language": args.report_language,
        },
        "verification": {"enabled": False if args.no_verification else None},
        "report": {"language": args.report_language},
        "log_level": args.log_level,
    }
    return Config.load(Path(args.config) if args.config else None, overrides=overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = build_config(args)
    except Reviewer2Error as exc:
        print(exc.format(), file=sys.stderr)
        return 2

    setup_logging(
        args.log_level or config.log_level,
        verbose=args.verbose,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    if args.check:
        from .doctor import main as doctor_main

        return doctor_main(config)

    if args.print_config:
        print(config.dump_yaml())
        return 0

    if not args.video and not args.transcript:
        parser.error("either --video or --transcript is required")
    if not args.reference:
        parser.error("at least one --reference is required")

    video = args.video or args.transcript
    try:
        pipeline: ReviewPipeline = build_pipeline(config, offline=args.offline)
        report = pipeline.run(
            video,
            args.reference,
            transcript_path=Path(args.transcript) if args.transcript else None,
            force_transcription=args.force_transcription,
            rebuild_index=args.rebuild_index,
            output=Path(args.output) if args.output else None,
        )
    except Reviewer2Error as exc:
        logger.error("Pipeline failed.")
        print(exc.format(), file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted by the user.", file=sys.stderr)
        return 130

    _print_summary(report, config.report.language)
    return 0


def _print_summary(report, language: str = "pt") -> None:
    """Print a short human summary to stdout after a successful run.

    Defects and undetermined claims are counted separately, exactly as the
    report does: "the sources do not cover this" is a limit of the reference
    material, not a mistake by the author, and merging the two sends them
    chasing problems that do not exist.
    """
    s = strings(language)
    stats = report.statistics
    defects = report.defects
    undetermined = report.undetermined

    print()
    print(f"Reviewer2 — {Path(report.video).name}")
    print(f"  verdict          : {report.quality.label(s)}")
    print(f"  claims analysed  : {stats.analysed_claims}/{stats.total_claims}")
    print(f"  errors found     : {len(defects)}")
    if undetermined:
        print(f"  undetermined     : {len(undetermined)} (not covered by the sources)")
    print(f"  evidence coverage: {stats.evidence_coverage:.0%}")
    print(f"  dropped critiques: {stats.dropped_critiques}")
    for critique in defects[:5]:
        text = " ".join(critique.claim.text.split())
        print(
            f"  - [{critique.claim.timestamp}] {token(critique.classification.value, s)}"
            f"/{token(critique.severity.value, s)}: {text[:88]}"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
