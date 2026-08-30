"""Transcription stage: video/audio in, timed transcript out."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Config
from ..logging_utils import get_logger
from ..models import Transcript
from .audio import extract_audio, ffmpeg_available, probe_duration
from .models import load_transcript, save_transcript, transcript_to_markdown
from .whisper import WhisperTranscriber

logger = get_logger(__name__)

__all__ = [
    "WhisperTranscriber",
    "extract_audio",
    "ffmpeg_available",
    "probe_duration",
    "save_transcript",
    "load_transcript",
    "transcript_to_markdown",
    "transcribe_media",
]


def transcribe_media(
    media_path: Path,
    config: Config,
    *,
    force: bool = False,
    transcript_path: Optional[Path] = None,
) -> Transcript:
    """Transcribe ``media_path``, reusing a cached transcript when allowed."""
    media_path = Path(media_path)
    cache_path = Path(transcript_path) if transcript_path else (
        Path(config.paths.transcripts_dir) / f"{media_path.stem}.json"
    )

    if cache_path.exists() and config.transcription.reuse_existing and not force:
        logger.info("Reusing existing transcript: %s", cache_path)
        return load_transcript(cache_path)

    transcriber = WhisperTranscriber(config.transcription)
    transcript = transcriber.transcribe(media_path, audio_dir=config.paths.cache_dir)
    save_transcript(transcript, cache_path)
    logger.info("Transcript saved to %s", cache_path)
    return transcript
