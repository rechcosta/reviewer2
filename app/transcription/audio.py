"""Audio extraction with FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..errors import TranscriptionError
from ..logging_utils import get_logger

logger = get_logger(__name__)

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


def ffmpeg_available() -> bool:
    """True when the ``ffmpeg`` binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def probe_duration(path: Path) -> float:
    """Return the media duration in seconds (0.0 when it cannot be probed)."""
    if shutil.which("ffprobe") is None:
        return 0.0
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return float(result.stdout.strip() or 0.0)
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def extract_audio(
    video_path: Path,
    output_path: Optional[Path] = None,
    *,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> Path:
    """Extract a mono PCM WAV track suitable for Whisper.

    Files that are already audio are returned untouched, so no re-encoding
    happens when the user feeds an audio file directly.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise TranscriptionError(
            f"Media file not found: {video_path}",
            module="transcription.audio",
            stage="extract_audio",
            cause="The path given with --video does not exist.",
            action="Check the path, or place the file under data/videos/.",
        )

    if video_path.suffix.lower() == ".wav":
        return video_path

    if output_path is None:
        output_path = video_path.with_suffix(".reviewer2.wav")
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info("Reusing extracted audio: %s", output_path)
        return output_path

    if not ffmpeg_available():
        raise TranscriptionError(
            "FFmpeg is not installed.",
            module="transcription.audio",
            stage="extract_audio",
            cause="The 'ffmpeg' binary was not found on PATH.",
            action="Install FFmpeg (apt install ffmpeg / brew install ffmpeg) and try again.",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le", "-loglevel", "error",
        str(output_path),
    ]
    logger.debug("Running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        raise TranscriptionError(
            f"FFmpeg failed to extract audio from {video_path.name}.",
            module="transcription.audio",
            stage="extract_audio",
            cause=(result.stderr or "unknown ffmpeg error").strip()[:400],
            action="Confirm the file is a valid media file and that it contains an audio track.",
        )
    return output_path
