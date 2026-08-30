"""ASR stage based on faster-whisper.

Timestamps and per-segment confidence are preserved; low-confidence
segments are flagged so that downstream critiques can be softened when the
transcription itself is unreliable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

from ..config import TranscriptionConfig
from ..errors import MissingDependencyError, TranscriptionError
from ..logging_utils import get_logger
from ..models import Transcript, TranscriptSegment, Word
from .audio import extract_audio, probe_duration

logger = get_logger(__name__)


class WhisperTranscriber:
    """Thin wrapper around ``faster_whisper.WhisperModel``."""

    def __init__(self, config: Optional[TranscriptionConfig] = None) -> None:
        self.config = config or TranscriptionConfig()
        self._model = None

    # ------------------------------------------------------------------ #
    @property
    def model_name(self) -> str:
        return f"faster-whisper/{self.config.model}"

    def _load_model(self):
        """Instantiate the Whisper model lazily (loading it is expensive)."""
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependencyError(
                "faster-whisper",
                module="transcription.whisper",
                stage="load_model",
                extra="required for automatic speech recognition",
            ) from exc

        device = self.config.device
        if device == "auto":
            device = _detect_device()
        compute_type = self.config.compute_type
        if device == "cpu" and compute_type in {"float16", "fp16"}:
            compute_type = "int8"

        logger.info("Loading Whisper model '%s' (device=%s, compute_type=%s)", self.config.model, device, compute_type)
        try:
            self._model = WhisperModel(self.config.model, device=device, compute_type=compute_type)
        except Exception as exc:
            raise TranscriptionError(
                f"Could not load the Whisper model '{self.config.model}'.",
                module="transcription.whisper",
                stage="load_model",
                cause=str(exc)[:300],
                action=(
                    "Check the model name (tiny/base/small/medium/large-v3), the available disk space "
                    "and the network connection for the first download."
                ),
            ) from exc
        return self._model

    # ------------------------------------------------------------------ #
    def transcribe(self, media_path: Path, *, audio_dir: Optional[Path] = None) -> Transcript:
        """Transcribe a video or audio file into a :class:`Transcript`."""
        media_path = Path(media_path)
        audio_path = extract_audio(
            media_path,
            output_path=(Path(audio_dir) / f"{media_path.stem}.wav") if audio_dir else None,
            sample_rate=self.config.audio_sample_rate,
        )

        model = self._load_model()
        logger.info("Running transcription")
        try:
            raw_segments, info = model.transcribe(
                str(audio_path),
                language=self.config.language,
                beam_size=self.config.beam_size,
                vad_filter=self.config.vad_filter,
                word_timestamps=self.config.word_timestamps,
            )
        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed for {media_path.name}.",
                module="transcription.whisper",
                stage="transcribe",
                cause=str(exc)[:300],
                action="Try a smaller model (--whisper-model small) or check the extracted audio file.",
            ) from exc

        segments: List[TranscriptSegment] = []
        for index, raw in enumerate(raw_segments):
            confidence = _segment_confidence(raw)
            words = [
                Word(
                    word=str(getattr(word, "word", "")).strip(),
                    start=float(getattr(word, "start", 0.0) or 0.0),
                    end=float(getattr(word, "end", 0.0) or 0.0),
                    probability=float(getattr(word, "probability", 1.0) or 1.0),
                )
                for word in (getattr(raw, "words", None) or [])
            ]
            text = str(getattr(raw, "text", "")).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    segment_id=index,
                    start=float(getattr(raw, "start", 0.0) or 0.0),
                    end=float(getattr(raw, "end", 0.0) or 0.0),
                    text=text,
                    confidence=confidence,
                    low_confidence=confidence < self.config.low_confidence_threshold,
                    words=words,
                )
            )

        duration = float(getattr(info, "duration", 0.0) or 0.0) or probe_duration(media_path)
        transcript = Transcript(
            source=str(media_path),
            language=getattr(info, "language", None),
            duration=duration,
            model=self.model_name,
            segments=segments,
        )
        low = sum(1 for s in segments if s.low_confidence)
        logger.info(
            "Transcribed %d segments (%.0fs, language=%s, %d low-confidence)",
            len(segments), duration, transcript.language, low,
        )
        if not segments:
            raise TranscriptionError(
                "The transcription produced no segments.",
                module="transcription.whisper",
                stage="transcribe",
                cause="The audio track may be silent, too short, or in an unsupported format.",
                action="Check the media file and try forcing the language with --language.",
            )
        return transcript


def _segment_confidence(raw) -> float:
    """Convert Whisper's ``avg_logprob`` into a rough 0-1 confidence.

    ``exp(avg_logprob)`` is the geometric mean token probability. Segments
    flagged as likely non-speech (high ``no_speech_prob``) are penalised.
    """
    avg_logprob = getattr(raw, "avg_logprob", None)
    if avg_logprob is None:
        words = getattr(raw, "words", None) or []
        if words:
            probs = [float(getattr(w, "probability", 1.0) or 1.0) for w in words]
            return max(0.0, min(1.0, sum(probs) / len(probs)))
        return 1.0
    confidence = math.exp(max(-10.0, float(avg_logprob)))
    no_speech = float(getattr(raw, "no_speech_prob", 0.0) or 0.0)
    confidence *= max(0.0, 1.0 - no_speech)
    return max(0.0, min(1.0, confidence))


def _detect_device() -> str:
    """Prefer CUDA when torch reports a usable GPU, otherwise CPU."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
