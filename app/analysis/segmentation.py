"""Semantic segmentation of the transcript.

ASR segments are short and arbitrary. Claims usually span several of them,
so segments are grouped into windows that end at a long pause or when the
window reaches its size limit.
"""

from __future__ import annotations

from typing import List, Optional

from ..config import AnalysisConfig
from ..logging_utils import get_logger
from ..models import Transcript, TranscriptSegment, TranscriptWindow

logger = get_logger(__name__)


def build_windows(transcript: Transcript, config: Optional[AnalysisConfig] = None) -> List[TranscriptWindow]:
    """Group transcript segments into semantically coherent windows."""
    config = config or AnalysisConfig()
    windows: List[TranscriptWindow] = []
    buffer: List[TranscriptSegment] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(segment.text.strip() for segment in buffer).strip()
        if not text:
            buffer.clear()
            return
        confidences = [segment.confidence for segment in buffer]
        windows.append(
            TranscriptWindow(
                window_id=len(windows),
                start=buffer[0].start,
                end=buffer[-1].end,
                text=text,
                segment_ids=[segment.segment_id for segment in buffer],
                mean_confidence=sum(confidences) / len(confidences),
            )
        )
        buffer.clear()

    for segment in transcript.segments:
        if buffer:
            pause = segment.start - buffer[-1].end
            length = sum(len(s.text) for s in buffer)
            duration = buffer[-1].end - buffer[0].start
            ends_sentence = buffer[-1].text.strip().endswith((".", "!", "?"))
            # A pause only ends a window once the window carries enough text to
            # be worth a model call of its own. Speakers pause between ordinary
            # sentences all the time, and splitting on every one of them turns a
            # short lecture into a dozen calls whose prompts are almost entirely
            # instructions — the same claims, several times the CPU time.
            long_enough = length >= config.window_min_chars
            if (
                (pause >= config.window_pause_seconds and ends_sentence and long_enough)
                or length >= config.window_max_chars
                or duration >= config.window_max_seconds
            ):
                flush()
        buffer.append(segment)
    flush()

    logger.debug("Built %d transcript windows from %d segments", len(windows), len(transcript.segments))
    return windows
