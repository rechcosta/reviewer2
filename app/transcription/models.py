"""Transcription data structures and (de)serialisation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..models import Transcript, TranscriptSegment, Word, format_timestamp

__all__ = ["Transcript", "TranscriptSegment", "Word", "format_timestamp", "save_transcript", "load_transcript"]


def save_transcript(transcript: Transcript, path: Path) -> Path:
    """Persist a transcript as JSON (timestamps and confidences preserved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_transcript(path: Path) -> Transcript:
    """Load a transcript from a JSON file produced by :func:`save_transcript`.

    Also accepts a bare list of segments, so transcripts coming from other
    tools can be reused.
    """
    data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        segments = [TranscriptSegment.model_validate(item) for item in data]
        return Transcript(source=str(path), segments=_renumber(segments))
    return Transcript.model_validate(data)


def transcript_to_markdown(transcript: Transcript) -> str:
    """Render a transcript as readable Markdown with timestamps."""
    lines: List[str] = [
        f"# Transcrição — {Path(transcript.source).name}",
        "",
        f"- Modelo ASR: `{transcript.model}`",
        f"- Idioma detectado: `{transcript.language or 'desconhecido'}`",
        f"- Duração: {format_timestamp(transcript.duration)}",
        f"- Confiança média: {transcript.mean_confidence:.2f}",
        "",
        "## Segmentos",
        "",
    ]
    for segment in transcript.segments:
        flag = " ⚠️ baixa confiança" if segment.low_confidence else ""
        lines.append(
            f"**[{format_timestamp(segment.start)} → {format_timestamp(segment.end)}]**"
            f" (conf. {segment.confidence:.2f}{flag})"
        )
        lines.append("")
        lines.append(segment.text.strip())
        lines.append("")
    return "\n".join(lines)


def _renumber(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    for index, segment in enumerate(segments):
        segment.segment_id = index
    return segments


def segments_to_dicts(segments: List[TranscriptSegment]) -> List[Dict[str, Any]]:
    """Compact representation used inside LLM prompts."""
    return [
        {"segment_id": s.segment_id, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
        for s in segments
    ]
