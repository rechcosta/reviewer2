"""Transcription models, persistence and audio extraction."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.errors import TranscriptionError
from app.models import Transcript, TranscriptSegment, format_timestamp
from app.transcription import extract_audio, load_transcript, save_transcript, transcript_to_markdown
from app.transcription.whisper import _segment_confidence


class _FakeSegment:
    """Stand-in for a faster-whisper segment."""

    def __init__(self, avg_logprob=None, no_speech_prob=0.0, words=None):
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.words = words


class _FakeWord:
    def __init__(self, probability):
        self.probability = probability


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(271) == "00:04:31"
    assert format_timestamp(3671.9) == "01:01:11"
    assert format_timestamp(-5) == "00:00:00"


def test_transcript_round_trip(tmp_path: Path, transcript: Transcript) -> None:
    path = save_transcript(transcript, tmp_path / "t.json")
    restored = load_transcript(path)
    assert len(restored.segments) == len(transcript.segments)
    assert restored.segments[1].text == transcript.segments[1].text
    assert restored.segments[1].start == pytest.approx(transcript.segments[1].start)
    assert restored.language == "pt"


def test_load_transcript_accepts_bare_segment_list(tmp_path: Path) -> None:
    path = tmp_path / "segments.json"
    path.write_text('[{"start": 0.0, "end": 2.0, "text": "olá"}]', encoding="utf-8")
    restored = load_transcript(path)
    assert restored.segments[0].text == "olá"
    assert restored.segments[0].segment_id == 0


def test_transcript_statistics(transcript: Transcript) -> None:
    assert transcript.mean_confidence == pytest.approx(0.928, abs=0.01)
    assert transcript.low_confidence_ratio == 0.0
    assert "desempenho" in transcript.text


def test_low_confidence_ratio() -> None:
    segments = [
        TranscriptSegment(segment_id=0, start=0, end=1, text="a", confidence=0.9),
        TranscriptSegment(segment_id=1, start=1, end=2, text="b", confidence=0.3, low_confidence=True),
    ]
    assert Transcript(source="x", segments=segments).low_confidence_ratio == 0.5


def test_window_selects_overlapping_segments(transcript: Transcript) -> None:
    window = transcript.window(20.0, 30.0)
    assert [segment.segment_id for segment in window] == [1, 2]


def test_segment_confidence_from_logprob() -> None:
    assert _segment_confidence(_FakeSegment(avg_logprob=0.0)) == pytest.approx(1.0)
    assert _segment_confidence(_FakeSegment(avg_logprob=-2.0)) == pytest.approx(0.135, abs=0.01)


def test_segment_confidence_penalises_non_speech() -> None:
    confident = _segment_confidence(_FakeSegment(avg_logprob=-0.1, no_speech_prob=0.0))
    silent = _segment_confidence(_FakeSegment(avg_logprob=-0.1, no_speech_prob=0.9))
    assert silent < confident


def test_segment_confidence_falls_back_to_word_probabilities() -> None:
    segment = _FakeSegment(words=[_FakeWord(0.8), _FakeWord(0.6)])
    assert _segment_confidence(segment) == pytest.approx(0.7)


def test_transcript_markdown_contains_timestamps(transcript: Transcript) -> None:
    markdown = transcript_to_markdown(transcript)
    assert "00:00:12" in markdown
    assert "## Segmentos" in markdown


def test_extract_audio_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError) as excinfo:
        extract_audio(tmp_path / "nope.mp4")
    assert excinfo.value.stage == "extract_audio"
    assert "recommended action" in excinfo.value.format()


def test_extract_audio_passes_through_wav(tmp_path: Path) -> None:
    wav = tmp_path / "already.wav"
    wav.write_bytes(b"RIFF....WAVE")
    assert extract_audio(wav) == wav


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_extract_audio_from_media(tmp_path: Path) -> None:
    source = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-loglevel", "error", str(source)],
        check=True,
    )
    output = extract_audio(source, tmp_path / "out.wav")
    assert output.exists() and output.stat().st_size > 1000
