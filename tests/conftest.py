"""Shared fixtures.

The whole suite runs offline: the heuristic LLM provider, hashing embeddings
and the NumPy vector store are used, so no model is downloaded and no
network call is made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Config
from app.llm import HeuristicProvider
from app.models import Transcript, TranscriptSegment

REFERENCE_TEXT = """# Arquitetura — Referência de Teste

## Frequência e desempenho

O aumento da frequência do processador melhora o desempenho apenas quando a carga de
trabalho é limitada por computação. Em cargas limitadas por memória o desempenho passa a
ser determinado pela latência de acesso à memória principal.

## Hierarquia de memória

A memória cache é uma memória volátil construída com células SRAM. O conteúdo da cache é
perdido quando a alimentação elétrica é removida.

## Prefetching

O prefetching reduz a latência efetiva de acesso quando as predições estão corretas.
Quando as predições estão incorretas, o prefetching aumenta o tráfego no barramento,
porém a correlação observada não estabelece causalidade.
"""

TRANSCRIPT_SEGMENTS = [
    (0.0, 10.0, "Hoje vamos falar sobre desempenho de processadores.", 0.95),
    (12.4, 22.0, "Aumentar a frequência do processador sempre aumenta o desempenho.", 0.93),
    (24.0, 34.0, "A memória cache é uma memória não volátil.", 0.9),
    (36.0, 46.0, "Os dados da cache são perdidos quando o computador é desligado.", 0.92),
    (48.0, 58.0, "O prefetching reduz a latência de acesso à memória.", 0.94),
]


@pytest.fixture
def llm() -> HeuristicProvider:
    """Deterministic offline provider."""
    return HeuristicProvider()


@pytest.fixture
def transcript() -> Transcript:
    """A small, fully deterministic transcript."""
    segments = [
        TranscriptSegment(
            segment_id=index, start=start, end=end, text=text, confidence=confidence
        )
        for index, (start, end, text, confidence) in enumerate(TRANSCRIPT_SEGMENTS)
    ]
    return Transcript(
        source="tests/aula.mp4", language="pt", duration=60.0, model="test/manual", segments=segments
    )


@pytest.fixture
def reference_file(tmp_path: Path) -> Path:
    """A Markdown reference document on disk."""
    path = tmp_path / "referencia.md"
    path.write_text(REFERENCE_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def transcript_file(tmp_path: Path, transcript: Transcript) -> Path:
    """The transcript fixture serialised to JSON."""
    path = tmp_path / "aula.json"
    path.write_text(
        json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    return path


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """An offline configuration rooted in a temporary directory."""
    return Config.load(
        use_env=False,
        overrides={
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "videos_dir": str(tmp_path / "data/videos"),
                "documents_dir": str(tmp_path / "data/documents"),
                "transcripts_dir": str(tmp_path / "data/transcripts"),
                "reports_dir": str(tmp_path / "data/reports"),
                "index_dir": str(tmp_path / "data/index"),
                "cache_dir": str(tmp_path / "data/cache"),
            },
            "llm": {"provider": "heuristic"},
            "embeddings": {"backend": "hashing", "hashing_dimension": 256},
            "retrieval": {"backend": "numpy", "persist": False},
        },
    )
