"""Vector store and evidence retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Config
from app.embeddings import HashingEmbeddings
from app.errors import RetrievalError
from app.models import DocumentChunk
from app.retrieval import EvidenceRetriever, NumpyVectorStore, load_vector_store


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(chunk_id="d_0001_00", document="d.md", text="A cache é uma memória volátil.", page=1),
        DocumentChunk(chunk_id="d_0002_00", document="d.md", text="O prefetching reduz a latência.", page=2),
        DocumentChunk(chunk_id="d_0003_00", document="d.md", text="Receita de bolo de chocolate.", page=3),
    ]


def test_store_add_and_search() -> None:
    model = HashingEmbeddings(256)
    chunks = _chunks()
    store = NumpyVectorStore(model.dimension)
    store.add(chunks, model.encode([chunk.text for chunk in chunks]))

    results = store.search(model.encode_one("memória cache volátil"), top_k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id == "d_0001_00"
    assert results[0][1] >= results[1][1]


def test_search_on_empty_store_returns_nothing() -> None:
    assert NumpyVectorStore(16).search(np.zeros(16, dtype=np.float32)) == []


def test_dimension_mismatch_is_rejected() -> None:
    store = NumpyVectorStore(8)
    with pytest.raises(RetrievalError) as excinfo:
        store.add(_chunks()[:1], np.zeros((1, 16), dtype=np.float32))
    assert "dimension" in excinfo.value.message.lower()


def test_store_round_trip(tmp_path: Path) -> None:
    model = HashingEmbeddings(128)
    chunks = _chunks()
    store = NumpyVectorStore(model.dimension)
    store.add(chunks, model.encode([chunk.text for chunk in chunks]))
    store.save(tmp_path / "index")

    restored = load_vector_store(tmp_path / "index")
    assert len(restored) == 3
    assert restored.chunks[0].page == 1
    assert restored.search(model.encode_one("prefetching latência"), top_k=1)[0][0].chunk_id == "d_0002_00"


def test_loading_a_missing_index_reports_action(tmp_path: Path) -> None:
    with pytest.raises(RetrievalError) as excinfo:
        load_vector_store(tmp_path / "nothing")
    assert "recommended action" in excinfo.value.format()


def test_retriever_records_traces(config: Config, reference_file: Path) -> None:
    retriever = EvidenceRetriever(config)
    retriever.index_references([str(reference_file)])

    trace = retriever.retrieve("frequência do processador e desempenho", claim_id="claim_001")
    assert trace.claim_id == "claim_001"
    assert trace.results
    assert trace.results[0].score >= trace.threshold
    assert retriever.traces[-1] is trace
    assert all(result.chunk.document == "referencia.md" for result in trace.results)


def test_retrieval_threshold_filters_unrelated_queries(config: Config, reference_file: Path) -> None:
    retriever = EvidenceRetriever(config)
    retriever.index_references([str(reference_file)])
    trace = retriever.retrieve("resultados do campeonato de futebol de 1998", min_score=0.5)
    assert trace.results == []
    assert trace.best_score == 0.0


def test_traces_are_serialisable(config: Config, reference_file: Path, tmp_path: Path) -> None:
    retriever = EvidenceRetriever(config)
    retriever.index_references([str(reference_file)])
    retriever.retrieve("cache volátil", claim_id="claim_001")
    path = retriever.save_traces(tmp_path / "traces.json")

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["claim_id"] == "claim_001"
    assert "results" in payload[0]


def test_indexing_without_content_reports_action(config: Config, tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    retriever = EvidenceRetriever(config)
    with pytest.raises((RetrievalError, Exception)):
        retriever.index_references([str(empty)])
