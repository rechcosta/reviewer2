"""Retrieval over the reference corpus.

Every query is recorded as a :class:`RetrievalTrace` (query, chunks, scores,
document, page) so that any conclusion in the final report can be audited
back to the evidence that produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Sequence

from ..config import Config
from ..documents import chunk_documents, load_documents
from ..documents.parser import ParsedDocument
from ..embeddings import EmbeddingModel, build_embedding_model
from ..errors import RetrievalError
from ..logging_utils import get_logger
from ..models import DocumentChunk, RetrievalTrace, RetrievedChunk
from .vector_store import VectorStore, build_vector_store, load_vector_store

logger = get_logger(__name__)


class EvidenceRetriever:
    """Indexes reference documents and retrieves evidence for claims."""

    def __init__(
        self,
        config: Config,
        embedding_model: Optional[EmbeddingModel] = None,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.config = config
        self.embeddings = embedding_model or build_embedding_model(config.embeddings)
        self.store = store or build_vector_store(self.embeddings.dimension, config.retrieval)
        self.traces: List[RetrievalTrace] = []
        self.documents: List[str] = []

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_documents(self, documents: Sequence[ParsedDocument]) -> int:
        """Chunk, embed and index a set of parsed documents."""
        chunks = chunk_documents(list(documents), self.config.documents)
        return self.index_chunks(chunks, document_names=[d.name for d in documents])

    def index_chunks(self, chunks: Sequence[DocumentChunk], document_names: Optional[List[str]] = None) -> int:
        """Embed and index already-chunked material."""
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            raise RetrievalError(
                "There is nothing to index: the reference material produced no chunks.",
                module="retrieval.retriever",
                stage="index",
                cause="Every document was empty or unreadable.",
                action="Check the reference files (a scanned PDF needs OCR first).",
            )
        vectors = self.embeddings.encode([chunk.text for chunk in chunks])
        self.store.add(chunks, vectors)
        self.documents = document_names or sorted({chunk.document for chunk in chunks})
        logger.info("Indexed %d chunks from %d document(s)", len(chunks), len(set(c.document for c in chunks)))
        return len(chunks)

    def index_references(self, references: Sequence[str], *, rebuild: bool = False) -> int:
        """Load, chunk and index reference paths/URLs, using the disk cache."""
        index_dir = Path(self.config.paths.index_dir) / _corpus_key(references, self.config)
        if self.config.retrieval.persist and not rebuild and (index_dir / "chunks.json").exists():
            try:
                self.store = load_vector_store(index_dir)
                self.documents = sorted({chunk.document for chunk in self.store.chunks})
                logger.info("Reusing existing index (%d chunks) from %s", len(self.store), index_dir)
                return len(self.store)
            except RetrievalError as exc:
                logger.warning("Could not reuse the index (%s); rebuilding.", exc.message)

        documents = load_documents(list(references), self.config.documents)
        count = self.index_documents(documents)
        if self.config.retrieval.persist:
            self.store.save(index_dir)
            logger.debug("Index saved to %s", index_dir)
        return count

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        *,
        claim_id: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RetrievalTrace:
        """Retrieve evidence candidates for ``query`` and record the trace."""
        top_k = top_k or self.config.retrieval.top_k
        min_score = self.config.retrieval.min_score if min_score is None else min_score

        if len(self.store) == 0:
            trace = RetrievalTrace(query=query, claim_id=claim_id, threshold=min_score, top_k=top_k)
            self.traces.append(trace)
            return trace

        vector = self.embeddings.encode_one(query, is_query=True)
        raw = self.store.search(vector, top_k=max(top_k * 2, top_k))

        boost = self.config.retrieval.lexical_boost
        scored: List[RetrievedChunk] = []
        for chunk, score in raw:
            final = score + boost * _lexical_overlap(query, chunk.text) if boost else score
            scored.append(RetrievedChunk(chunk=chunk, score=round(float(final), 4)))
        scored.sort(key=lambda item: item.score, reverse=True)

        results = [item for item in scored if item.score >= min_score][:top_k]
        trace = RetrievalTrace(
            query=query, claim_id=claim_id, results=results, threshold=min_score, top_k=top_k
        )
        self.traces.append(trace)
        logger.debug(
            "Retrieval for %s: %d/%d chunks above %.2f (best=%.3f)",
            claim_id or "query", len(results), len(scored), min_score, trace.best_score,
        )
        return trace

    def save_traces(self, path: Path) -> Path:
        """Write every retrieval trace as JSON for offline auditing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [trace.model_dump(mode="json") for trace in self.traces]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _lexical_overlap(query: str, text: str) -> float:
    """Fraction of the query's content words present in ``text``."""
    import re

    def tokens(value: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z0-9À-ÿ]+", value.lower()) if len(t) > 3}

    query_tokens = tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & tokens(text)) / len(query_tokens)


def _corpus_key(references: Sequence[str], config: Config) -> str:
    """Stable cache key: references + chunking + embedding settings."""
    material = json.dumps(
        {
            "references": sorted(str(r) for r in references),
            "chunk_size": config.documents.chunk_size,
            "chunk_overlap": config.documents.chunk_overlap,
            "embeddings": f"{config.embeddings.backend}:{config.embeddings.model}",
            "mtimes": [
                str(Path(r).stat().st_mtime) if Path(r).exists() else "url" for r in sorted(map(str, references))
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
