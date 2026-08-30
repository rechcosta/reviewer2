"""Vector stores.

FAISS is used when available; otherwise a small NumPy brute-force index is
used, which is perfectly adequate for the corpus sizes involved (one video
plus a handful of reference documents) and keeps the project runnable with
zero extra installation.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import RetrievalConfig
from ..errors import RetrievalError
from ..logging_utils import get_logger
from ..models import DocumentChunk

logger = get_logger(__name__)


class VectorStore(abc.ABC):
    """Stores chunk vectors and answers similarity queries."""

    name: str = "vector-store"

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.chunks: List[DocumentChunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @abc.abstractmethod
    def add(self, chunks: Sequence[DocumentChunk], vectors: np.ndarray) -> None:
        """Index ``chunks`` with their (already normalised) ``vectors``."""

    @abc.abstractmethod
    def search(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Return the ``top_k`` most similar chunks with cosine scores."""

    @abc.abstractmethod
    def save(self, directory: Path) -> None:
        """Persist the index so a second run can skip re-embedding."""

    @classmethod
    @abc.abstractmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Restore an index previously written by :meth:`save`."""

    # -------------------------------------------------------------- #
    def _validate(self, chunks: Sequence[DocumentChunk], vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise RetrievalError(
                "Chunk/vector count mismatch while indexing.",
                module="retrieval.vector_store",
                stage="add",
                cause=f"{len(chunks)} chunks but vectors with shape {vectors.shape}.",
                action="This is an internal error; please report it with the input files.",
            )
        if vectors.shape[1] != self.dimension:
            raise RetrievalError(
                "Embedding dimension mismatch.",
                module="retrieval.vector_store",
                stage="add",
                cause=f"Index expects {self.dimension} dimensions, got {vectors.shape[1]}.",
                action="Delete data/index/ and rebuild after changing the embedding model.",
            )
        return vectors

    def _write_chunks(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.name,
            "dimension": self.dimension,
            "chunks": [chunk.model_dump(mode="json") for chunk in self.chunks],
        }
        (directory / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _read_chunks(directory: Path) -> tuple[int, List[DocumentChunk]]:
        path = Path(directory) / "chunks.json"
        if not path.exists():
            raise RetrievalError(
                f"Vector index metadata not found in {directory}.",
                module="retrieval.vector_store",
                stage="load",
                cause="chunks.json is missing.",
                action="Rebuild the index by running the pipeline with --rebuild-index.",
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk.model_validate(item) for item in payload.get("chunks", [])]
        return int(payload.get("dimension", 0)), chunks


class NumpyVectorStore(VectorStore):
    """Exact brute-force cosine search backed by a single NumPy matrix."""

    name = "numpy"

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self._matrix = np.zeros((0, self.dimension), dtype=np.float32)

    def add(self, chunks: Sequence[DocumentChunk], vectors: np.ndarray) -> None:
        vectors = self._validate(chunks, vectors)
        if len(chunks) == 0:
            return
        self._matrix = np.vstack([self._matrix, vectors]) if len(self.chunks) else vectors
        self.chunks.extend(chunks)

    def search(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        scores = self._matrix @ query
        top_k = max(1, min(int(top_k), len(self.chunks)))
        best = np.argpartition(-scores, top_k - 1)[:top_k]
        best = best[np.argsort(-scores[best])]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in best]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        self._write_chunks(directory)
        np.save(directory / "vectors.npy", self._matrix)

    @classmethod
    def load(cls, directory: Path) -> "NumpyVectorStore":
        directory = Path(directory)
        dimension, chunks = cls._read_chunks(directory)
        matrix = np.load(directory / "vectors.npy")
        store = cls(dimension or int(matrix.shape[1]))
        store.chunks = chunks
        store._matrix = np.asarray(matrix, dtype=np.float32)
        return store


class FaissVectorStore(VectorStore):
    """FAISS ``IndexFlatIP`` over normalised vectors (cosine similarity)."""

    name = "faiss"

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        import faiss  # type: ignore

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self.dimension)

    def add(self, chunks: Sequence[DocumentChunk], vectors: np.ndarray) -> None:
        vectors = self._validate(chunks, vectors)
        if len(chunks) == 0:
            return
        self._index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, vector: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        top_k = max(1, min(int(top_k), len(self.chunks)))
        scores, indices = self._index.search(query, top_k)
        results: List[Tuple[DocumentChunk, float]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append((self.chunks[int(index)], float(score)))
        return results

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        self._write_chunks(directory)
        self._faiss.write_index(self._index, str(directory / "index.faiss"))

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        import faiss  # type: ignore

        directory = Path(directory)
        dimension, chunks = cls._read_chunks(directory)
        store = cls(dimension)
        store._index = faiss.read_index(str(directory / "index.faiss"))
        store.chunks = chunks
        return store


def build_vector_store(dimension: int, config: Optional[RetrievalConfig] = None) -> VectorStore:
    """Instantiate the configured vector store, falling back to NumPy."""
    config = config or RetrievalConfig()
    backend = (config.backend or "auto").strip().lower()

    if backend in {"numpy", "bruteforce", "memory"}:
        return NumpyVectorStore(dimension)

    try:
        store = FaissVectorStore(dimension)
        logger.debug("Using FAISS vector store (dim=%d)", dimension)
        return store
    except ImportError:
        if backend == "faiss":
            raise RetrievalError(
                "FAISS is not installed.",
                module="retrieval.vector_store",
                stage="build",
                cause="retrieval.backend is set to 'faiss' but the package is missing.",
                action="Install it with: pip install faiss-cpu — or set retrieval.backend: auto",
            ) from None
        logger.info("FAISS not available; using the NumPy vector store.")
        return NumpyVectorStore(dimension)


def load_vector_store(directory: Path) -> VectorStore:
    """Load a persisted index, choosing the backend from what is on disk."""
    directory = Path(directory)
    if (directory / "index.faiss").exists():
        try:
            return FaissVectorStore.load(directory)
        except ImportError:
            logger.warning("Index was built with FAISS but FAISS is missing; rebuild required.")
    if (directory / "vectors.npy").exists():
        return NumpyVectorStore.load(directory)
    raise RetrievalError(
        f"No vector index found in {directory}.",
        module="retrieval.vector_store",
        stage="load",
        cause="Neither index.faiss nor vectors.npy is present.",
        action="Run the pipeline again to build the index.",
    )
