"""Embedding backends.

Two implementations are provided:

* :class:`SentenceTransformerEmbeddings` — the quality default, fully local
  and free (Sentence Transformers).
* :class:`HashingEmbeddings` — a dependency-light deterministic fallback
  used for tests and for machines where downloading a model is not an
  option. It is lexical, so it is weaker; the report states when it is used.
"""

from __future__ import annotations

import abc
import hashlib
import re
import unicodedata
from typing import List, Optional, Sequence

import numpy as np

from ..config import EmbeddingsConfig
from ..errors import EmbeddingError, MissingDependencyError
from ..logging_utils import get_logger

logger = get_logger(__name__)


class EmbeddingModel(abc.ABC):
    """Turns text into L2-normalised vectors."""

    name: str = "embedding"
    dimension: int = 0

    @abc.abstractmethod
    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        """Return a ``(len(texts), dimension)`` float32 array."""

    def encode_one(self, text: str, *, is_query: bool = False) -> np.ndarray:
        """Encode a single string."""
        return self.encode([text], is_query=is_query)[0]

    @staticmethod
    def normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise rows so that dot product equals cosine similarity."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)


class SentenceTransformerEmbeddings(EmbeddingModel):
    """Local Sentence Transformers backend."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, *, device: str = "cpu", batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise MissingDependencyError(
                "sentence-transformers",
                module="embeddings.model",
                stage="load",
                extra="semantic embeddings",
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load the embedding model '{model_name}'.",
                module="embeddings.model",
                stage="load",
                cause=str(exc)[:300],
                action=(
                    "Check the model name and the network connection for the first download, "
                    "or set embeddings.backend: hashing for an offline run."
                ),
            ) from exc
        self.dimension = int(_embedding_dimension(self._model))

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return self.normalize(np.asarray(vectors, dtype=np.float32))


class HashingEmbeddings(EmbeddingModel):
    """Deterministic lexical embeddings (hashed bag of words + character n-grams).

    No model download, no network, stable across runs. Good enough to make
    retrieval work offline and to keep the test suite hermetic.
    """

    name = "hashing"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = int(dimension)
        self.model_name = f"hashing-{self.dimension}"

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in _features(text):
                index = _bucket(feature, self.dimension)
                matrix[row, index] += weight
        return self.normalize(matrix)


def _embedding_dimension(model) -> int:
    """Read the vector size across Sentence Transformers versions.

    ``get_sentence_embedding_dimension`` was renamed to
    ``get_embedding_dimension`` in v6; both are supported here.
    """
    for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attribute, None)
        if callable(getter):
            value = getter()
            if value:
                return int(value)
    # Last resort: encode a probe string and measure the result.
    return int(len(model.encode(["dimension probe"], convert_to_numpy=True)[0]))


# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _features(text: str) -> List[tuple[str, float]]:
    """Words, word bigrams and character 4-grams with sub-linear weights."""
    tokens = _TOKEN_RE.findall(_strip_accents(text))
    features: List[tuple[str, float]] = [(f"w:{token}", 1.0) for token in tokens if len(token) > 1]
    features += [(f"b:{a}_{b}", 0.6) for a, b in zip(tokens, tokens[1:])]
    flat = " ".join(tokens)
    features += [(f"c:{flat[i : i + 4]}", 0.25) for i in range(0, max(0, len(flat) - 3))]
    return features


def _bucket(feature: str, dimension: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def build_embedding_model(config: Optional[EmbeddingsConfig] = None) -> EmbeddingModel:
    """Instantiate the configured embedding backend, with a safe fallback."""
    config = config or EmbeddingsConfig()
    backend = (config.backend or "auto").strip().lower()

    if backend in {"hashing", "offline", "fallback"}:
        logger.info("Using hashing embeddings (dimension=%d)", config.hashing_dimension)
        return HashingEmbeddings(config.hashing_dimension)

    try:
        model = SentenceTransformerEmbeddings(config.model, device=config.device, batch_size=config.batch_size)
        logger.info("Using sentence-transformers embeddings: %s (dim=%d)", config.model, model.dimension)
        return model
    except (MissingDependencyError, EmbeddingError) as exc:
        if backend == "sentence-transformers":
            raise
        logger.warning(
            "Falling back to hashing embeddings: %s", getattr(exc, "message", str(exc))
        )
        return HashingEmbeddings(config.hashing_dimension)
