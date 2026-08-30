"""Embedding backends."""

from __future__ import annotations

import numpy as np

from app.config import EmbeddingsConfig
from app.embeddings import HashingEmbeddings, build_embedding_model


def test_vectors_are_normalised() -> None:
    model = HashingEmbeddings(128)
    vectors = model.encode(["texto de teste", "outro texto"])
    assert vectors.shape == (2, 128)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encoding_is_deterministic() -> None:
    first = HashingEmbeddings(128).encode(["arquitetura de computadores"])
    second = HashingEmbeddings(128).encode(["arquitetura de computadores"])
    assert np.allclose(first, second)


def test_related_texts_are_closer_than_unrelated() -> None:
    model = HashingEmbeddings(512)
    vectors = model.encode(
        [
            "aumentar a frequência do processador aumenta o desempenho",
            "a frequência do processador influencia o desempenho",
            "receita de bolo de chocolate com cobertura",
        ]
    )
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_accents_do_not_break_similarity() -> None:
    model = HashingEmbeddings(512)
    vectors = model.encode(["memória volátil", "memoria volatil"])
    assert float(vectors[0] @ vectors[1]) > 0.95


def test_empty_input_returns_empty_matrix() -> None:
    assert HashingEmbeddings(64).encode([]).shape == (0, 64)


def test_build_falls_back_to_hashing_when_backend_missing() -> None:
    model = build_embedding_model(EmbeddingsConfig(backend="hashing", hashing_dimension=64))
    assert isinstance(model, HashingEmbeddings)
    assert model.dimension == 64
