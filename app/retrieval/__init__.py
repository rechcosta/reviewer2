"""Retrieval layer: vector store plus evidence retriever."""

from .retriever import EvidenceRetriever
from .vector_store import (
    FaissVectorStore,
    NumpyVectorStore,
    VectorStore,
    build_vector_store,
    load_vector_store,
)

__all__ = [
    "EvidenceRetriever",
    "VectorStore",
    "NumpyVectorStore",
    "FaissVectorStore",
    "build_vector_store",
    "load_vector_store",
]
