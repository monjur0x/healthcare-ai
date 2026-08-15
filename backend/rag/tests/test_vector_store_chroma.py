"""
Tests for the persistent ChromaDB vector store.

Mirrors the in-memory store's test cases against ``ChromaVectorStore``
using a temp directory so each test gets an isolated collection.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag import ChromaVectorStore
from rag.exceptions import EmptyCorpusError


@pytest.fixture
def store(tmp_path) -> ChromaVectorStore:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    store = ChromaVectorStore(persist_dir=str(tmp_path))
    store.add(["a", "b", "c"], vectors)
    return store


def test_len_and_search(store) -> None:
    assert len(store) == 3
    hits = store.search(np.array([1.0, 0.0, 0.0]), top_k=2)
    assert hits[0][0] == "a"
    assert len(hits) == 2


def test_search_returns_sorted_scores(store) -> None:
    hits = store.search(np.array([0.9, 0.1, 0.0]), top_k=3)
    assert [score for _, score in hits] == sorted((s for _, s in hits), reverse=True)
    assert hits[0][0] == "c"


def test_cosine_normalizes_vectors(tmp_path) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path))
    store.add(["big"], np.array([[10.0, 0.0, 0.0]], dtype=np.float32))
    hits = store.search(np.array([1.0, 0.0, 0.0]), top_k=1)
    assert hits[0][0] == "big"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)


def test_search_empty_store_raises(tmp_path) -> None:
    with pytest.raises(EmptyCorpusError):
        ChromaVectorStore(persist_dir=str(tmp_path)).search(np.array([1.0]), top_k=1)


def test_add_mismatched_ids_vectors(tmp_path) -> None:
    with pytest.raises(ValueError):
        ChromaVectorStore(persist_dir=str(tmp_path)).add(
            ["only-one"], np.zeros((2, 3))
        )


def test_invalid_metric(tmp_path) -> None:
    with pytest.raises(ValueError, match="cosine"):
        ChromaVectorStore(persist_dir=str(tmp_path), metric="euclidean")


def test_persists_across_reopen(tmp_path) -> None:
    path = str(tmp_path / "chroma")
    store = ChromaVectorStore(persist_dir=path)
    store.add(["a", "b"], np.eye(2, dtype=np.float32))
    reopened = ChromaVectorStore(persist_dir=path)
    assert len(reopened) == 2
    hits = reopened.search(np.array([1.0, 0.0]), top_k=1)
    assert hits[0][0] == "a"


def test_build_vector_store_backends() -> None:
    from rag import VectorStore, build_vector_store

    assert isinstance(build_vector_store("memory"), VectorStore)
    with pytest.raises(ValueError):
        build_vector_store("nope")