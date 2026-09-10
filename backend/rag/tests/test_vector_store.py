"""
Tests for the in-memory vector store.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag import VectorStore
from rag.exceptions import EmptyCorpusError


@pytest.fixture
def store() -> VectorStore:
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ]
    )
    store = VectorStore()
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


def test_cosine_normalizes_vectors(store) -> None:
    store.add(["big"], np.array([[10.0, 0.0, 0.0]]))
    hits = store.search(np.array([1.0, 0.0, 0.0]), top_k=1)
    assert hits[0][0] == "big"
    assert hits[0][1] == pytest.approx(1.0)


def test_search_empty_store_raises() -> None:
    with pytest.raises(EmptyCorpusError):
        VectorStore().search(np.array([1.0]), top_k=1)


def test_add_mismatched_ids_vectors() -> None:
    with pytest.raises(ValueError):
        VectorStore().add(["only-one"], np.zeros((2, 3)))


def test_add_dimension_mismatch(store) -> None:
    with pytest.raises(ValueError):
        store.add(["x"], np.zeros((1, 5)))


def test_invalid_metric() -> None:
    with pytest.raises(ValueError):
        VectorStore(metric="euclidean")
