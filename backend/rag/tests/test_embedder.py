"""
Tests for the RAG embedders.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag import HashingEmbedder, TfidfEmbedder, build_embedder
from rag.exceptions import EmbeddingError


def test_tfidf_fit_transform_shape() -> None:
    embedder = TfidfEmbedder(max_features=100)
    vectors = embedder.fit_transform(["diabetes mellitus", "heart failure"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] == embedder.dims
    assert embedder.dims > 0


def test_tfidf_requires_fit_before_embed() -> None:
    embedder = TfidfEmbedder(max_features=10)
    with pytest.raises(EmbeddingError):
        embedder.embed(["hello"])


def test_tfidf_rejects_empty_fit() -> None:
    with pytest.raises(EmbeddingError):
        TfidfEmbedder(max_features=10).fit([])


def test_hashing_fixed_dims_no_fit() -> None:
    embedder = HashingEmbedder(dims=32)
    vectors = embedder.fit_transform(["one two", "three four"])
    assert vectors.shape == (2, 32)
    assert embedder.dims == 32


def test_embedders_reject_empty_input() -> None:
    with pytest.raises(EmbeddingError):
        TfidfEmbedder(max_features=10).fit(["diabetes"]).embed([])
    with pytest.raises(EmbeddingError):
        HashingEmbedder(dims=8).embed([])


def test_build_embedder_defaults_and_unknown() -> None:
    assert isinstance(build_embedder("tfidf"), TfidfEmbedder)
    assert isinstance(build_embedder("hashing"), HashingEmbedder)
    with pytest.raises(ValueError):
        build_embedder("nope")


def test_similar_texts_are_more_similar() -> None:
    embedder = TfidfEmbedder(max_features=50)
    vectors = embedder.fit_transform(
        [
            "the patient has type 2 diabetes",
            "insulin resistance in type 2 diabetes",
            "the capital of france is paris",
        ]
    )

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2])
