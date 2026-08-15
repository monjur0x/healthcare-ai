"""
Tests for the sentence-transformer embedder (opt-in, network-dependent).

The embedder downloads a small model from Hugging Face on first use, so
these tests skip gracefully when the dependency is missing or the model
cannot be fetched (mirroring the Opacus availability pattern in
``federated/tests/test_privacy.py``). The default TF-IDF embedder stays
the hermetic, always-runnable path.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag import SentenceTransformerEmbedder, build_embedder
from rag.exceptions import EmbeddingError


def _model_available() -> bool:
    """True when sentence-transformers is installed and reachable."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    try:
        embedder = SentenceTransformerEmbedder()
        embedder.embed(["availability probe"])
    except Exception:
        return False
    return True


MODEL_AVAILABLE = _model_available()

requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="sentence-transformers not installed or model unavailable offline",
)


def test_build_embedder_dispatches_sentence_transformer() -> None:
    assert isinstance(
        build_embedder("sentence-transformer"), SentenceTransformerEmbedder
    )
    with pytest.raises(ValueError):
        build_embedder("nope")


def test_sentence_transformer_fit_is_noop() -> None:
    embedder = SentenceTransformerEmbedder()
    assert embedder.fit(["any text"]) is embedder


@requires_model
def test_sentence_transformer_embed_shape() -> None:
    embedder = SentenceTransformerEmbedder()
    vectors = embedder.fit_transform(["diabetes mellitus", "heart failure"])
    assert vectors.shape == (2, embedder.dims)
    assert vectors.shape[1] > 0


@requires_model
def test_sentence_transformer_similar_texts_are_more_similar() -> None:
    embedder = SentenceTransformerEmbedder(query_instruction=False)
    vectors = embedder.embed(
        [
            "the patient has type 2 diabetes",
            "insulin resistance in type 2 diabetes",
            "the capital of france is paris",
        ]
    )

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cosine(vectors[0], vectors[1]) > cosine(vectors[0], vectors[2])


def test_sentence_transformer_rejects_empty_input() -> None:
    with pytest.raises(EmbeddingError):
        SentenceTransformerEmbedder().embed([])


def test_sentence_transformer_missing_dependency_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    embedder = SentenceTransformerEmbedder(model_name="unused-local-path")
    with pytest.raises(EmbeddingError, match="sentence-transformers"):
        embedder.embed(["hello"])