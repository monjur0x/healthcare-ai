"""
Embedding generation for retrieval.

An :class:`Embedder` maps a list of texts to a dense ``(N, D)`` matrix.
Two dependency-light implementations ship with the module:

- :class:`TfidfEmbedder` — corpus-fitted TF-IDF vectors (default).
- :class:`HashingEmbedder` — fit-free, fixed-dimension hashing vectors
  for fully offline or test usage.

Swap in a transformer embedder (e.g. sentence-transformers) later
without changing the retriever, which only depends on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from preprocessing.logger import get_logger

from .config import settings
from .exceptions import EmbeddingError

logger = get_logger(__name__)


class Embedder(ABC):
    """
    Maps texts to a dense embedding matrix.
    """

    @abstractmethod
    def fit(self, texts: list[str]) -> Embedder:
        """
        Fit any corpus-dependent state (e.g. vocabulary).

        Parameters
        ----------
        texts : list[str]
            Corpus texts to learn from.

        Returns
        -------
        Embedder
            Self, fitted.
        """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        np.ndarray
            ``(N, D)`` embedding matrix.
        """

    @property
    @abstractmethod
    def dims(self) -> int:
        """
        Embedding dimensionality.

        Returns
        -------
        int
            Width of the produced embedding matrix.
        """

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """
        Fit on the corpus and return its embeddings.

        Parameters
        ----------
        texts : list[str]
            Corpus texts.

        Returns
        -------
        np.ndarray
            ``(N, D)`` embedding matrix.
        """

        return self.fit(texts).embed(texts)


class TfidfEmbedder(Embedder):
    """
    Corpus-fitted TF-IDF embedder.

    Parameters
    ----------
    max_features : int | None
        Vocabulary size cap; defaults to ``settings.MAX_FEATURES``.
    seed : int | None
        Unused, kept for interface consistency.
    """

    def __init__(
        self,
        max_features: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._max_features = (
            settings.MAX_FEATURES if max_features is None else int(max_features)
        )
        self._vectorizer: TfidfVectorizer | None = None

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        """Fit the TF-IDF vocabulary on the corpus."""
        if not texts:
            raise EmbeddingError("Cannot fit an embedder on an empty corpus.")
        self._vectorizer = TfidfVectorizer(max_features=self._max_features)
        self._vectorizer.fit(texts)
        logger.info(
            "Fitted TF-IDF embedder with %d features",
            len(self._vectorizer.vocabulary_),
        )
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts with the fitted TF-IDF vocabulary."""
        if self._vectorizer is None:
            raise EmbeddingError("TfidfEmbedder must be fitted before embedding texts.")
        if not texts:
            raise EmbeddingError("Cannot embed an empty list of texts.")
        return np.asarray(self._vectorizer.transform(texts).todense())

    @property
    def dims(self) -> int:
        """Number of vocabulary features."""
        if self._vectorizer is None:
            raise EmbeddingError("TfidfEmbedder must be fitted before reading dims.")
        return len(self._vectorizer.vocabulary_)


class HashingEmbedder(Embedder):
    """
    Fit-free hashing embedder with a fixed dimensionality.

    Parameters
    ----------
    dims : int | None
        Fixed embedding width; defaults to ``settings.MAX_FEATURES``.
    seed : int | None
        Unused, kept for interface consistency.
    """

    def __init__(
        self,
        dims: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._dims = settings.MAX_FEATURES if dims is None else int(dims)
        if self._dims < 1:
            raise ValueError("dims must be a positive integer.")
        self._vectorizer = HashingVectorizer(
            n_features=self._dims,
            norm="l2",
            alternate_sign=False,
        )

    def fit(self, texts: list[str]) -> HashingEmbedder:
        """Hashing embedding needs no corpus fit."""
        del texts
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """Hash-embed texts into the fixed-dimension space."""
        if not texts:
            raise EmbeddingError("Cannot embed an empty list of texts.")
        return np.asarray(self._vectorizer.transform(texts).todense())

    @property
    def dims(self) -> int:
        """Fixed embedding width."""
        return self._dims


def build_embedder(model: str | None = None) -> Embedder:
    """
    Build the configured embedder by name.

    Parameters
    ----------
    model : str | None
        Embedder name (``"tfidf"`` or ``"hashing"``). Defaults to
        ``settings.EMBEDDING_MODEL``.

    Returns
    -------
    Embedder
        A configured embedder instance.

    Raises
    ------
    ValueError
        If the model name is unknown.
    """

    name = settings.EMBEDDING_MODEL if model is None else model
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedding model '{name}'.")


__all__ = ["Embedder", "HashingEmbedder", "TfidfEmbedder", "build_embedder"]
