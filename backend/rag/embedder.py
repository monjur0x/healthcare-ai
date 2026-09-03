"""
Embedding generation for retrieval.

An :class:`Embedder` maps a list of texts to a dense ``(N, D)`` matrix.
Three implementations ship with the module:

- :class:`TfidfEmbedder` — corpus-fitted TF-IDF vectors (default).
- :class:`HashingEmbedder` — fit-free, fixed-dimension hashing vectors
  for fully offline or test usage.
- :class:`SentenceTransformerEmbedder` — dense transformer embeddings
  (opt-in; downloads a small model from Hugging Face on first use).
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

    #: True when embeddings depend on the fitted corpus vocabulary and
    #: must be recomputed for every chunk if new documents arrive.
    corpus_dependent: bool = False

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

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed one retrieval query.

        Defaults to plain :meth:`embed`; dense models with asymmetric
        query handling (e.g. BGE instructions) override this.

        Parameters
        ----------
        query : str
            Query text.

        Returns
        -------
        np.ndarray
            ``(D,)`` query vector.
        """

        return self.embed([query])[0]


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

    corpus_dependent = True

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


class SentenceTransformerEmbedder(Embedder):
    """
    Dense embeddings from a sentence-transformer model.

    Wraps a pretrained ``sentence_transformers.SentenceTransformer``
    (e.g. BAAI BGE or all-MiniLM). The model is loaded lazily on first
    use and cached, so the class constructs cheaply even before the
    dependency or a network connection is available.

    For BGE models the official retrieval usage prepends a query
    instruction to queries (not to documents); the flag
    ``query_instruction`` defaults to the BGE English prompt for exactly
    that behaviour.

    Parameters
    ----------
    model_name : str | None
        Sentence-transformer model name; defaults to
        ``settings.SENTENCE_TRANSFORMER_MODEL``.
    seed : int | None
        Unused, kept for interface consistency (these models are
        deterministic).
    query_instruction : bool
        Prepend the BGE query instruction to embedded queries.
    """

    def __init__(
        self,
        model_name: str | None = None,
        seed: int | None = None,
        query_instruction: bool = True,
    ) -> None:
        del seed  # pretrained transformers are deterministic
        self._model_name = (
            settings.SENTENCE_TRANSFORMER_MODEL if model_name is None else model_name
        )
        self._query_instruction = query_instruction
        self._model: object | None = None
        self._dims: int | None = None

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        """Pretrained transformers need no corpus fit."""
        del texts
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed document texts with the loaded sentence-transformer model."""
        if not texts:
            raise EmbeddingError("Cannot embed an empty list of texts.")
        return self._encode(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query, prepending the BGE instruction (queries only)."""
        if not query.strip():
            raise EmbeddingError("Cannot embed an empty query.")
        texts = [query]
        if self._query_instruction:
            texts = [
                f"Represent this sentence for searching relevant passages: {query}"
            ]
        return self._encode(texts)[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Run the model and cache the embedding width."""
        model = self._load_model()
        embeddings = np.asarray(model.encode(texts, convert_to_numpy=True))
        if self._dims is None:
            self._dims = embeddings.shape[1]
        return embeddings.astype(np.float32)

    @property
    def dims(self) -> int:
        """Embedding width of the loaded model."""
        if self._dims is None:
            self._load_model()
        return int(self._dims or 0)

    def _load_model(self) -> object:
        """Load (and cache) the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise EmbeddingError(
                    "SentenceTransformerEmbedder requires "
                    "'sentence-transformers>=3.0.0'; add it to requirements "
                    "or use EMBEDDING_MODEL=tfidf."
                ) from error
            logger.info(
                "Loading sentence-transformer model %s (first use may download)",
                self._model_name,
            )
            self._model = SentenceTransformer(self._model_name)
            dimension = getattr(self._model, "get_embedding_dimension", None)
            if dimension is None:
                dimension = self._model.get_sentence_embedding_dimension
            self._dims = int(dimension())
        return self._model


def build_embedder(model: str | None = None) -> Embedder:
    """
    Build the configured embedder by name.

    Parameters
    ----------
    model : str | None
        Embedder name (``"tfidf"``, ``"hashing"``, or
        ``"sentence-transformer"``). Defaults to
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
    if name == "sentence-transformer":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedding model '{name}'.")


__all__ = [
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "TfidfEmbedder",
    "build_embedder",
]
