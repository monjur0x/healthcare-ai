"""
In-memory vector store for retrieval.

Stores embedding vectors keyed by chunk id and returns the nearest
neighbours under a similarity metric (cosine by default). The store is
NumPy-only so retrieval stays hermetic and deterministic; a persistent
backend (e.g. Qdrant) can replace it behind the same interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import settings
from .exceptions import EmptyCorpusError


class VectorStore:
    """
    Flat nearest-neighbour store over dense vectors.

    Parameters
    ----------
    metric : str | None
        Similarity metric (``"cosine"`` or ``"dot"``). Defaults to
        ``settings.SIMILARITY_METRIC``.
    """

    def __init__(self, metric: str | None = None) -> None:
        self._metric = settings.SIMILARITY_METRIC if metric is None else metric
        if self._metric not in {"cosine", "dot"}:
            raise ValueError(f"Unsupported similarity metric '{self._metric}'.")
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float64)

    @property
    def metric(self) -> str:
        """Similarity metric used for search."""
        return self._metric

    def __len__(self) -> int:
        """Number of stored vectors."""
        return len(self._ids)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """
        Insert vectors keyed by chunk id.

        Parameters
        ----------
        ids : Sequence[str]
            Chunk ids, one per vector.
        vectors : np.ndarray
            ``(N, D)`` embedding matrix.
        """

        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D (N, D) matrix.")
        if len(ids) != matrix.shape[0]:
            raise ValueError(f"Got {len(ids)} ids but {matrix.shape[0]} vectors.")
        if len(self._ids) and self._vectors.shape[1] != matrix.shape[1]:
            raise ValueError(
                "Vector dimensionality must match existing stored vectors "
                f"({self._vectors.shape[1]})."
            )
        if self._metric == "cosine":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1.0, norms)

        self._vectors = (
            matrix if self._vectors.size == 0 else np.vstack([self._vectors, matrix])
        )
        self._ids.extend(ids)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return the top-k nearest chunk ids and their scores.

        Parameters
        ----------
        query : np.ndarray
            ``(D,)`` query vector.
        top_k : int
            Number of neighbours to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(chunk_id, score)`` pairs, highest score first.

        Raises
        ------
        EmptyCorpusError
            If the store is empty.
        """

        if not self._ids:
            raise EmptyCorpusError(
                "Cannot search an empty vector store; ingest documents first."
            )
        vector = np.asarray(query, dtype=np.float64).ravel()
        if self._metric == "cosine":
            norm = np.linalg.norm(vector)
            vector = vector if norm == 0 else vector / norm
        if vector.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"Query vector has {vector.shape[0]} dims but store expects "
                f"{self._vectors.shape[1]}."
            )

        scores = self._vectors @ vector
        top_k = min(max(int(top_k), 1), len(self._ids))
        order = np.argsort(scores)[::-1][:top_k]
        return [(self._ids[index], float(scores[index])) for index in order]


def build_vector_store(backend: str | None = None) -> VectorStore | ChromaVectorStore:
    """
    Build the configured vector store by name.

    Parameters
    ----------
    backend : str | None
        Store backend name (``"memory"`` or ``"chroma"``). Defaults to
        ``settings.VECTOR_STORE``.

    Returns
    -------
    VectorStore | ChromaVectorStore
        A configured store instance implementing the same ``add`` /
        ``search`` / ``__len__`` interface.

    Raises
    ------
    ValueError
        If the backend name is unknown.
    """

    name = settings.VECTOR_STORE if backend is None else backend
    if name == "memory":
        return VectorStore()
    if name == "chroma":
        from .store_chroma import ChromaVectorStore

        return ChromaVectorStore()
    raise ValueError(f"Unknown vector store backend '{name}'.")


__all__ = ["VectorStore", "build_vector_store"]
