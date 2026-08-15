"""
Persistent ChromaDB-backed vector store for retrieval.

Implements the same narrow interface as :class:`rag.store.VectorStore`
(add / search / ``__len__`` / ``metric``) over a persistent
``chromadb.PersistentClient`` collection, so the retriever and pipeline
never need to know which backend is active. Cosine similarity is the
only metric supported by the Chroma store.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import settings
from .exceptions import EmptyCorpusError

try:
    import chromadb
except ImportError as error:  # pragma: no cover - exercised only when deps are missing
    raise ImportError(
        "ChromaDB is not installed; add 'chromadb>=0.5.0' to requirements "
        "or set RAG_VECTOR_STORE=memory."
    ) from error


class ChromaVectorStore:
    """
    Persistent nearest-neighbour store over dense vectors.

    Parameters
    ----------
    persist_dir : str | None
        Directory for the persistent ChromaDB data. When empty, an
        in-memory client is used (nothing is written to disk).
    collection_name : str | None
        ChromaDB collection name; defaults to ``settings.CHROMA_COLLECTION``.
    metric : str | None
        Similarity metric. ChromaDB supports ``"cosine"`` (default);
        ``"dot"`` raises ``ValueError``.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        metric: str | None = None,
    ) -> None:
        self._metric = settings.SIMILARITY_METRIC if metric is None else metric
        if self._metric != "cosine":
            raise ValueError(
                f"Unsupported similarity metric '{self._metric}' for ChromaDB; "
                "only 'cosine' is supported."
            )
        directory = settings.CHROMA_PERSIST_DIR if persist_dir is None else persist_dir
        name = (
            settings.CHROMA_COLLECTION if collection_name is None else collection_name
        )
        if directory:
            self._client = chromadb.PersistentClient(path=directory)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def metric(self) -> str:
        """Similarity metric used for search."""
        return self._metric

    def __len__(self) -> int:
        """Number of stored vectors."""
        return int(self._collection.count())

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

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D (N, D) matrix.")
        if len(ids) != matrix.shape[0]:
            raise ValueError(f"Got {len(ids)} ids but {matrix.shape[0]} vectors.")
        self._collection.add(
            ids=list(ids),
            embeddings=matrix.tolist(),
        )

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return the top-k nearest chunk ids and their cosine scores.

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

        if len(self) == 0:
            raise EmptyCorpusError(
                "Cannot search an empty vector store; ingest documents first."
            )
        vector = np.asarray(query, dtype=np.float32).ravel()
        if vector.ndim != 1:
            raise ValueError(f"Query vector must be 1D, got {vector.ndim}D.")
        limit = max(int(top_k), 1)
        response = self._collection.query(
            query_embeddings=vector.tolist(),
            n_results=min(limit, len(self)),
        )
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        hits = [
            (str(id_), float(1.0 - distance))
            for id_, distance in zip(ids, distances, strict=True)
        ]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits


__all__ = ["ChromaVectorStore"]
