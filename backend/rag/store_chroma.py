"""
Persistent ChromaDB-backed vector store for retrieval.

Implements the same narrow interface as :class:`rag.store.VectorStore`
(add / search / ``__len__`` / ``metric``) over a persistent
``chromadb.PersistentClient`` collection, so the retriever and pipeline
never need to know which backend is active. Cosine similarity is the
only metric supported by the Chroma store.
"""

from __future__ import annotations

import numpy as np

from .config import settings

try:
    import chromadb

    _CHROMADB_AVAILABLE = True
except ImportError:
    _CHROMADB_AVAILABLE = False
    chromadb = None


class ChromaVectorStore:
    """
    ChromaDB-backed vector store for retrieval.

    Requires ``chromadb`` package. If not available, raises ImportError
    on instantiation.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "healthcare_rag",
    ) -> None:
        if not _CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed; add 'chromadb>=0.5.0' to requirements "
                "or set RAG_VECTOR_STORE=memory."
            )
        persist = persist_dir or settings.CHROMA_PERSIST_DIR
        self._client = chromadb.PersistentClient(path=persist)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._metric = "cosine"

    @property
    def metric(self) -> str:
        """Similarity metric used for search."""
        return self._metric

    def __len__(self) -> int:
        """Number of stored vectors."""
        return self._collection.count()

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """
        Insert vectors keyed by chunk id.

        Parameters
        ----------
        ids : list[str]
            Chunk ids, one per vector.
        vectors : np.ndarray
            ``(N, D)`` embedding matrix.
        """
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D (N, D) matrix.")
        if len(ids) != matrix.shape[0]:
            raise ValueError(f"Got {len(ids)} ids but {matrix.shape[0]} vectors.")
        if len(self._collection.get()["ids"]) and self._collection.get()["embeddings"]:
            existing_dim = len(self._collection.get()["embeddings"][0])
            if existing_dim != matrix.shape[1]:
                raise ValueError(
                    "Vector dimensionality must match existing stored vectors "
                    f"({existing_dim})."
                )

        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms == 0, 1.0, norms)

        self._collection.add(
            ids=ids,
            embeddings=matrix.tolist(),
        )

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
        if self._collection.count() == 0:
            from .exceptions import EmptyCorpusError

            raise EmptyCorpusError(
                "Cannot search an empty vector store; ingest documents first."
            )
        vector = np.asarray(query, dtype=np.float64).ravel()
        if self._metric == "cosine":
            norm = np.linalg.norm(vector)
            vector = vector if norm == 0 else vector / norm

        results = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k, self._collection.count()),
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        # Convert distances to similarity scores (cosine: 1 - distance)
        scores = [1.0 - d for d in distances]
        return list(zip(ids, scores, strict=True))


_CHROMADB_AVAILABLE = False
