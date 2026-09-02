"""
Retrieval over an embedded corpus.

The :class:`Retriever` owns the embedder + vector store pairing: it
embeds chunk texts, ingests them into the store, then maps query
vectors back to their nearest chunks. :func:`build_context` composes
the top results into a prompt-ready context block.
"""

from __future__ import annotations

from collections.abc import Sequence

from preprocessing.logger import get_logger

from .config import settings
from .documents import Chunk, RetrievalResult
from .embedder import Embedder
from .exceptions import EmptyCorpusError, EmptyQueryError
from .store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """
    Chunk embedding + nearest-neighbour retrieval.

    Parameters
    ----------
    embedder : Embedder
        Embedding model used for both chunks and queries.
    store : VectorStore | None
        Vector store; a fresh one is created when omitted.
    top_k : int | None
        Default number of results; defaults to ``settings.TOP_K``.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore | None = None,
        top_k: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store if store is not None else VectorStore()
        self._top_k = settings.TOP_K if top_k is None else int(top_k)
        if self._top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        self._chunks: dict[str, Chunk] = {}

    @property
    def embedder(self) -> Embedder:
        """Embedding model in use."""
        return self._embedder

    @property
    def store(self) -> VectorStore:
        """Vector store in use."""
        return self._store

    @property
    def n_chunks(self) -> int:
        """Number of ingested chunks."""
        return len(self._chunks)

    def ingest(self, chunks: Sequence[Chunk]) -> None:
        """
        Embed and index a set of chunks.

        Parameters
        ----------
        chunks : Sequence[Chunk]
            Chunks to ingest.
        """

        if not chunks:
            logger.warning("No chunks to ingest; skipping.")
            return
        if not self._chunks:
            self._embedder.fit([chunk.text for chunk in chunks])
        texts = [chunk.text for chunk in chunks]
        vectors = self._embedder.embed(texts)
        self._store.add([chunk.id for chunk in chunks], vectors)
        self._chunks.update({chunk.id: chunk for chunk in chunks})
        logger.info("Ingested %d chunks (%d total)", len(chunks), len(self._chunks))

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        topic: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the nearest chunks for a query.

        When ``topic`` is given, a wider candidate pool is fetched and
        chunks whose document metadata lists that topic are score-boosted
        before the pool is cut to ``top_k``. This prioritizes
        disease-relevant evidence without hard-excluding general
        clinical sources (e.g. laboratory reference values).

        Parameters
        ----------
        query : str
            Query text.
        top_k : int | None
            Number of results; defaults to the retriever's setting.
        topic : str | None
            Clinical topic tag (e.g. ``"diabetes"``) to prioritize.

        Returns
        -------
        list[RetrievalResult]
            Chunks ordered by descending (possibly boosted) score.

        Raises
        ------
        EmptyQueryError
            If the query is empty or whitespace-only.
        EmptyCorpusError
            If no chunks have been ingested yet.
        """

        if not query.strip():
            raise EmptyQueryError("Query must be non-empty.")
        if not self._chunks:
            raise EmptyCorpusError(
                "Cannot retrieve from an empty corpus; ingest documents first."
            )

        limit = self._top_k if top_k is None else int(top_k)
        query_vector = self._embedder.embed([query])[0]
        candidate_k = limit * 4 if topic else limit
        hits = self._store.search(query_vector, top_k=candidate_k)

        results = []
        for id_, score in hits:
            chunk = self._chunks[id_]
            adjusted = float(score)
            if topic and topic in (chunk.metadata or {}).get("topics", []):
                # Multiplicative boost keeps relative ordering inside the
                # matching group while lifting it above general sources.
                adjusted *= 1.5
            results.append(RetrievalResult(chunk=chunk, score=adjusted))
        if topic:
            results.sort(key=lambda result: result.score, reverse=True)
        results = results[:limit]
        logger.info("Retrieved %d chunks for query", len(results))
        return results

    def build_context(self, query: str, top_k: int | None = None) -> str:
        """
        Compose the top results into a prompt-ready context block.

        Parameters
        ----------
        query : str
            Query text.
        top_k : int | None
            Number of results; defaults to the retriever's setting.

        Returns
        -------
        str
            Source-labelled context text ready to prepend to a prompt.
        """

        results = self.retrieve(query, top_k=top_k)
        labelled = [
            f"[{result.chunk.document_id}] ({result.score:.4f})\n{result.chunk.text}"
            for result in results
        ]
        blocks = [
            (
                f"[{result.chunk.document_id} ({result.chunk.source})] "
                f"({result.score:.4f})\n{result.chunk.text}"
                if result.chunk.source
                else label
            )
            for result, label in zip(results, labelled, strict=True)
        ]
        return "\n\n".join(blocks)


__all__ = ["Retriever"]
