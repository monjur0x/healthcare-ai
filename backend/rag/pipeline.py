"""
High-level RAG pipeline.

The reusable entry point for CrewAI, FastAPI, and examples. It composes
chunking -> embedding -> vector store -> retrieval into a single object
with ``ingest_documents`` / ``ingest_texts`` and ``query`` / context
building, mirroring the CSV and image preprocessing pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable

from preprocessing.logger import get_logger

from .chunker import TextChunker
from .documents import Document, RetrievalResult
from .embedder import Embedder, build_embedder
from .retriever import Retriever
from .store import VectorStore

logger = get_logger(__name__)


class RAGPipeline:
    """
    End-to-end retrieval pipeline over a text corpus.

    Parameters
    ----------
    chunker : TextChunker | None
        Chunking strategy; a default one is created when omitted.
    embedder : Embedder | None
        Embedding model; the configured default is created when omitted.
    store : VectorStore | None
        Vector store; a fresh one is created when omitted.
    top_k : int | None
        Default number of results; defaults to ``settings.TOP_K``.
    """

    def __init__(
        self,
        chunker: TextChunker | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        top_k: int | None = None,
    ) -> None:
        self._chunker = chunker if chunker is not None else TextChunker()
        self._embedder = embedder if embedder is not None else build_embedder()
        self._retriever = Retriever(embedder=self._embedder, store=store, top_k=top_k)

    @property
    def retriever(self) -> Retriever:
        """Underlying retriever."""
        return self._retriever

    @property
    def n_chunks(self) -> int:
        """Number of ingested chunks."""
        return self._retriever.n_chunks

    def ingest_documents(self, documents: Iterable[Document]) -> int:
        """
        Chunk, embed, and index a collection of documents.

        Parameters
        ----------
        documents : Iterable[Document]
            Documents to ingest.

        Returns
        -------
        int
            Number of chunks indexed.
        """

        chunks = self._chunker.chunk_documents(documents)
        self._retriever.ingest(chunks)
        logger.info(
            "Indexed %d documents as %d chunks", sum(1 for _ in documents), len(chunks)
        )
        return len(chunks)

    def ingest_texts(
        self,
        texts: Iterable[str],
        sources: Iterable[str] | None = None,
    ) -> int:
        """
        Ingest raw texts as anonymous documents.

        Parameters
        ----------
        texts : Iterable[str]
            Raw texts, one per document.
        sources : Iterable[str] | None
            Optional source label per text.

        Returns
        -------
        int
            Number of chunks indexed.
        """

        items = list(texts)
        labels = list(sources) if sources is not None else [""] * len(items)
        documents = [
            Document(id=f"doc-{index}", text=text, source=label)
            for index, (text, label) in enumerate(zip(items, labels, strict=True))
        ]
        return self.ingest_documents(documents)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """
        Retrieve the nearest chunks for a query.

        Parameters
        ----------
        query : str
            Query text.
        top_k : int | None
            Number of results; defaults to the pipeline setting.

        Returns
        -------
        list[RetrievalResult]
            Chunks ordered by descending score.
        """

        return self._retriever.retrieve(query, top_k=top_k)

    def build_context(self, query: str, top_k: int | None = None) -> str:
        """
        Compose a prompt-ready context block for a query.

        Parameters
        ----------
        query : str
            Query text.
        top_k : int | None
            Number of results; defaults to the pipeline setting.

        Returns
        -------
        str
            Source-labelled context text ready to prepend to a prompt.
        """

        return self._retriever.build_context(query, top_k=top_k)


__all__ = ["RAGPipeline"]
