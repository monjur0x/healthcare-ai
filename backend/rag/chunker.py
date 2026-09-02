"""
Text chunking for retrieval.

Splits long documents into overlapping word-based chunks so each chunk
fits a bounded embedding window while keeping neighbouring context via
a configurable overlap.
"""

from __future__ import annotations

from collections.abc import Iterable

from .config import settings
from .documents import Chunk, Document


class TextChunker:
    """
    Deterministic word-based chunker with sliding-window overlap.

    Parameters
    ----------
    chunk_size : int | None
        Maximum number of words per chunk. Defaults to
        ``settings.CHUNK_SIZE``.
    overlap : int | None
        Number of words shared between consecutive chunks. Defaults to
        ``settings.CHUNK_OVERLAP``.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self._chunk_size = (
            settings.CHUNK_SIZE if chunk_size is None else int(chunk_size)
        )
        self._overlap = settings.CHUNK_OVERLAP if overlap is None else int(overlap)

        if self._chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        if self._overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if self._overlap >= self._chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into overlapping chunks.

        Parameters
        ----------
        document : Document
            Document to chunk.

        Returns
        -------
        list[Chunk]
            Chunks in document order; a single short document yields one
            chunk.
        """

        words = document.text.split()
        if not words:
            return []

        step = self._chunk_size - self._overlap
        chunks: list[Chunk] = []
        for index, start in enumerate(range(0, len(words), step)):
            text = " ".join(words[start : start + self._chunk_size])
            chunks.append(
                Chunk(
                    id=f"{document.id}::{index}",
                    document_id=document.id,
                    text=text,
                    index=index,
                    source=document.source,
                    metadata=dict(document.metadata),
                )
            )
        return chunks

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        """
        Chunk a collection of documents.

        Parameters
        ----------
        documents : Iterable[Document]
            Documents to chunk.

        Returns
        -------
        list[Chunk]
            All chunks across documents, in document order.
        """

        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk(document))
        return chunks


__all__ = ["TextChunker"]
