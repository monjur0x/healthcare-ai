"""
Core data structures for the RAG (retrieval) module.

A corpus is a collection of :class:`Document` objects, each split into
overlapping :class:`Chunk` objects before embedding and storage.
Retrieval returns :class:`RetrievalResult` items pairing a chunk with
its similarity score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .exceptions import InvalidDocumentError


@dataclass(frozen=True)
class Document:
    """
    A single retrievable text document.

    Parameters
    ----------
    id : str
        Unique document identifier (e.g. the source file name).
    text : str
        Full document text.
    source : str
        Human-readable source label (e.g. ``"PubMed"``, ``"protocols"``).
    metadata : dict[str, Any]
        Arbitrary provenance metadata.
    """

    id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise InvalidDocumentError("Document id must be a non-empty string.")
        if not self.text:
            raise InvalidDocumentError("Document text must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the document to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Document fields keyed by name.
        """

        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Chunk:
    """
    A slice of a document ready for embedding.

    Parameters
    ----------
    id : str
        Unique chunk identifier (``{document_id}::{index}``).
    document_id : str
        Owning document identifier.
    text : str
        Chunk text.
    index : int
        Zero-based position within the document.
    source : str
        Source label inherited from the owning document.
    metadata : dict[str, Any]
        Provenance metadata inherited from the owning document (e.g.
        ``{"topics": ["diabetes"]}`` for topic-aware retrieval).
    """

    id: str
    document_id: str
    text: str
    index: int
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the chunk to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Chunk fields keyed by name.
        """

        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "index": self.index,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """
    A retrieved chunk paired with its similarity score.

    Parameters
    ----------
    chunk : Chunk
        The retrieved chunk.
    score : float
        Similarity score (higher is more relevant).
    """

    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the result to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Chunk data plus the similarity score.
        """

        return {"score": self.score, **self.chunk.to_dict()}


__all__ = ["Chunk", "Document", "RetrievalResult"]
