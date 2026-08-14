"""
Custom exceptions used by the RAG (retrieval) module.
"""


class RAGError(Exception):
    """
    Base retrieval exception.
    """


class EmptyCorpusError(RAGError):
    """
    Raised when retrieval is attempted before any document is ingested.
    """


class EmptyQueryError(RAGError):
    """
    Raised when a retrieval query is empty or whitespace-only.
    """


class InvalidDocumentError(RAGError):
    """
    Raised when a document is malformed (missing text or empty id).
    """


class EmbeddingError(RAGError):
    """
    Raised when embeddings cannot be produced for the input texts.
    """


class RetrievalError(RAGError):
    """
    Raised when a retrieval operation fails.
    """


__all__ = [
    "EmbeddingError",
    "EmptyCorpusError",
    "EmptyQueryError",
    "InvalidDocumentError",
    "RAGError",
    "RetrievalError",
]
