"""
Bundled medical corpus loading for RAG.

The repository ships a small, curated medical knowledge corpus under
``backend/rag/corpus/`` covering the four supported conditions (diabetes,
hypertension / heart disease, chronic kidney disease, sepsis) plus
general clinical topics (laboratory values, obesity, coronary heart
disease). This module discovers ``.txt`` / ``.md`` documents and loads
them into :class:`Document` objects with source metadata so the API can
serve real evidence out of the box without external data.
"""

from __future__ import annotations

from pathlib import Path

from .documents import Document

#: Directory containing the bundled knowledge documents.
BUNDLED_CORPUS_DIR = Path(__file__).parent / "corpus"

#: File extensions treated as knowledge documents.
DOCUMENT_SUFFIXES = {".txt", ".md"}


def load_documents(directory: Path | str) -> list[Document]:
    """
    Load every knowledge document under a directory into documents.

    Files are discovered recursively and sorted by path for determinism.
    Each document carries the source label derived from its file name.

    Parameters
    ----------
    directory : Path | str
        Directory to scan for ``.txt`` / ``.md`` files.

    Returns
    -------
    list[Document]
        Loaded documents, ordered by path.

    Raises
    ------
    NotADirectoryError
        If the given path is not a directory.
    """

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus directory not found: {root}")
    documents = [
        Document(
            id=path.stem,
            text=path.read_text(encoding="utf-8"),
            source=path.name,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    ]
    return documents


def load_bundled_corpus() -> list[Document]:
    """
    Load the repository's bundled medical corpus.

    Returns
    -------
    list[Document]
        Documents from ``backend/rag/corpus/``; an empty list when the
        corpus directory is missing (should not happen in a checkout).
    """

    if not BUNDLED_CORPUS_DIR.is_dir():
        return []
    return load_documents(BUNDLED_CORPUS_DIR)


__all__ = [
    "BUNDLED_CORPUS_DIR",
    "DOCUMENT_SUFFIXES",
    "load_bundled_corpus",
    "load_documents",
]
