"""
Tests for the RAG text chunker.
"""

from __future__ import annotations

import pytest

from rag import TextChunker
from rag.documents import Document
from rag.exceptions import InvalidDocumentError


def test_single_short_document_yields_one_chunk() -> None:
    chunker = TextChunker(chunk_size=10, overlap=2)
    chunks = chunker.chunk(Document(id="d1", text="a b c d"))
    assert len(chunks) == 1
    assert chunks[0].id == "d1::0"
    assert chunks[0].document_id == "d1"
    assert chunks[0].text == "a b c d"


def test_long_document_splits_with_overlap() -> None:
    chunker = TextChunker(chunk_size=4, overlap=1)
    text = " ".join(f"w{i}" for i in range(9))
    chunks = chunker.chunk(Document(id="d1", text=text))

    assert len(chunks) == 3
    assert chunks[0].text == "w0 w1 w2 w3"
    assert chunks[1].text == "w3 w4 w5 w6"
    assert chunks[2].text == "w6 w7 w8"


def test_chunk_ids_are_sequential() -> None:
    chunker = TextChunker(chunk_size=2, overlap=0)
    chunks = chunker.chunk(Document(id="doc", text="one two three four"))
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert [chunk.id for chunk in chunks] == ["doc::0", "doc::1"]


def test_chunk_documents_flattens() -> None:
    chunker = TextChunker(chunk_size=2, overlap=0)
    chunks = chunker.chunk_documents(
        [
            Document(id="a", text="one two three"),
            Document(id="b", text="four five"),
        ]
    )
    assert [chunk.document_id for chunk in chunks] == ["a", "a", "b"]


def test_empty_text_yields_no_chunks() -> None:
    chunker = TextChunker(chunk_size=10, overlap=0)
    assert chunker.chunk(Document(id="d", text="   ")) == []


def test_invalid_chunker_args() -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)
    with pytest.raises(ValueError):
        TextChunker(overlap=-1)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=5, overlap=5)


def test_document_requires_id_and_text() -> None:
    with pytest.raises(InvalidDocumentError):
        Document(id="", text="body")
    with pytest.raises(InvalidDocumentError):
        Document(id="d", text="")
