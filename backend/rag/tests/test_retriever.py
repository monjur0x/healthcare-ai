"""
Tests for the RAG retriever.
"""

from __future__ import annotations

import pytest

from rag import HashingEmbedder, Retriever, TfidfEmbedder
from rag.documents import Chunk, Document
from rag.exceptions import EmptyCorpusError, EmptyQueryError


def make_chunks() -> list[Chunk]:
    return [
        Chunk(id="d1::0", document_id="d1", text="diabetes insulin glucose", index=0),
        Chunk(
            id="d2::0",
            document_id="d2",
            text="heart failure ejection fraction",
            index=0,
        ),
        Chunk(
            id="d3::0", document_id="d3", text="pneumonia cough fever lungs", index=0
        ),
    ]


def test_retrieves_relevant_chunk_first() -> None:
    retriever = Retriever(embedder=TfidfEmbedder(max_features=50))
    retriever.ingest(make_chunks())
    results = retriever.retrieve("diabetes insulin", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.document_id == "d1"
    assert results[0].score >= results[1].score


def test_build_context_labels_documents() -> None:
    retriever = Retriever(embedder=TfidfEmbedder(max_features=50))
    retriever.ingest(make_chunks())
    context = retriever.build_context("heart failure", top_k=1)
    assert "d2" in context
    assert "heart failure ejection fraction" in context


def test_empty_query_raises() -> None:
    retriever = Retriever(embedder=HashingEmbedder(dims=16))
    retriever.ingest(make_chunks())
    with pytest.raises(EmptyQueryError):
        retriever.retrieve("   ")


def test_retrieve_before_ingest_raises() -> None:
    retriever = Retriever(embedder=HashingEmbedder(dims=16))
    with pytest.raises(EmptyCorpusError):
        retriever.retrieve("diabetes")


def test_ingest_is_incremental() -> None:
    retriever = Retriever(embedder=HashingEmbedder(dims=16))
    retriever.ingest(make_chunks()[:1])
    retriever.ingest(make_chunks()[1:])
    assert retriever.n_chunks == 3
    assert len(retriever.retrieve("diabetes")) == 3


def test_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        Retriever(embedder=HashingEmbedder(dims=8), top_k=0)


def test_retrieves_from_documents_pipeline_style() -> None:
    from rag import RAGPipeline

    pipeline = RAGPipeline(chunker=None, embedder=HashingEmbedder(dims=16))
    pipeline.ingest_documents(
        [Document(id="d1", text="asthma wheezing bronchodilator", source="hosp")]
    )
    results = pipeline.retrieve("asthma wheezing")
    assert results[0].chunk.document_id == "d1"
