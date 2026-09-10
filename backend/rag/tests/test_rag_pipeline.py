"""
End-to-end tests for the RAG pipeline.
"""

from __future__ import annotations

import pytest

from rag import RAGPipeline
from rag.documents import Document
from rag.exceptions import EmptyQueryError

CORPUS = [
    Document(
        id="hypertension",
        text="chronic hypertension raises stroke risk",
        source="guideline",
    ),
    Document(
        id="sepsis",
        text="sepsis is organ dysfunction from infection",
        source="protocol",
    ),
    Document(
        id="diabetes",
        text="type 2 diabetes managed with metformin and diet",
        source="protocol",
    ),
]


def test_ingest_and_query_roundtrip() -> None:
    pipeline = RAGPipeline()
    assert pipeline.ingest_documents(CORPUS) == 3
    assert pipeline.n_chunks == 3

    results = pipeline.retrieve("metformin for diabetes", top_k=1)
    assert results[0].chunk.document_id == "diabetes"


def test_build_context_contains_source() -> None:
    pipeline = RAGPipeline()
    pipeline.ingest_documents(CORPUS)
    context = pipeline.build_context("organ dysfunction infection", top_k=1)
    assert "sepsis" in context
    assert "protocol" in context


def test_ingest_texts_anonymous() -> None:
    pipeline = RAGPipeline()
    pipeline.ingest_texts(["tumor on the lung", "benign lymph nodes"])
    assert pipeline.n_chunks == 2
    assert len(pipeline.retrieve("lung tumor")) == 2


def test_empty_query_raises() -> None:
    pipeline = RAGPipeline()
    pipeline.ingest_documents(CORPUS)
    with pytest.raises(EmptyQueryError):
        pipeline.build_context(" ")


def test_retrieve_before_ingest_raises() -> None:
    from rag.exceptions import EmptyCorpusError

    with pytest.raises(EmptyCorpusError):
        RAGPipeline().retrieve("anything")


def test_long_documents_are_chunked_and_retrievable() -> None:
    long_text = " ".join(f"glucose level reading number {i}" for i in range(200))
    pipeline = RAGPipeline()
    pipeline.ingest_documents([Document(id="glucose-log", text=long_text)])
    assert pipeline.n_chunks > 1

    results = pipeline.retrieve("glucose reading 99", top_k=1)
    assert results[0].chunk.document_id == "glucose-log"
