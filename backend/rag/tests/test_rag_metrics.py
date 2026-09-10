"""
Tests for the RAGAS-style generation metrics (paper Section 12).

Uses hermetic TF-IDF / hashing embedders plus a deterministic fake
embedder so faithfulness and answer relevancy are tested against
clear-cut synthetic cases (verbatim-chunk answers score high; unrelated
answers score low).
"""

from __future__ import annotations

import pytest

from rag import (
    HashingEmbedder,
    RAGQualityMetrics,
    TfidfEmbedder,
    rag_quality_metrics,
)
from rag.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

CHUNK_IDS = ["c1", "c2", "c3"]
CHUNK_TEXTS = [
    "diabetes is managed with metformin and lifestyle changes",
    "hypertension is treated with blood pressure lowering drugs",
    "sepsis requires broad-spectrum antibiotics within one hour",
]


def test_context_precision() -> None:
    assert context_precision(["c1", "c2", "c3"], {"c1", "c3"}) == 2 / 3
    assert context_precision(["c1", "c2"], {"c3"}) == 0.0
    assert context_precision([], {"c1"}) == 0.0
    assert context_precision(["c1", "c2", "c3"], set()) == 0.0


def test_context_precision_accepts_id_text_tuples() -> None:
    chunks = [(id_, text) for id_, text in zip(CHUNK_IDS, CHUNK_TEXTS, strict=True)]
    assert context_precision(chunks, {"c2"}) == 1 / 3


def test_context_recall() -> None:
    assert context_recall(["c1", "c2"], {"c1", "c2", "c3"}) == 2 / 3
    assert context_recall(["c1"], {"c1", "c2"}) == 0.5
    assert context_recall(["c1"], set()) == 0.0
    assert context_recall(["c1"], {"c2"}) == 0.0


def test_context_recall_accepts_id_text_tuples() -> None:
    chunks = [(id_, text) for id_, text in zip(CHUNK_IDS, CHUNK_TEXTS, strict=True)]
    assert context_recall(chunks, {"c1", "c3"}) == 1.0
    assert context_recall(chunks, {"c1", "c2", "c3"}) == 1.0
    assert context_recall(chunks, {"c4"}) == 0.0


def test_faithfulness_verbatim_answer_scores_high() -> None:
    embedder = TfidfEmbedder(max_features=100).fit(CHUNK_TEXTS)
    answer = CHUNK_TEXTS[0]
    score = faithfulness(answer, CHUNK_TEXTS, embedder)
    assert score >= 0.9


def test_faithfulness_unrelated_answer_scores_low() -> None:
    corpus = [*CHUNK_TEXTS, "the moon is made of cheese"]
    embedder = TfidfEmbedder(max_features=100).fit(corpus)
    score = faithfulness("the moon is made of cheese", CHUNK_TEXTS, embedder)
    assert score <= 0.4


def test_faithfulness_empty_answer_scores_zero() -> None:
    embedder = TfidfEmbedder(max_features=100).fit(CHUNK_TEXTS)
    assert faithfulness("", CHUNK_TEXTS, embedder) == 0.0


def test_faithfulness_no_context_scores_zero() -> None:
    embedder = TfidfEmbedder(max_features=100).fit(CHUNK_TEXTS)
    assert faithfulness(CHUNK_TEXTS[0], [], embedder) == 0.0


def test_faithfulness_accepts_id_text_tuples() -> None:
    embedder = TfidfEmbedder(max_features=100).fit(CHUNK_TEXTS)
    chunks = [(id_, text) for id_, text in zip(CHUNK_IDS, CHUNK_TEXTS, strict=True)]
    score = faithfulness(CHUNK_TEXTS[0], chunks, embedder)
    assert score >= 0.9


def test_answer_relevancy_same_text_scores_high() -> None:
    embedder = HashingEmbedder(dims=256)
    assert answer_relevancy("metformin", "metformin", embedder) >= 0.9


def test_answer_relevancy_empty_inputs_score_zero() -> None:
    embedder = HashingEmbedder(dims=256)
    assert answer_relevancy("", "query", embedder) == 0.0
    assert answer_relevancy("answer", "  ", embedder) == 0.0


def test_answer_relevancy_clamped_to_non_negative() -> None:
    embedder = HashingEmbedder(dims=256)
    score = answer_relevancy("totally unrelated topic", "diabetes care", embedder)
    assert 0.0 <= score <= 1.0


def test_rag_quality_metrics_aggregate() -> None:
    embedder = TfidfEmbedder(max_features=100).fit(CHUNK_TEXTS)
    chunks = [(id_, text) for id_, text in zip(CHUNK_IDS, CHUNK_TEXTS, strict=True)]
    metrics = rag_quality_metrics(
        query="diabetes management",
        answer=CHUNK_TEXTS[0],
        retrieved_chunks=chunks,
        relevant_chunk_ids={"c1", "c3"},
        embedder=embedder,
    )
    assert isinstance(metrics, RAGQualityMetrics)
    assert metrics.context_precision == pytest.approx(2 / 3)
    assert metrics.context_recall == pytest.approx(1.0)
    assert metrics.faithfulness >= 0.9
    assert 0.0 <= metrics.answer_relevancy <= 1.0
    payload = metrics.to_dict()
    assert set(payload) == {
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    }
