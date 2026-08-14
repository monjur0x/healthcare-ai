"""
Tests for RAG retrieval quality metrics.
"""

from __future__ import annotations

from rag import retrieval_metrics
from rag.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


def test_precision_at_k() -> None:
    assert precision_at_k({"a", "b"}, ["a", "x", "b"], k=2) == 0.5
    assert precision_at_k({"a", "b"}, ["a", "x", "b"], k=3) == 2 / 3
    assert precision_at_k(set(), ["a", "b"]) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k({"a", "b", "c"}, ["a", "x"], k=2) == 1 / 3
    assert recall_at_k({"a", "b", "c"}, ["a", "b"], k=2) == 2 / 3
    assert recall_at_k(set(), ["a", "b"]) == 0.0


def test_mean_reciprocal_rank() -> None:
    assert mean_reciprocal_rank({"b"}, ["x", "a", "b", "c"]) == 1 / 3
    assert mean_reciprocal_rank({"a"}, ["a", "b"]) == 1.0
    assert mean_reciprocal_rank({"z"}, ["a", "b"]) == 0.0


def test_retrieval_metrics_aggregate() -> None:
    metrics = retrieval_metrics({"a", "b"}, ["a", "x", "b"], k=2)
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 0.5
    assert metrics.mrr == 1.0
    assert metrics.to_dict() == {
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "mrr": 1.0,
    }
