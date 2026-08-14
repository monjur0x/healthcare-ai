"""
Retrieval quality metrics.

Pure helpers that score a ranked retrieval list against a set of
ground-truth relevant documents: precision at k, recall at k, and mean
reciprocal rank. Kept dependency-free so callers can compute RAG
metrics on any ranked output.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any


def precision_at_k(
    relevant_ids: Collection[str], retrieved_ids: Sequence[str], k: int | None = None
) -> float:
    """
    Fraction of the top-k retrieved ids that are relevant.

    Parameters
    ----------
    relevant_ids : Collection[str]
        Ground-truth relevant document ids.
    retrieved_ids : Sequence[str]
        Ranked retrieved document ids.
    k : int | None
        Cutoff; defaults to the full retrieved list.

    Returns
    -------
    float
        Precision at k in ``[0, 1]``.
    """

    relevant = set(relevant_ids)
    top = list(retrieved_ids[:k]) if k is not None else list(retrieved_ids)
    return float(sum(1 for id_ in top if id_ in relevant) / max(len(top), 1))


def recall_at_k(
    relevant_ids: Collection[str], retrieved_ids: Sequence[str], k: int | None = None
) -> float:
    """
    Fraction of all relevant ids found in the top-k results.

    Parameters
    ----------
    relevant_ids : Collection[str]
        Ground-truth relevant document ids.
    retrieved_ids : Sequence[str]
        Ranked retrieved document ids.
    k : int | None
        Cutoff; defaults to the full retrieved list.

    Returns
    -------
    float
        Recall at k in ``[0, 1]``.
    """

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = set(retrieved_ids[:k]) if k is not None else set(retrieved_ids)
    return float(len(relevant & top) / len(relevant))


def mean_reciprocal_rank(
    relevant_ids: Collection[str], retrieved_ids: Sequence[str]
) -> float:
    """
    Reciprocal rank of the first relevant hit (1.0 if first).

    Parameters
    ----------
    relevant_ids : Collection[str]
        Ground-truth relevant document ids.
    retrieved_ids : Sequence[str]
        Ranked retrieved document ids.

    Returns
    -------
    float
        MRR in ``[0, 1]``, ``0.0`` when nothing relevant was retrieved.
    """

    relevant = set(relevant_ids)
    for rank, id_ in enumerate(retrieved_ids, start=1):
        if id_ in relevant:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalMetrics:
    """
    Retrieval quality metrics for one query.

    Parameters
    ----------
    precision_at_k : float
        Precision at the k cutoff.
    recall_at_k : float
        Recall at the k cutoff.
    mrr : float
        Mean reciprocal rank.
    """

    precision_at_k: float
    recall_at_k: float
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the metrics to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Metrics keyed by name.
        """

        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
        }


def retrieval_metrics(
    relevant_ids: Collection[str],
    retrieved_ids: Sequence[str],
    k: int | None = None,
) -> RetrievalMetrics:
    """
    Compute the retrieval metrics for one query.

    Parameters
    ----------
    relevant_ids : Collection[str]
        Ground-truth relevant document ids.
    retrieved_ids : Sequence[str]
        Ranked retrieved document ids.
    k : int | None
        Cutoff for precision/recall; defaults to the full list.

    Returns
    -------
    RetrievalMetrics
        Aggregated precision, recall, and MRR.
    """

    return RetrievalMetrics(
        precision_at_k=precision_at_k(relevant_ids, retrieved_ids, k),
        recall_at_k=recall_at_k(relevant_ids, retrieved_ids, k),
        mrr=mean_reciprocal_rank(relevant_ids, retrieved_ids),
    )


__all__ = [
    "RetrievalMetrics",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "retrieval_metrics",
]
