"""
Retrieval and RAG generation quality metrics.

IR metrics (precision at k, recall at k, mean reciprocal rank) score a
ranked retrieval list against ground-truth relevant documents. The
RAGAS-style generation metrics (context precision, context recall,
faithfulness, answer relevancy) score how well a generated answer is
grounded in retrieved context and aligned with the query. All are pure
helpers over the module's :class:`~rag.embedder.Embedder` interface, so
they work with whichever embedder is configured (TF-IDF, hashing, or
sentence-transformers) and need no LLM judge.
"""

from __future__ import annotations

import re

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .embedder import Embedder


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


def _split_sentences(text: str) -> list[str]:
    """Split a text into trimmed sentences, dropping empties."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors in ``[-1, 1]``."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


#: Minimum best cosine similarity for a sentence to count as grounded.
# Default grounded-similarity cutoff; overridden by
# ``settings.FAITHFULNESS_THRESHOLD`` at call time.
_FAITHFULNESS_THRESHOLD_DEFAULT = 0.5


def context_precision(
    retrieved_chunks: Sequence[str],
    relevant_chunk_ids: Collection[str],
) -> float:
    """
    Fraction of the retrieved context that is actually relevant.

    RAGAS context precision asks how much of what was retrieved was
    useful. Here the retrieved chunks are supplied as ``(id, text)``
    tuples (see :class:`RetrievalResult`-style pairs) and scored against
    the ground-truth relevant chunk ids.

    Parameters
    ----------
    retrieved_chunks : Sequence[str]
        Chunk ids in retrieval order (or ``(id, text)`` tuples — the id
        is taken from the first element).
    relevant_chunk_ids : Collection[str]
        Ground-truth relevant chunk ids.

    Returns
    -------
    float
        Precision in ``[0, 1]``; ``0.0`` for an empty retrieved list.
    """

    relevant = set(relevant_chunk_ids)
    ids = [
        chunk[0] if isinstance(chunk, tuple) else chunk for chunk in retrieved_chunks
    ]
    if not ids:
        return 0.0
    return float(sum(1 for id_ in ids if id_ in relevant) / len(ids))


def context_recall(
    retrieved_chunks: Sequence[str],
    relevant_chunk_ids: Collection[str],
) -> float:
    """
    Fraction of all relevant chunks that were retrieved.

    Parameters
    ----------
    retrieved_chunks : Sequence[str]
        Chunk ids in retrieval order (or ``(id, text)`` tuples — the id
        is taken from the first element).
    relevant_chunk_ids : Collection[str]
        Ground-truth relevant chunk ids.

    Returns
    -------
    float
        Recall in ``[0, 1]``; ``0.0`` when nothing relevant is retrieved.
    """

    relevant = set(relevant_chunk_ids)
    ids = [
        chunk[0] if isinstance(chunk, tuple) else chunk for chunk in retrieved_chunks
    ]
    if not relevant:
        return 0.0
    return float(len(relevant & set(ids)) / len(relevant))


def faithfulness(
    answer: str,
    retrieved_chunks: Sequence[str],
    embedder: Embedder,
    threshold: float | None = None,
) -> float:
    """
    Fraction of the answer's sentences grounded in the retrieved context.

    A dependency-light, LLM-free proxy for RAGAS faithfulness: each
    sentence of the answer is embedded and scored against every retrieved
    chunk (embedded via the same :class:`Embedder`); a sentence is
    considered grounded when its best cosine similarity to some chunk
    reaches the configured threshold. The metric is the fraction of
    grounded sentences, so an answer copied verbatim from a chunk scores
    1.0 and an unrelated answer scores near 0.0.

    Parameters
    ----------
    answer : str
        The generated answer to grade.
    retrieved_chunks : Sequence[str]
        Retrieved context texts (or ``(id, text)`` tuples — the text is
        taken from the last element).
    embedder : Embedder
        Embedding model used for both sentences and chunks.

    Returns
    -------
    float
        Faithfulness in ``[0, 1]``; ``0.0`` for an empty answer.
    """

    from .config import settings

    effective_threshold = threshold
    if effective_threshold is None:
        effective_threshold = getattr(
            settings,
            "FAITHFULNESS_THRESHOLD",
            _FAITHFULNESS_THRESHOLD_DEFAULT,
        )
    texts = [
        chunk[-1] if isinstance(chunk, tuple) else chunk for chunk in retrieved_chunks
    ]
    sentences = _split_sentences(answer)
    if not sentences or not texts:
        return 0.0
    sentence_vectors = embedder.embed(sentences)
    chunk_vectors = embedder.embed(texts)
    grounded = 0
    for sentence in sentence_vectors:
        best = max(_cosine_similarity(sentence, chunk) for chunk in chunk_vectors)
        if best >= effective_threshold:
            grounded += 1
    return float(grounded / len(sentences))


def answer_relevancy(answer: str, query: str, embedder: Embedder) -> float:
    """
    Cosine similarity between the answer's and the query's embeddings.

    A dependency-light proxy for RAGAS answer relevancy: how aligned the
    generated answer is with the question. Uses the same :class:`Embedder`
    so it works with any configured embedding model.

    Parameters
    ----------
    answer : str
        The generated answer to grade.
    query : str
        The original user query.
    embedder : Embedder
        Embedding model used for both texts.

    Returns
    -------
    float
        Relevancy in ``[0, 1]`` (clamped from cosine ``[-1, 1]``).
    """

    if not answer.strip() or not query.strip():
        return 0.0
    vectors = embedder.embed([answer, query])
    return float(max(0.0, _cosine_similarity(vectors[0], vectors[1])))


@dataclass(frozen=True)
class RAGQualityMetrics:
    """
    RAGAS-style generation quality metrics for one query/answer pair.

    Parameters
    ----------
    context_precision : float
        Fraction of retrieved chunks that are relevant.
    context_recall : float
        Fraction of relevant chunks that were retrieved.
    faithfulness : float
        Fraction of answer sentences grounded in the retrieved context.
    answer_relevancy : float
        Cosine alignment between the answer and the query embeddings.
    """

    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the metrics to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Metrics keyed by name.
        """

        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
        }


def rag_quality_metrics(
    query: str,
    answer: str,
    retrieved_chunks: Sequence[str],
    relevant_chunk_ids: Collection[str],
    embedder: Embedder,
) -> RAGQualityMetrics:
    """
    Compute the RAGAS-style generation metrics for one query.

    Parameters
    ----------
    query : str
        The user query.
    answer : str
        The generated answer to grade.
    retrieved_chunks : Sequence[str]
        Retrieved chunks in rank order, either chunk ids or
        ``(id, text)`` tuples (ids feed context precision/recall, texts
        feed faithfulness).
    relevant_chunk_ids : Collection[str]
        Ground-truth relevant chunk ids.
    embedder : Embedder
        Embedding model used for the embedding-based metrics.

    Returns
    -------
    RAGQualityMetrics
        Aggregated generation quality metrics.
    """

    return RAGQualityMetrics(
        context_precision=context_precision(retrieved_chunks, relevant_chunk_ids),
        context_recall=context_recall(retrieved_chunks, relevant_chunk_ids),
        faithfulness=faithfulness(answer, retrieved_chunks, embedder),
        answer_relevancy=answer_relevancy(answer, query, embedder),
    )


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
    "RAGQualityMetrics",
    "RetrievalMetrics",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "faithfulness",
    "mean_reciprocal_rank",
    "precision_at_k",
    "rag_quality_metrics",
    "recall_at_k",
    "retrieval_metrics",
]
