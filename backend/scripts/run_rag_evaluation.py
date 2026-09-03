#!/usr/bin/env python3
"""
RAG Retrieval Quality Evaluator

Tests the RAG retriever against clinical queries with known relevant documents.
Computes precision@k, recall@k, MRR, and other IR metrics.
"""

from __future__ import annotations

import json
import logging
import sys

from dataclasses import dataclass
from pathlib import Path

# Add backend to path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import RAGPipeline
from rag.corpus import load_bundled_corpus
from rag.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k

logger = logging.getLogger(__name__)


@dataclass
class QueryEval:
    """A clinical query with expected relevant documents."""

    query: str
    expected_topic: str
    relevant_docs: list[str]  # Document IDs from the corpus
    category: str


# Clinical evaluation queries with ground truth relevant documents
EVALUATION_QUERIES = [
    {
        "query": "Risk factors for diabetic kidney disease",
        "expected_topic": "diabetes complications",
        "relevant_docs": ["diabetes-mellitus", "chronic-kidney-disease"],
        "category": "diabetes",
    },
    {
        "query": "Metformin dosage for type 2 diabetes",
        "expected_topic": "diabetes treatment",
        "relevant_docs": ["diabetes-mellitus"],
        "category": "diabetes",
    },
    {
        "query": "HbA1c targets for diabetes management",
        "expected_topic": "diabetes monitoring",
        "relevant_docs": ["diabetes-mellitus", "clinical-laboratory-values"],
        "category": "diabetes",
    },
    {
        "query": "Heart failure treatment guidelines",
        "expected_topic": "heart failure management",
        "relevant_docs": ["coronary-heart-disease", "hypertension"],
        "category": "heart",
    },
    {
        "query": "Atrial fibrillation anticoagulation guidelines",
        "expected_topic": "atrial fibrillation treatment",
        "relevant_docs": ["coronary-heart-disease", "hypertension"],
        "category": "heart",
    },
    {
        "query": "Hypertension management in elderly patients",
        "expected_topic": "hypertension treatment",
        "relevant_docs": ["hypertension", "coronary-heart-disease"],
        "category": "heart",
    },
    {
        "query": "CKD staging criteria based on GFR",
        "expected_topic": "CKD staging",
        "relevant_docs": ["chronic-kidney-disease", "clinical-laboratory-values"],
        "category": "kidney",
    },
    {
        "query": "Anemia management in chronic kidney disease",
        "expected_topic": "CKD complications",
        "relevant_docs": ["chronic-kidney-disease", "kdigo-ckd-2024"],
        "category": "kidney",
    },
    {
        "query": "Sepsis recognition and early management",
        "expected_topic": "sepsis recognition",
        "relevant_docs": ["sepsis", "surviving-sepsis-campaign-2021"],
        "category": "sepsis",
    },
    {
        "query": "Septic shock fluid resuscitation protocol",
        "expected_topic": "septic shock treatment",
        "relevant_docs": ["sepsis", "surviving-sepsis-campaign-2021"],
        "category": "sepsis",
    },
    {
        "query": "Diabetic ketoacidosis emergency management",
        "expected_topic": "diabetic emergency",
        "relevant_docs": ["diabetes-mellitus", "hospital-protocol-diabetes-management"],
        "category": "diabetes",
    },
    {
        "query": "Acute coronary syndrome immediate management",
        "expected_topic": "ACS treatment",
        "relevant_docs": ["coronary-heart-disease", "key-clinical-trials-evidence"],
        "category": "heart",
    },
    {
        "query": "Heart failure with reduced ejection fraction treatment",
        "expected_topic": "HFrEF treatment",
        "relevant_docs": ["acc-aha-heart-failure-2022", "coronary-heart-disease"],
        "category": "heart",
    },
    {
        "query": "Contrast-induced nephropathy prevention",
        "expected_topic": "kidney injury prevention",
        "relevant_docs": ["chronic-kidney-disease", "nice-ckd-guideline"],
        "category": "kidney",
    },
    {
        "query": "Sepsis-induced AKI management",
        "expected_topic": "sepsis AKI",
        "relevant_docs": ["sepsis", "chronic-kidney-disease"],
        "category": "sepsis",
    },
    {
        "query": "Diabetic foot ulcer prevention and management",
        "expected_topic": "diabetic foot care",
        "relevant_docs": ["diabetes-mellitus", "ada-diabetes-standards-2024"],
        "category": "diabetes",
    },
    {
        "query": "Heart failure with preserved ejection fraction treatment",
        "expected_topic": "HFpEF treatment",
        "relevant_docs": ["acc-aha-heart-failure-2022", "coronary-heart-disease"],
        "category": "heart",
    },
    {
        "query": "Renal replacement therapy indications in AKI",
        "expected_topic": "AKI dialysis indications",
        "relevant_docs": ["kdigo-ckd-2024", "surviving-sepsis-campaign-2021"],
        "category": "kidney",
    },
]


def run_evaluation(
    output_path: str | None = None,
) -> dict:
    """Run the full RAG evaluation pipeline.

    Retrieves for every evaluation query, computes P@k / R@k / MRR,
    aggregates the means over all queries, and persists them to
    ``output_path``.

    A relative ``output_path`` resolves against the backend directory
    (not the caller CWD) so runs from the repo root land next to the
    other experiment artifacts.

    Returns
    -------
    dict
        The aggregate metrics payload that was persisted.
    """
    backend_dir = Path(__file__).resolve().parent.parent
    if output_path is None:
        target = backend_dir / "artifacts" / "experiments" / "rag_evaluation.json"
    else:
        target = Path(output_path)
        if not target.is_absolute():
            target = backend_dir / target
    logger.info("Initializing RAG pipeline...")
    corpus = load_bundled_corpus()
    pipeline = RAGPipeline()
    pipeline.ingest_documents(corpus)

    logger.info("Evaluating %d queries...", len(EVALUATION_QUERIES))

    all_precisions: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
    all_recalls: dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
    all_mrrs: list[float] = []

    for q in EVALUATION_QUERIES:
        relevant = set(q["relevant_docs"])
        results = pipeline.retrieve(q["query"], top_k=10)
        retrieved_ids = [r.chunk.document_id for r in results]

        # Precision / recall at the standard k cut-offs (shared helpers,
        # exact-match — same relevance rule as MRR below).
        for k in (1, 3, 5, 10):
            precision = precision_at_k(relevant, retrieved_ids, k)
            recall = recall_at_k(relevant, retrieved_ids, k)
            all_precisions[k].append(precision)
            all_recalls[k].append(recall)
            logger.info("  P@%d: %.4f, R@%d: %.4f", k, precision, k, recall)

        mrr = mean_reciprocal_rank(relevant, retrieved_ids)
        all_mrrs.append(mrr)
        logger.info("Query: %s", q["query"])
        logger.info("  MRR: %.4f", mrr)

    total = len(EVALUATION_QUERIES)
    metrics = {
        "note": "Mean precision@k / recall@k / MRR over the evaluation queries",
        "total_queries": total,
        "precision_at_k": {
            str(k): sum(values) / total for k, values in all_precisions.items()
        },
        "recall_at_k": {
            str(k): sum(values) / total for k, values in all_recalls.items()
        },
        "mrr": sum(all_mrrs) / total if total else 0.0,
    }

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Results saved to %s", target)
    return metrics


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    run_evaluation()
