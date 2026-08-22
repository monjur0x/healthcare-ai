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
        "relevant_docs": ["chronic-kidney-disease", "anemia"],
        "category": "kidney",
    },
    {
        "query": "Sepsis recognition and early management",
        "expected_topic": "sepsis recognition",
        "relevant_docs": ["sepsis", "sepsis-icu-synthetic"],
        "category": "sepsis",
    },
    {
        "query": "Septic shock fluid resuscitation protocol",
        "expected_topic": "septic shock treatment",
        "relevant_docs": ["sepsis", "sepsis-icu-synthetic"],
        "category": "sepsis",
    },
    {
        "query": "Diabetic ketoacidosis emergency management",
        "expected_topic": "diabetic emergency",
        "relevant_docs": ["diabetes-mellitus", "diabetic-ketoacidosis"],
        "category": "diabetes",
    },
    {
        "query": "Acute coronary syndrome immediate management",
        "expected_topic": "ACS treatment",
        "relevant_docs": ["coronary-heart-disease", "acute-coronary-syndrome"],
        "category": "heart",
    },
    {
        "query": "Heart failure with reduced ejection fraction treatment",
        "expected_topic": "HFrEF treatment",
        "relevant_docs": ["heart-failure", "coronary-heart-disease"],
        "category": "heart",
    },
    {
        "query": "Contrast-induced nephropathy prevention",
        "expected_topic": "kidney injury prevention",
        "relevant_docs": ["chronic-kidney-disease", "contrast-nephropathy"],
        "category": "kidney",
    },
    {
        "query": "Sepsis-induced AKI management",
        "expected_topic": "sepsis AKI",
        "relevant_docs": ["sepsis", "acute-kidney-injury"],
        "category": "sepsis",
    },
    {
        "query": "Diabetic foot ulcer prevention and management",
        "expected_topic": "diabetic foot care",
        "relevant_docs": ["diabetes-mellitus", "diabetic-foot"],
        "category": "diabetes",
    },
    {
        "query": "Heart failure with preserved ejection fraction treatment",
        "expected_topic": "HFpEF treatment",
        "relevant_docs": ["heart-failure", "coronary-heart-disease"],
        "category": "heart",
    },
    {
        "query": "Renal replacement therapy indications in AKI",
        "expected_topic": "AKI dialysis indications",
        "relevant_docs": ["acute-kidney-injury", "renal-replacement-therapy"],
        "category": "kidney",
    },
]


def run_evaluation(
    eval_file: str = "backend/rag/evaluation_set.json",
    output_path: str = "artifacts/experiments/rag_evaluation.json",
):
    """Run the full RAG evaluation pipeline."""
    import logging

    from pathlib import Path

    # Add backend to path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize RAG pipeline
    logger.info("Initializing RAG pipeline...")
    corpus = load_bundled_corpus()
    pipeline = RAGPipeline()
    pipeline.ingest_documents(corpus)

    # Load queries
    from rag.evaluation_set import EVALUATION_QUERIES

    logger.info(f"Evaluating {len(EVALUATION_QUERIES)} queries...")

    all_precisions = {1: [], 3: [], 5: [], 10: []}
    all_recalls = {1: [], 3: [], 5: [], 10: []}
    all_mrrs = []

    # Initialize pipeline once
    corpus = load_bundled_corpus()
    pipeline = RAGPipeline()
    pipeline.ingest_documents(corpus)

    for q in EVALUATION_QUERIES:
        query = q["query"]
        relevant = set(q["relevant_docs"])

        # Retrieve
        results = pipeline.retrieve(q["query"], top_k=10)
        retrieved_ids = [r.chunk.document_id for r in results]

        # Compute metrics at different k
        for k in [1, 3, 5, 10]:
            retrieved_at_k = retrieved_ids[:k]
            p = sum(1 for d in retrieved_ids[:k] if d in relevant) / k if k > 0 else 0
            r = (
                len([d for d in retrieved_ids[:k] if d in relevant]) / len(relevant)
                if (relevant := set(q["relevant_docs"]))
                else 0
            )
            logger.info(f"  P@{k}: {p:.4f}, R@{k}: {r:.4f}")

        # MRR
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids, 1):
            if any(doc_id.startswith(rel) for rel in q["relevant_docs"]):
                mrr = 1.0 / i
                break
        logger.info(f"Query: {q['query']}")
        logger.info(f"  MRR: {mrr:.4f}")
        print()

    # Aggregate metrics
    metrics = {
        "note": "Full evaluation metrics available in logs above",
        "total_queries": len(EVALUATION_QUERIES),
    }

    # Save results
    Path("artifacts/experiments").mkdir(parents=True, exist_ok=True)
    with open("artifacts/experiments/rag_evaluation.json", "w") as f:
        json.dump(
            {
                "note": "Full evaluation metrics available in logs above",
                "total_queries": len(EVALUATION_QUERIES),
            },
            f,
            indent=2,
        )

    logger.info("Results saved to artifacts/experiments/rag_evaluation.json")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    run_evaluation()
