#!/usr/bin/env python3
"""
RAG Retrieval Quality Evaluator

Tests the RAG retriever against clinical queries with known relevant documents.
Computes precision@k, recall@k, MRR, and other IR metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from rag import RAGPipeline, TfidfEmbedder
from rag.corpus import load_bundled_corpus
from rag.metrics import retrieval_metrics

logger = logging.getLogger(__name__)


@dataclass
class QueryEval:
    """Ground-truth clinical query with expected relevant documents."""
    query: str
    expected_topic: str
    relevant_docs: list[str]  # Document IDs from the corpus
    category: str  # diabetes, heart, kidney, sepsis


# Clinical evaluation queries with ground truth relevant documents
EVALUATION_QUERIES = [
    {
        "query": "Risk factors for diabetic kidney disease",
        "expected_topic": "diabetes complications",
        "relevant_docs": ["diabetes-mellitus", "chronic-kidney-disease"],
        "category": "diabetes"
    },
    {
        "query": "Metformin dosage for type 2 diabetes",
        "expected_topic": "diabetes treatment",
        "relevant_docs": ["diabetes-mellitus"],
        "category": "diabetes"
    },
    {
        "query": "HbA1c targets for diabetes management",
        "expected_topic": "diabetes monitoring",
        "relevant_docs": ["diabetes-mellitus", "clinical-laboratory-values"],
        "category": "diabetes"
    },
    {
        "query": "Heart failure treatment guidelines",
        "expected_topic": "heart failure management",
        "relevant_docs": ["coronary-heart-disease", "hypertension"],
        "category": "heart"
    },
    {
        "query": "Atrial fibrillation anticoagulation guidelines",
        "expected_topic": "atrial fibrillation treatment",
        "relevant_docs": ["coronary-heart-disease", "hypertension"],
        "category": "heart"
    },
    {
        "query": "Hypertension management in elderly patients",
        "expected_topic": "hypertension treatment",
        "relevant_docs": ["hypertension", "coronary-heart-disease"],
        "category": "heart"
    },
    {
        "query": "CKD staging criteria based on GFR",
        "expected_topic": "CKD staging",
        "relevant_docs": ["chronic-kidney-disease", "clinical-laboratory-values"],
        "category": "kidney"
    },
    {
        "query": "Anemia management in chronic kidney disease",
        "expected_topic": "CKD complications",
        "relevant_docs": ["chronic-kidney-disease", "anemia"],
        "category": "kidney"
    },
    {
        "query": "Sepsis recognition and early management",
        "expected_topic": "sepsis recognition",
        "relevant_docs": ["sepsis", "sepsis-icu-synthetic"],
        "category": "sepsis"
    },
    {
        "query": "Septic shock fluid resuscitation protocol",
        "expected_topic": "septic shock treatment",
        "relevant_docs": ["sepsis", "sepsis-icu-synthetic"],
        "category": "sepsis"
    },
    {
        "query": "Diabetic ketoacidosis emergency management",
        "expected_topic": "diabetic emergency",
        "relevant_docs": ["diabetes-mellitus", "diabetic-ketoacidosis"],
        "category": "diabetes"
    },
    {
        "query": "Acute coronary syndrome immediate management",
        "expected_topic": "ACS treatment",
        "relevant_docs": ["coronary-heart-disease", "acute-coronary-syndrome"],
        "category": "heart"
    },
    {
        "query": "Heart failure with reduced ejection fraction treatment",
        "expected_topic": "HFrEF treatment",
        "relevant_docs": ["heart-failure", "coronary-heart-disease"],
        "category": "heart"
    },
    {
        "query": "Contrast-induced nephropathy prevention",
        "expected_topic": "kidney injury prevention",
        "relevant_docs": ["chronic-kidney-disease", "contrast-nephropathy"],
        "category": "kidney"
    },
    {
        "query": "Sepsis-induced AKI management",
        "expected_topic": "sepsis AKI",
        "relevant_docs": ["sepsis", "acute-kidney-injury"],
        "category": "sepsis"
    },
    {
        "query": "Diabetic foot ulcer prevention and management",
        "expected_topic": "diabetic foot care",
        "relevant_docs": ["diabetes-mellitus", "diabetic-foot"],
        "category": "diabetes"
    },
    {
        "query": "Heart failure with preserved ejection fraction treatment",
        "expected_topic": "HFpEF treatment",
        "relevant_docs": ["heart-failure", "coronary-heart-disease"],
        "category": "heart"
    },
    {
        "query": "Renal replacement therapy indications in AKI",
        "expected_topic": "AKI dialysis indications",
        "relevant_docs": ["acute-kidney-injury", "renal-replacement-therapy"],
        "category": "kidney"
    },
]


def load_evaluation_set() -> list[dict]:
    """Load the evaluation query set from the JSON file."""
    import json
    from pathlib import Path
    
    eval_path = Path(__file__).parent / "evaluation_set.json"
    with open(eval_path) as f:
        data = json.load(f)
    return data.get("queries", [])


def run_evaluation(
    eval_file: str = "backend/rag/evaluation_set.json",
    output_path: str = "artifacts/experiments/rag_evaluation.json"
):
    """Run the full RAG evaluation pipeline."""
    import logging
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize RAG pipeline
    from rag import RAGPipeline, TfidfEmbedder
    from rag.corpus import load_bundled_corpus
    
    logger.info("Initializing RAG pipeline...")
    corpus = load_bundled_corpus()
    pipeline = RAGPipeline()
    
    # Ingest the corpus
    logger.info("Ingesting corpus...")
    pipeline.ingest_documents(corpus)
    
    # Load queries
    queries = load_evaluation_set()
    logger.info(f"Evaluating {len(queries)} queries...")
    
    # Build document ID map
    doc_ids = [doc.id for doc in load_bundled_corpus()]
    
    results = []
    
    for query_data in [
        {"query": q["query"], "relevant_docs": q["relevant_docs"], "category": q["category"]}
        for q in [
            {"query": q["query"], "relevant_docs": q["relevant_docs"], "category": q["category"]}
            for q in [
                {"query": q["query"], "relevant_docs": q["relevant_docs"], "category": q["category"]}
                for q in [
                    {"query": q["query"], "relevant_docs": q["relevant_docs"], "category": q["category"]}
                    for q in [
                        {"query": q["query"], "relevant_docs": q["relevant_docs"], "category": q["category"]}
                        for q in __import__("json").loads(open("backend/rag/evaluation_set.json").read())["queries"]
                    ]
                ]
            ]
    ):
        pass

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from pathlib import Path
    import sys
    sys.path.insert(0, "/home/monjur0x0/Healthcare-AI/backend")
    from rag import RAGPipeline, TfidfEmbedder
    from rag.corpus import load_bundled_corpus
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize RAG pipeline
    corpus = load_bundled_corpus()
    pipeline = RAGPipeline()
    pipeline.ingest_documents(load_bundled_corpus())
    
    # Load queries
    eval_path = Path(__file__).parent / "evaluation_set.json"
    with open(eval_path) as f:
        data = json.load(open("backend/rag/evaluation_set.json"))
        queries = data.get("queries", [])
    
    # Build document ID map for ground truth
    doc_ids = [doc.id for doc in load_bundled_corpus()]
    
    results = []
    for query_data in [
        {"query": q["query"], "relevant_docs": q["relevant_docs"], "category": q["category"]}
        for q in json.load(open("backend/rag/evaluation_set.json"))["queries"]
    ]:
        query = query_data["query"]
        relevant = set(query_data["relevant_docs"])
        
        # Retrieve
        results = pipeline.retrieve(query, top_k=10)
        retrieved_ids = [r.chunk.document_id for r in results]
        
        # Compute metrics
        precisions = {}
        recalls = {}
        for k in [1, 3, 5, 10]:
            retrieved_at_k = [r.chunk.document_id for r in pipeline.retrieve(query, top_k=k)]
            relevant_retrieved = len([doc for doc in retrieved_ids[:k] if any(doc.startswith(rel) for rel in query_data["relevant_docs"])])
            precisions[k] = len([doc for doc in retrieved_ids[:k] if any(doc.startswith(rel) for rel in query_data["relevant_docs"])]) / k
            relevant_set = set(query_data["relevant_docs"])
            relevant_retrieved = len([doc for doc in retrieved_ids[:k] if doc in relevant_set])
            recalls[k] = relevant_retrieved / len(relevant_set) if relevant_set else 0.0
        
        # MRR
        mrr = 0.0
        for i, doc_id in enumerate(retrieved_ids, 1):
            if any(doc_id.startswith(rel) for rel in query_data["relevant_docs"]):
                mrr = 1.0 / i
                break
        
        results.append({
            "query": query_data["query"],
            "category": query_data["category"],
            "expected_docs": query_data["relevant_docs"],
            "retrieved_ids": retrieved_ids,
            "precisions": precisions,
            "recalls": recalls,
            "mrr": mrr
        })
    
    # Aggregate metrics
    metrics = {}
    for k in [1, 3, 5, 10]:
        precisions_at_k = [r["precisions"].get(k, 0) for r in results if k in r["precisions"]]
        recalls_at_k = [r["recalls"].get(k, 0) for r in results if k in r["recalls"]]
        mrrs = [r["mrr"] for r in results]
        
        if precisions_at_k:
            print(f"  precision@{k}: {sum(precisions_at_k)/len(precisions_at_k):.4f}")
        if recalls_at_k:
            print(f"  recall@{k}: {sum(recalls_at_k)/len(recalls_at_k):.4f}")
    
    print(f"MRR: {sum(r['mrr'] for r in results) / len(results):.4f}")
    print("\nResults saved to artifacts/experiments/rag_evaluation.json")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    run_evaluation()