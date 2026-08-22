#!/usr/bin/env python3
"""
RAG Evaluation Suite - Clinical Query Relevance Testing

Creates a ground-truth evaluation set for RAG retrieval quality testing.
Tests the retriever against clinical queries with known relevant documents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from pathlib import Path


@dataclass
class ClinicalQuery:
    """A clinical query with expected relevant documents."""
    query: str
    expected_topic: str
    relevant_docs: List[str]  # Document IDs from the corpus
    category: str  # diabetes, heart, kidney, sepsis, general


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
        "relevant_docs": ["heart-failure", "coronary-heart-disease"],
        "category": "heart"
    },
    {
        "query": "Atrial fibrillation anticoagulation guidelines",
        "expected_topic": "atrial fibrillation treatment",
        "relevant_docs": ["atrial-fibrillation", "coronary-heart-disease"],
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


def save_evaluation_set(output_path: str = "backend/rag/evaluation_set.json"):
    """Save the evaluation queries to a JSON file."""
    data = {
        "queries": [
            {
                "query": q["query"],
                "expected_topic": q["expected_topic"],
                "relevant_docs": q["relevant_docs"],
                "category": q["category"]
            }
            for q in EVALUATION_QUERIES
        ]
    }
    
    output_path = Path(__file__).parent.parent / output_path
    with open(output_path, 'w') as f:
        json.dump({"queries": EVALUATION_QUERIES}, f, indent=2)
    
    print(f"Saved {len(EVALUATION_QUERIES)} evaluation queries to {output_path}")


if __name__ == "__main__":
    from pathlib import Path
    save_evaluation_set()