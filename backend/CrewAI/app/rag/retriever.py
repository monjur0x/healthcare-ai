"""Medical knowledge retriever using RAG pipeline."""

from typing import Any
import logging
from .vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class MedicalRetriever:
    """Retrieves medical evidence from knowledge base."""

    # Medical knowledge sources
    SOURCES = {
        "WHO": "World Health Organization",
        "CDC": "Centers for Disease Control and Prevention",
        "NIH": "National Institutes of Health",
        "PubMed": "PubMed Medical Database",
        "Mayo Clinic": "Mayo Clinic Guidelines",
        "NHS": "National Health Service"
    }

    def __init__(self):
        """Initialize the retriever."""
        self.vector_store = QdrantVectorStore()
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Initialize with medical knowledge documents."""
        documents = [
            {
                "content": "Diabetes Management: Regular blood glucose monitoring, HbA1c testing every 3-6 months, maintaining target glucose levels, and medication adherence are crucial for diabetes management.",
                "source": "WHO",
                "reference": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
                "category": "diabetes"
            },
            {
                "content": "Hypertension Guidelines: Blood pressure target below 130/80 mmHg for most adults. Lifestyle modifications include salt restriction, regular exercise, weight management, and limiting alcohol.",
                "source": "CDC",
                "reference": "https://www.cdc.gov/bloodpressure",
                "category": "cardiovascular"
            },
            {
                "content": "Cardiovascular Risk Assessment: Use Framingham Risk Score or ASCVD Risk Calculator for 10-year cardiovascular risk assessment. Consider age, gender, cholesterol, blood pressure, diabetes, and smoking status.",
                "source": "NIH",
                "reference": "https://www.nhlbi.nih.gov",
                "category": "cardiovascular"
            },
            {
                "content": "Chronic Kidney Disease: Estimated GFR and albumin-to-creatinine ratio are key markers. Regular monitoring of creatinine and electrolytes recommended for at-risk patients.",
                "source": "PubMed",
                "reference": "https://pubmed.ncbi.nlm.nih.gov",
                "category": "renal"
            },
            {
                "content": "Obesity Management: BMI-based classification with targets for weight reduction. Recommended 5-10% weight loss for health benefits. Consider pharmacotherapy for BMI >30 or >27 with comorbidities.",
                "source": "Mayo Clinic",
                "reference": "https://www.mayoclinic.org",
                "category": "metabolic"
            },
            {
                "content": "Chest X-ray Interpretation: Systematic approach including assessment of airway, breathing, circulation, diaphragm, and everything else (ABCDE). Common findings include pneumonia, effusion, and cardiomegaly.",
                "source": "NHS",
                "reference": "https://www.nhs.uk",
                "category": "radiology"
            },
            {
                "content": "Anemia Diagnosis: Hemoglobin levels below 12 g/dL (women) or 13 g/dL (men) indicate anemia. Further testing includes iron studies, B12, folate, and reticulocyte count.",
                "source": "WHO",
                "reference": "https://www.who.int",
                "category": "hematology"
            },
            {
                "content": "Lipid Management: LDL cholesterol targets vary by risk category. High-intensity statins for ASCVD risk >7.5%. Lifestyle modifications include diet, exercise, and weight management.",
                "source": "NIH",
                "reference": "https://www.nhlbi.nih.gov",
                "category": "cardiovascular"
            }
        ]

        try:
            self.vector_store.add_documents(documents)
            logger.info("Initialized medical knowledge base")
        except Exception as e:
            logger.warning(f"Could not initialize knowledge base: {e}")

    def retrieve_evidence(self, query: str, num_results: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant medical evidence for a query."""
        try:
            results = self.vector_store.search(query, limit=num_results)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "source": result.get("source", "Unknown"),
                    "source_name": self.SOURCES.get(result.get("source", ""), result.get("source", "Unknown")),
                    "summary": result.get("content", ""),
                    "reference": result.get("reference", ""),
                    "relevance_score": round(result.get("score", 0), 3)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error retrieving evidence: {e}")
            return self._get_fallback_evidence(query)

    def _get_fallback_evidence(self, query: str) -> list[dict]:
        """Provide fallback evidence when vector store is unavailable."""
        return [
            {
                "source": "WHO",
                "source_name": "World Health Organization",
                "summary": "Consult WHO guidelines for comprehensive disease management recommendations.",
                "reference": "https://www.who.int/health-topics",
                "relevance_score": 0.75
            },
            {
                "source": "CDC",
                "source_name": "Centers for Disease Control",
                "summary": "Follow CDC preventive care guidelines for optimal health outcomes.",
                "reference": "https://www.cdc.gov",
                "relevance_score": 0.72
            }
        ]

    def get_treatment_evidence(self, condition: str) -> list[dict[str, Any]]:
        """Retrieve treatment-specific evidence for a condition."""
        query = f"treatment guidelines {condition} evidence-based recommendations"
        return self.retrieve_evidence(query, num_results=5)

    def get_prevention_evidence(self, risk_factors: list[str]) -> list[dict[str, Any]]:
        """Retrieve prevention evidence based on risk factors."""
        query = f"prevention strategies {' '.join(risk_factors)}"
        return self.retrieve_evidence(query, num_results=3)
