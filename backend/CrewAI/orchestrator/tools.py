"""
CrewAI tools for the healthcare crew.

Each tool is a thin wrapper over a deterministic service from
``services.py`` plus, where relevant, a fitted model or an ingested RAG
pipeline. Tools consume the outputs of preprocessing and prediction
modules; they never implement machine learning themselves.
"""

from __future__ import annotations

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from preprocessing.logger import get_logger

from .schemas import EvidenceItem, PatientInfo, PredictionResult, RiskResult
from .services import (
    assemble_clinical_report,
    assess_risk,
    retrieve_evidence,
    run_prediction,
)

logger = get_logger(__name__)


class PredictionInput(BaseModel):
    """Feature values for one patient row."""

    features: dict[str, float] = Field(
        description="Numeric feature values for the patient row"
    )


class PredictionTool(BaseTool):
    """
    Predict a patient's condition from a fitted model.
    """

    name: str = "disease_prediction"
    description: str = (
        "Run a fitted clinical model on one patient's feature row and "
        "return the predicted class, per-class probabilities, and confidence."
    )
    args_schema: type[BaseModel] = PredictionInput

    def __init__(self, model, **kwargs) -> None:
        """Bind the fitted model to the tool."""
        super().__init__(**kwargs)
        self._model = model

    def _run(self, features: dict[str, float]) -> dict:
        """Predict for the given feature row."""
        result = run_prediction(self._model, features)
        logger.info("PredictionTool produced %s", result.predicted_class)
        return result.model_dump()


class RiskAssessmentInput(BaseModel):
    """Prediction result plus optional clinical markers."""

    prediction: dict[str, object] = Field(description="Model prediction result dict")
    markers: dict[str, float] = Field(
        default_factory=dict, description="Optional clinical markers"
    )


class RiskAssessmentTool(BaseTool):
    """
    Assess risk from the predicted probability of the disease class.
    """

    name: str = "risk_assessment"
    description: str = (
        "Score risk from the predicted probability of the disease class "
        "and any elevated clinical markers; returns a risk level, "
        "contributing factors, and a monitoring schedule."
    )
    args_schema: type[BaseModel] = RiskAssessmentInput

    def _run(self, prediction: dict, markers: dict | None = None) -> dict:
        """Assess risk for the given prediction."""
        result = assess_risk(PredictionResult(**prediction), markers)
        return result.model_dump()


class RAGRetrievalInput(BaseModel):
    """Evidence query."""

    query: str = Field(description="Clinical query to retrieve evidence for")
    top_k: int = Field(default=3, description="Number of evidence items to return")


class RAGRetrievalTool(BaseTool):
    """
    Retrieve clinical evidence from the RAG knowledge base.
    """

    name: str = "evidence_retrieval"
    description: str = (
        "Retrieve clinical evidence chunks relevant to a query from the "
        "RAG knowledge base, with source labels and similarity scores."
    )
    args_schema: type[BaseModel] = RAGRetrievalInput

    def __init__(self, pipeline, **kwargs) -> None:
        """Bind the ingested RAG pipeline to the tool."""
        super().__init__(**kwargs)
        self._pipeline = pipeline

    def _run(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve evidence for the query."""
        items = retrieve_evidence(self._pipeline, query, top_k=top_k)
        logger.info("RAGRetrievalTool returned %d items", len(items))
        return [item.model_dump() for item in items]


class ReportInput(BaseModel):
    """Components to merge into the clinical report."""

    patient: dict[str, object] = Field(description="Patient info dict")
    input_type: str = Field(default="csv", description="Input modality")
    prediction: dict[str, object] | None = Field(
        default=None, description="Prediction result dict"
    )
    risk: dict[str, object] | None = Field(default=None, description="Risk result dict")
    evidence: list[dict[str, object]] = Field(
        default_factory=list, description="Evidence item dicts"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendation strings"
    )


class ClinicalReportTool(BaseTool):
    """
    Assemble the final structured clinical report.
    """

    name: str = "clinical_report"
    description: str = (
        "Merge patient context, prediction, risk, evidence, and "
        "recommendations into the final structured clinical report."
    )
    args_schema: type[BaseModel] = ReportInput

    def _run(
        self,
        patient: dict,
        input_type: str = "csv",
        prediction: dict | None = None,
        risk: dict | None = None,
        evidence: list[dict] | None = None,
        recommendations: list[str] | None = None,
    ) -> dict:
        """Assemble and return the clinical report."""
        report = assemble_clinical_report(
            patient=PatientInfo(**patient),
            input_type=input_type,
            prediction=PredictionResult(**prediction) if prediction else None,
            risk=RiskResult(**risk) if risk else None,
            evidence=[EvidenceItem(**item) for item in (evidence or [])],
            recommendations=recommendations,
        )
        return report.to_dict()


__all__ = [
    "ClinicalReportTool",
    "PredictionTool",
    "RAGRetrievalTool",
    "RiskAssessmentTool",
]
