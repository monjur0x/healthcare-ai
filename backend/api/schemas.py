"""
Pydantic request / response models for the Healthcare AI API.

Request models validate incoming JSON; response models reuse the
orchestrator schemas (``ClinicalReport``, ``PredictionResult``,
``EvidenceItem``) so serialized shapes stay consistent across modules.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from CrewAI.orchestrator.schemas import (
    ClinicalReport,
    EvidenceItem,
    PatientInfo,
    PredictionResult,
)


class PredictRequest(BaseModel):
    """
    A single feature row to classify.

    Parameters
    ----------
    features : dict[str, float]
        Feature values keyed by column name.
    """

    features: dict[str, float] = Field(..., min_length=1)


class RetrieveRequest(BaseModel):
    """
    An evidence retrieval query.

    Parameters
    ----------
    query : str
        Query text (non-empty).
    top_k : int | None
        Number of evidence items to return; defaults to the
        orchestrator's configured RAG top-k.
    """

    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AnalyzeRequest(BaseModel):
    """
    A full clinical analysis request.

    Parameters
    ----------
    patient : PatientInfo
        Patient context (name, study id, age, notes).
    features : dict[str, float]
        Preprocessed feature row for the prediction step.
    markers : dict[str, float] | None
        Optional raw clinical markers feeding the risk assessment.
    recommendations : list[str] | None
        Optional recommendation strings for the report.
    input_type : str
        Data modality analyzed (``"csv"`` / ``"image"`` / ...).
    """

    patient: PatientInfo = Field(default_factory=PatientInfo)
    features: dict[str, float] = Field(default_factory=dict)
    markers: dict[str, float] | None = None
    recommendations: list[str] | None = None
    input_type: str = "csv"


class HealthResponse(BaseModel):
    """
    Server metadata response.

    Parameters
    ----------
    status : str
        Liveness status.
    name : str
        Application name.
    version : str
        Application version.
    """

    status: str
    name: str
    version: str


__all__ = [
    "AnalyzeRequest",
    "ClinicalReport",
    "EvidenceItem",
    "HealthResponse",
    "PatientInfo",
    "PredictRequest",
    "PredictionResult",
    "RetrieveRequest",
]
