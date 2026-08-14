"""
Pydantic schemas for the CrewAI orchestration module.

These models describe the data that flows between agents, tools, and
the final clinical report. All are serializable to JSON-friendly dicts
so results can be persisted or rendered by the dashboard.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class PatientInfo(BaseModel):
    """
    Demographic context for a clinical analysis.

    Parameters
    ----------
    name : str
        Patient display name.
    id : str
        Patient identifier (never PHI; treated as a study id).
    age : int | None
        Age in years.
    notes : str
        Free-form clinical notes.
    """

    name: str = "Unknown"
    id: str = "Unknown"
    age: int | None = None
    notes: str = ""


class PredictionResult(BaseModel):
    """
    Output of a model prediction for one patient row.

    Parameters
    ----------
    predicted_class : str
        Predicted label.
    probabilities : dict[str, float]
        Probability per class label.
    confidence : float
        Probability of the predicted class in ``[0, 1]``.
    model_name : str
        Model identifier that produced the prediction.
    """

    predicted_class: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str = "tabular"


class RiskResult(BaseModel):
    """
    Deterministic risk assessment for a prediction.

    Parameters
    ----------
    risk_score : float
        Risk score in ``[0, 1]``.
    risk_level : str
        One of ``"low"`` / ``"medium"`` / ``"high"``.
    risk_factors : list[str]
        Clinical markers that exceeded configured thresholds.
    monitoring_schedule : list[dict[str, str]]
        Recommended monitoring tests by risk level.
    """

    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: str
    risk_factors: list[str] = Field(default_factory=list)
    monitoring_schedule: list[dict[str, str]] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """
    A single retrieved evidence chunk.

    Parameters
    ----------
    document_id : str
        Source document identifier.
    source : str
        Knowledge source label (e.g. ``"PubMed"``, ``"protocols"``).
    score : float
        Retrieval similarity score.
    text : str
        Retrieved text.
    """

    document_id: str
    source: str = ""
    score: float = 0.0
    text: str = ""


class ClinicalReport(BaseModel):
    """
    The final structured clinical report.

    Assembled deterministically from tool outputs; the LLM layer may
    enrich the narrative fields when configured.
    """

    patient: PatientInfo
    input_type: str = "csv"
    patient_summary: str = ""
    prediction: PredictionResult | None = None
    risk: RiskResult | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    context: str = ""
    recommendations: list[str] = Field(default_factory=list)
    limitations: str = (
        "AI analysis has inherent limitations. This report is a decision "
        "support tool only and must be reviewed by a licensed physician."
    )
    doctor_notice: str = (
        "This report is AI-assisted. Final diagnosis must be made by a "
        "licensed physician."
    )

    @model_validator(mode="after")
    def _derive_context(self) -> ClinicalReport:
        """Populate the composed context when evidence is present."""
        if not self.context and self.evidence:
            self.context = "\n\n".join(
                f"[{item.document_id}] ({item.score:.4f})\n{item.text}"
                for item in self.evidence
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the report to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Report fields keyed by name.
        """

        return self.model_dump()


__all__ = [
    "ClinicalReport",
    "EvidenceItem",
    "PatientInfo",
    "PredictionResult",
    "RiskResult",
]
