"""
CrewAI orchestration module.

Agents orchestrate reasoning over the outputs of the preprocessing,
prediction, and retrieval modules. The deterministic services run the
analysis offline; CrewAI agents optionally enrich the narrative when an
LLM API key is configured (ADR-008).
"""

from .config import CrewSettings, settings
from .crew import ClinicalCrew
from .exceptions import (
    CrewError,
    LLMNotConfiguredError,
    OrchestrationError,
    PredictionToolError,
    ReportError,
    RetrievalToolError,
    RiskToolError,
)
from .schemas import (
    ClinicalReport,
    EvidenceItem,
    PatientInfo,
    PredictionResult,
    RiskResult,
)
from .services import (
    assemble_clinical_report,
    assess_risk,
    retrieve_evidence,
    run_prediction,
)
from .tools import (
    ClinicalReportTool,
    PredictionTool,
    RAGRetrievalTool,
    RiskAssessmentTool,
)

__all__ = [
    "ClinicalCrew",
    "ClinicalReport",
    "ClinicalReportTool",
    "CrewError",
    "CrewSettings",
    "EvidenceItem",
    "LLMNotConfiguredError",
    "OrchestrationError",
    "PatientInfo",
    "PredictionResult",
    "PredictionTool",
    "PredictionToolError",
    "RAGRetrievalTool",
    "ReportError",
    "RetrievalToolError",
    "RiskAssessmentTool",
    "RiskResult",
    "RiskToolError",
    "assemble_clinical_report",
    "assess_risk",
    "retrieve_evidence",
    "run_prediction",
    "settings",
]
