"""
Clinical orchestration module.

Seven deterministic agent steps (patient analysis → prediction →
evidence → treatment → explanation → risk monitoring → report)
orchestrate reasoning over the outputs of the preprocessing,
prediction, and retrieval modules. No LLM is involved: every value in
the report is computed, never generated.
"""

from .config import CrewSettings, settings
from .crew import ClinicalCrew
from .exceptions import (
    CrewError,
    OrchestrationError,
    PredictionToolError,
    ReportError,
    RetrievalToolError,
    RiskToolError,
)
from .metrics import (
    AgentMetrics,
    agent_collaboration_score,
    compute_agent_metrics,
    decision_consistency,
    task_completion_rate,
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
    build_disease_query,
    build_evidence_query,
    build_explanation,
    build_treatment_recommendations,
    retrieve_evidence,
    run_prediction,
    summarize_patient,
)
from .tools import (
    ClinicalReportTool,
    PredictionTool,
    RAGRetrievalTool,
    RiskAssessmentTool,
)

__all__ = [
    "AgentMetrics",
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
    "agent_collaboration_score",
    "assemble_clinical_report",
    "assess_risk",
    "build_disease_query",
    "build_evidence_query",
    "build_explanation",
    "build_treatment_recommendations",
    "compute_agent_metrics",
    "decision_consistency",
    "retrieve_evidence",
    "run_prediction",
    "settings",
    "summarize_patient",
    "task_completion_rate",
]
