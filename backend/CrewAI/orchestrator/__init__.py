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
    "compute_agent_metrics",
    "decision_consistency",
    "retrieve_evidence",
    "run_prediction",
    "settings",
    "task_completion_rate",
]
