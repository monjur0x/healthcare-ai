"""
Custom exceptions for the CrewAI orchestration module.
"""


class CrewError(Exception):
    """
    Base orchestration exception.
    """


class OrchestrationError(CrewError):
    """
    Raised when the analysis pipeline cannot be assembled or run.
    """


class PredictionToolError(CrewError):
    """
    Raised when a model prediction cannot be produced.
    """


class RetrievalToolError(CrewError):
    """
    Raised when evidence retrieval fails.
    """


class RiskToolError(CrewError):
    """
    Raised when risk assessment fails.
    """


class ReportError(CrewError):
    """
    Raised when the clinical report cannot be assembled.
    """


class ExplanationError(CrewError):
    """
    Raised when a model-derived explanation cannot be produced.
    """


class LLMNotConfiguredError(CrewError):
    """
    Raised when LLM orchestration is requested without a configured
    ``CREW_LLM_API_KEY``.
    """


__all__ = [
    "CrewError",
    "ExplanationError",
    "LLMNotConfiguredError",
    "OrchestrationError",
    "PredictionToolError",
    "ReportError",
    "RetrievalToolError",
    "RiskToolError",
]
