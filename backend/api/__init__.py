"""
Healthcare AI - FastAPI Module

Responsible for exposing REST APIs: request validation, optional
authentication, response serialization. Routes contain no business
logic; they validate input and delegate to ``api/services.py``, which
orchestrates the prediction models, the CrewAI clinical crew, and the
RAG pipeline (see ``AGENTS.md``).
"""

from .config import APISettings, settings
from .exceptions import (
    APIError,
    AuthenticationError,
    InvalidInputError,
    NotFoundError,
    ServiceUnavailableError,
)
from .main import create_app
from .services import (
    AnalysisService,
    build_rag_pipeline,
    load_predictive_model,
)

__all__ = [
    "APIError",
    "APISettings",
    "AnalysisService",
    "AuthenticationError",
    "InvalidInputError",
    "NotFoundError",
    "ServiceUnavailableError",
    "build_rag_pipeline",
    "create_app",
    "load_predictive_model",
    "settings",
]
