"""
HTTP routes for the Healthcare AI API.

Routes only validate input and delegate to the ``AnalysisService``;
business logic stays in ``api/services.py`` (see ``AGENTS.md``). Auth
(optional bearer token) is enforced via the router dependency.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from CrewAI.orchestrator.schemas import (
    ClinicalReport,
    EvidenceItem,
    PredictionResult,
)

from .exceptions import AuthenticationError, ServiceUnavailableError
from .schemas import AnalyzeRequest, PredictRequest, RetrieveRequest
from .services import AnalysisService


def get_token_validator(request: Request) -> None:
    """
    Reject requests when a bearer token is configured and not supplied.

    Parameters
    ----------
    request : Request
        Incoming request.

    Raises
    ------
    AuthenticationError
        If ``API_TOKEN`` is configured and the ``Authorization`` header
        does not carry ``Bearer <token>``.
    """

    token = getattr(request.app.state, "api_token", "") or ""
    if not token:
        return
    authorization = request.headers.get("authorization", "")
    if authorization != f"Bearer {token}":
        raise AuthenticationError("Missing or invalid bearer token.")


def get_analysis_service(request: Request) -> AnalysisService:
    """
    Resolve the configured ``AnalysisService`` from app state.

    Parameters
    ----------
    request : Request
        Incoming request.

    Returns
    -------
    AnalysisService
        The configured service instance.

    Raises
    ------
    ServiceUnavailableError
        If no service was attached to the app.
    """

    service = getattr(request.app.state, "analysis_service", None)
    if service is None:
        raise ServiceUnavailableError("The analysis service is not configured.")
    return service


ServiceDependency = Annotated[AnalysisService, Depends(get_analysis_service)]

router = APIRouter(
    prefix="/api/v1",
    tags=["clinical"],
    dependencies=[Depends(get_token_validator)],
)


@router.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest, service: ServiceDependency) -> PredictionResult:
    """
    Classify a single feature row.

    Parameters
    ----------
    request : PredictRequest
        Feature row payload.
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    PredictionResult
        Predicted class, probabilities, and confidence.
    """

    return service.predict(request.features)


@router.post("/retrieve", response_model=list[EvidenceItem])
def retrieve(
    request: RetrieveRequest, service: ServiceDependency
) -> list[EvidenceItem]:
    """
    Retrieve evidence chunks for a query.

    Parameters
    ----------
    request : RetrieveRequest
        Query payload.
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    list[EvidenceItem]
        Retrieved evidence ordered by descending score.
    """

    return service.retrieve(request.query, top_k=request.top_k)


@router.post("/analyze", response_model=ClinicalReport)
def analyze(request: AnalyzeRequest, service: ServiceDependency) -> ClinicalReport:
    """
    Run the full clinical analysis and return the structured report.

    Parameters
    ----------
    request : AnalyzeRequest
        Analysis payload (patient, features, markers).
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    ClinicalReport
        The assembled structured report.
    """

    return service.analyze(
        patient=request.patient,
        features=request.features,
        markers=request.markers,
        recommendations=request.recommendations,
        input_type=request.input_type,
    )


__all__ = ["router"]
