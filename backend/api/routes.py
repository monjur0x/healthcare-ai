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
from .schemas import (
    AnalyzeCSVRequest,
    AnalyzeImageRequest,
    AnalyzeRequest,
    ModelInfo,
    PredictRequest,
    PresetInfo,
    RetrieveRequest,
    TrainRequest,
    TrainResponse,
)
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


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest, service: ServiceDependency) -> TrainResponse:
    """
    Train (or retrain) a tabular model and serve it immediately.

    Parameters
    ----------
    request : TrainRequest
        Training request (preset or dataset + target).
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    TrainResponse
        Artifact path and hold-out metrics.
    """

    result = service.train(
        preset=request.preset,
        dataset=request.dataset,
        target=request.target,
        model=request.model,
        test_size=request.test_size,
        seed=request.seed,
        max_rows=request.max_rows,
        federated=request.federated,
        distributed=request.distributed,
        clients=request.clients,
        rounds=request.rounds,
        differential_privacy=request.differential_privacy,
        noise_multiplier=request.noise_multiplier,
        max_grad_norm=request.max_grad_norm,
        privacy_delta=request.privacy_delta,
        secure_aggregation=request.secure_aggregation,
    )
    return TrainResponse(**result.to_dict())


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


@router.post("/analyze/image", response_model=ClinicalReport)
def analyze_image(
    request: AnalyzeImageRequest, service: ServiceDependency
) -> ClinicalReport:
    """
    Analyze an uploaded image with the image model and return a report.

    Parameters
    ----------
    request : AnalyzeImageRequest
        Analysis payload (patient, image bytes, markers).
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    ClinicalReport
        The assembled structured report.
    """

    return service.analyze_image(
        patient=request.patient,
        image=request.image,
        markers=request.markers,
        recommendations=request.recommendations,
    )


@router.post("/analyze/csv", response_model=ClinicalReport)
def analyze_csv(
    request: AnalyzeCSVRequest, service: ServiceDependency
) -> ClinicalReport:
    """
    Analyze the first row of an uploaded CSV and return a report.

    The CSV is preprocessed entirely on the backend
    (``preprocessing.csv.CSVPipeline``); the dashboard only uploads the
    raw bytes.

    Parameters
    ----------
    request : AnalyzeCSVRequest
        Analysis payload (patient, CSV bytes, markers).
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    ClinicalReport
        The assembled structured report.
    """

    return service.analyze_csv(
        patient=request.patient,
        csv=request.csv,
        markers=request.markers,
        recommendations=request.recommendations,
        input_type=request.input_type,
    )


@router.get("/model", response_model=ModelInfo)
def model_info(service: ServiceDependency) -> ModelInfo:
    """
    Describe the configured prediction models.

    Parameters
    ----------
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    ModelInfo
        Available models, types, classes, and expected features.
    """

    return ModelInfo(**service.model_info())


@router.get("/presets", response_model=list[PresetInfo])
def presets(service: ServiceDependency) -> list[PresetInfo]:
    """
    Describe the named dataset presets and their feature schemas.

    Parameters
    ----------
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    list[PresetInfo]
        Preset metadata ordered by name; ``available`` reflects whether a
        trained artifact exists.
    """

    return [PresetInfo(**info) for info in service.presets_info()]


__all__ = ["router"]
