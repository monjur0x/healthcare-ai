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
from preprocessing.logger import get_logger

from .exceptions import AuthenticationError, ServiceUnavailableError
from .schemas import (
    AgentStepRequest,
    AnalyzeCSVRequest,
    AnalyzeImageRequest,
    AnalyzeRequest,
    EscalationAlert,
    FederationModel,
    FederationRound,
    FederationRun,
    FederationStatus,
    FeedbackRecord,
    FeedbackRequest,
    FeedbackStatus,
    ModelInfo,
    PredictRequest,
    PresetInfo,
    RetrainRequest,
    RetrainResponse,
    RetrieveRequest,
    RiskHistoryResponse,
    RiskHistorySummary,
    RiskTrend,
    TrainRequest,
    TrainResponse,
)
from .services import AnalysisService

logger = get_logger(__name__)


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
        tls_enabled=request.tls_enabled,
        tls_ca_cert=request.tls_ca_cert,
        tls_server_cert=request.tls_server_cert,
        tls_server_key=request.tls_server_key,
        tls_client_cert=request.tls_client_cert,
        tls_client_key=request.tls_client_key,
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


@router.get("/federation/status", response_model=FederationStatus)
def federation_status(service: ServiceDependency) -> FederationStatus:
    """
    Summarize the federation model registry.

    Parameters
    ----------
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    FederationStatus
        Registry path, run/model counts, and per-preset latest models.
    """

    return FederationStatus(**service.federation_status())


@router.get("/federation/runs", response_model=list[FederationRun])
def federation_runs(
    service: ServiceDependency, preset: str | None = None
) -> list[FederationRun]:
    """
    List federation runs, newest first.

    Parameters
    ----------
    service : AnalysisService
        Injected analysis service.
    preset : str | None
        Restrict to a preset when given.

    Returns
    -------
    list[FederationRun]
        Run rows ordered by creation time (descending).
    """

    return [FederationRun(**row) for row in service.federation_runs(preset)]


@router.get("/federation/models", response_model=list[FederationModel])
def federation_models(
    service: ServiceDependency, preset: str | None = None
) -> list[FederationModel]:
    """
    List registered global models, newest first.

    Parameters
    ----------
    service : AnalysisService
        Injected analysis service.
    preset : str | None
        Restrict to a preset when given.

    Returns
    -------
    list[FederationModel]
        Model rows ordered by registration time (descending).
    """

    return [FederationModel(**row) for row in service.federation_models(preset)]


@router.get("/federation/runs/{run_id}/rounds", response_model=list[FederationRound])
def federation_rounds(run_id: str, service: ServiceDependency) -> list[FederationRound]:
    """
    Return the per-round metrics of a specific run.

    Parameters
    ----------
    run_id : str
        The run id.
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    list[FederationRound]
        Round rows ordered by round index.
    """

    return [FederationRound(**row) for row in service.federation_rounds(run_id)]


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


@router.post("/feedback", response_model=FeedbackRecord)
def record_feedback(
    request: FeedbackRequest, service: ServiceDependency
) -> FeedbackRecord:
    """
    Record a clinician-confirmed outcome label for a past analysis.

    Parameters
    ----------
    request : FeedbackRequest
        Feedback payload (preset, patient id, features, confirmed label).
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    FeedbackRecord
        The persisted feedback record.
    """

    return service.record_feedback(
        preset=request.preset,
        patient_id=request.patient_id,
        features=request.features,
        confirmed_label=request.confirmed_label,
        predicted_label=request.predicted_label,
        confidence=request.confidence,
    )


@router.get("/feedback/status", response_model=FeedbackStatus)
def feedback_status(service: ServiceDependency) -> FeedbackStatus:
    """
    Summarize accumulated feedback and retrain readiness per preset.

    Parameters
    ----------
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    FeedbackStatus
        Per-preset pending counts, thresholds, and readiness flags.
    """

    return service.feedback_status()


@router.post("/feedback/retrain", response_model=RetrainResponse)
def feedback_retrain(
    request: RetrainRequest, service: ServiceDependency
) -> RetrainResponse:
    """
    Retrain a preset model on the base dataset plus pending feedback rows.

    The retrained artifact is served immediately and the consumed
    feedback rows are marked so they are not reused.

    Parameters
    ----------
    request : RetrainRequest
        Retrain payload (preset, model, test size, seed).
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    RetrainResponse
        Training result plus feedback consumption counts.
    """

    result = service.retrain_from_feedback(
        preset=request.preset,
        model=request.model,
        test_size=request.test_size,
        seed=request.seed,
    )
    return RetrainResponse(
        train=TrainResponse(**result.train.to_dict()),
        feedback_consumed=result.feedback_consumed,
        pending_remaining=result.pending_remaining,
    )


@router.get("/risk/history", response_model=RiskHistoryResponse)
def risk_history(
    patient_id: str | None = None,
    preset: str | None = None,
    limit: int = 100,
    service: ServiceDependency = None,
) -> RiskHistoryResponse:
    """
    Get risk history summaries for patients.

    Parameters
    ----------
    patient_id : str | None
        Filter by patient study id (optional).
    preset : str | None
        Filter by dataset preset (optional).
    limit : int
        Maximum number of records per patient (default 100).
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    RiskHistoryResponse
        Summaries with trend analysis and alert count.
    """

    if not service.risk_history_store:
        return RiskHistoryResponse()

    if patient_id and preset:
        summary = service.risk_history_store.get_summary(patient_id, preset)
        return RiskHistoryResponse(
            summaries=[summary],
            alert_count=1 if summary.trend and summary.trend.escalation_alert else 0,
        )

    return service.risk_history_store.get_all_summaries()


@router.get("/risk/history/{patient_id}", response_model=RiskHistorySummary)
def risk_history_patient(
    patient_id: str,
    preset: str,
    service: ServiceDependency,
) -> RiskHistorySummary:
    """
    Get detailed risk history for a specific patient-preset.

    Parameters
    ----------
    patient_id : str
        Patient study id.
    preset : str
        Dataset preset.
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    RiskHistorySummary
        History with trend and latest record.
    """

    if not service.risk_history_store:
        raise ServiceUnavailableError("Risk history store is not configured.")
    return service.risk_history_store.get_summary(patient_id, preset)


@router.get("/risk/trends/{patient_id}", response_model=RiskTrend)
def risk_trends(
    patient_id: str,
    preset: str,
    window: int | None = None,
    service: ServiceDependency = None,
) -> RiskTrend:
    """
    Get computed risk trend for a patient-preset.

    Parameters
    ----------
    patient_id : str
        Patient study id.
    preset : str
        Dataset preset.
    window : int | None
        Number of recent analyses for trend (default from settings).
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    RiskTrend
        Computed trend with direction, slope, and escalation alert.
    """

    if not service.risk_history_store:
        raise ServiceUnavailableError("Risk history store is not configured.")
    return service.risk_history_store.compute_trend(patient_id, preset, window)


@router.get("/risk/alerts", response_model=list[EscalationAlert])
def risk_alerts(service: ServiceDependency) -> list[EscalationAlert]:
    """
    Get all active escalation alerts.

    An alert is generated when a patient's risk score increases
    by more than the configured threshold compared to the previous analysis.

    Parameters
    ----------
    service : AnalysisService
        Injected analysis service.

    Returns
    -------
    list[EscalationAlert]
        Active alerts sorted by timestamp (newest first).
    """

    if not service.risk_history_store:
        raise ServiceUnavailableError("Risk history store is not configured.")
    return service.risk_history_store.get_escalation_alerts()


__all__ = ["router"]


# ---------------------------------------------------------------------------
# Per-Agent Step Endpoints (M5: n8n orchestrates each agent individually)
# ---------------------------------------------------------------------------


@router.post("/agents/patient-analyst")
def agent_patient_analyst(
    request: AgentStepRequest, service: ServiceDependency
) -> dict:
    """Patient Analyst agent: summarize and validate patient features."""
    from CrewAI.orchestrator.services import summarize_patient

    summary = summarize_patient(
        request.patient, request.features, request.markers, input_type="csv"
    )
    abnormal = [
        k
        for k, v in request.features.items()
        if isinstance(v, (int, float)) and (v > 200 or v < 0)
    ]

    return {
        "patient_summary": summary,
        "data_quality_notes": "checked" if not abnormal else "outliers detected",
        "key_indicators": list(request.features.keys())[:8],
    }


@router.post("/agents/disease-predictor")
def agent_disease_predictor(
    request: AgentStepRequest, service: ServiceDependency
) -> dict:
    """Disease Predictor agent: run ML prediction + risk assessment."""
    prediction = service.predict(request.features)
    from CrewAI.orchestrator.services import assess_risk

    risk = assess_risk(prediction, request.markers)
    return {
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "risk_score": risk.risk_score,
        "risk_level": risk.risk_level,
        "risk_factors": risk.risk_factors,
    }


@router.post("/agents/evidence-retrieval")
def agent_evidence_retrieval(
    request: AgentStepRequest, service: ServiceDependency
) -> list[dict]:
    """Medical Researcher agent: retrieve clinical evidence via RAG."""
    from CrewAI.orchestrator.services import build_evidence_query

    query = build_evidence_query(request.features)
    evidence = service.retrieve(query, top_k=3)
    return [e.model_dump() for e in evidence]


@router.post("/agents/treatment-planner")
def agent_treatment_planner(
    request: AgentStepRequest, service: ServiceDependency
) -> dict:
    """Treatment Planner agent: generate recommendations from risk level."""
    from CrewAI.orchestrator.services import (
        assess_risk,
        build_treatment_recommendations,
    )

    prediction = None
    risk = None
    fallback = False
    try:
        prediction = service.predict(request.features)
        risk = assess_risk(prediction, request.markers)
    except Exception as error:  # noqa: BLE001 — best-effort step: fall back, but log
        logger.warning("treatment-planner step fell back to medium risk: %s", error)
        fallback = True

    if not fallback:
        recs, monitoring = build_treatment_recommendations(prediction, risk)
        recs = [
            *recs,
            "All recommendations require physician review before implementation.",
        ]
        return {
            "recommendations": recs,
            "monitoring_schedule": monitoring,
            "fallback": False,
        }

    monitoring = {
        "low": [
            {"test": "Annual physical examination", "frequency": "Yearly"},
            {"test": "Blood pressure check", "frequency": "Annually"},
        ],
        "medium": [
            {"test": "Medical consultation", "frequency": "Every 3-6 months"},
            {"test": "Blood pressure monitoring", "frequency": "Monthly"},
            {"test": "Lipid panel", "frequency": "Every 6-12 months"},
        ],
        "high": [
            {"test": "Medical consultation", "frequency": "Monthly"},
            {"test": "Blood pressure monitoring", "frequency": "Weekly"},
            {"test": "Comprehensive metabolic panel", "frequency": "Monthly"},
            {"test": "HbA1c", "frequency": "Every 3 months"},
        ],
    }

    recs = [
        "Schedule follow-up consultation to monitor condition progression.",
        "All recommendations require physician review before implementation.",
    ]

    return {
        "recommendations": recs,
        "monitoring_schedule": monitoring["medium"],
        "fallback": True,
    }


@router.post("/agents/explainability")
def agent_explainability(request: AgentStepRequest, service: ServiceDependency) -> dict:
    """Explainability Expert agent: explain the prediction."""
    from CrewAI.orchestrator.services import build_explanation

    prediction = None
    fallback = False
    try:
        prediction = service.predict(request.features)
    except Exception as error:  # noqa: BLE001 — best-effort step: fall back, but log
        logger.warning("explainability step could not run prediction: %s", error)
        fallback = True

    explanation, contributing = build_explanation(prediction, request.features)

    return {
        "explanation": explanation or "Insufficient data for explanation.",
        "contributing_features": contributing,
        "fallback": fallback or prediction is None,
    }
