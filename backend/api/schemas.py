"""
Pydantic request / response models for the Healthcare AI API.

Request models validate incoming JSON; response models reuse the
orchestrator schemas (``ClinicalReport``, ``PredictionResult``,
``EvidenceItem``) so serialized shapes stay consistent across modules.
"""

from __future__ import annotations

import base64
import binascii

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from CrewAI.orchestrator.schemas import (
    ClinicalReport,
    EvidenceItem,
    PatientInfo,
    PredictionResult,
)

DatasetPreset = Literal["diabetes", "heart", "kidney", "sepsis"]


class TrainRequest(BaseModel):
    """
    Train (or retrain) a tabular model from a dataset.

    Either ``preset`` or both ``dataset`` and ``target`` must be given.
    When ``federated`` is true the model is trained through the federated
    FedAvg path; otherwise a single central model is fitted.

    Parameters
    ----------
    preset : DatasetPreset | None
        Named dataset preset resolved against the configured dataset dir.
    dataset : str | None
        Explicit path to a CSV (mutually exclusive with ``preset``).
    target : str | None
        Target column name (required when ``dataset`` is given).
    model : Literal["mlp", "logistic"]
        Scikit-learn model family to fit.
    test_size : float
        Fraction of rows held out for evaluation.
    seed : int
        Random seed for reproducibility.
    max_rows : int | None
        Optional cap on the number of rows used.
    federated : bool
        When true, train through the federated FedAvg path.
    distributed : bool
        When true and ``federated`` is set, hospitals run as separate
        processes over Flower gRPC (requires ``preset``).
    clients : int
        Number of hospital clients (federated path).
    rounds : int
        Number of federated rounds (federated path).
    differential_privacy : bool
        When true and ``federated`` is set, local client training uses
        Opacus DP-SGD (requires a torch-backed model).
    noise_multiplier : float
        DP-SGD noise multiplier.
    max_grad_norm : float
        DP-SGD per-sample gradient clipping norm.
    privacy_delta : float
        Target privacy delta for the epsilon audit.
    secure_aggregation : bool
        When true, client updates are masked with the pairwise
        one-time-pad secure aggregator.
    """

    preset: DatasetPreset | None = None
    dataset: str | None = None
    target: str | None = None
    model: Literal["mlp", "logistic"] = "mlp"
    test_size: float = Field(default=0.25, ge=0.1, le=0.5)
    seed: int = 42
    max_rows: int | None = Field(default=None, ge=1)
    federated: bool = False
    distributed: bool = False
    clients: int = Field(default=3, ge=1, le=16)
    rounds: int = Field(default=3, ge=1, le=50)
    differential_privacy: bool = False
    noise_multiplier: float = Field(default=1.1, gt=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    privacy_delta: float = Field(default=1e-5, gt=0.0, le=1.0)
    secure_aggregation: bool = False


class TrainResponse(BaseModel):
    """
    Result of a training run.

    Parameters
    ----------
    model_path : str
        Path to the persisted model artifact.
    dataset : str
        Dataset the model was trained on.
    target : str
        Target column used.
    accuracy : float
        Hold-out accuracy.
    roc_auc : float | None
        Hold-out ROC-AUC (None if undefined).
    f1 : float | None
        Hold-out macro F1 (None if undefined).
    federated : bool
        Whether the federated path was used.
    federated_metrics : dict[str, Any] | None
        Federated round metrics (federated path only).
    """

    model_path: str
    dataset: str
    target: str
    accuracy: float
    roc_auc: float | None = None
    f1: float | None = None
    federated: bool = False
    federated_metrics: dict[str, Any] | None = None


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


class AnalyzeCSVRequest(BaseModel):
    """
    A CSV-based clinical analysis request.

    The frontend never parses or transforms the CSV for inference: it
    uploads the raw bytes and the backend preprocessing pipeline produces
    the feature row (ADR-003).

    Parameters
    ----------
    patient : PatientInfo
        Patient context (name, study id, age, notes).
    csv : bytes
        Raw CSV file bytes (UTF-8) for the preprocessing pipeline.
    markers : dict[str, float] | None
        Optional raw clinical markers feeding the risk assessment.
    recommendations : list[str] | None
        Optional recommendation strings for the report.
    input_type : str
        Data modality analyzed (default ``"csv"``).
    """

    patient: PatientInfo = Field(default_factory=PatientInfo)
    csv: bytes
    markers: dict[str, float] | None = None
    recommendations: list[str] | None = None
    input_type: str = "csv"

    @field_validator("csv", mode="before")
    @classmethod
    def decode_base64_csv(cls, value: Any) -> bytes:
        """
        Decode a base64-encoded JSON string into raw CSV bytes.

        Parameters
        ----------
        value : Any
            Raw JSON value (a base64 string or already bytes).

        Returns
        -------
        bytes
            Decoded CSV bytes.

        Raises
        ------
        ValueError
            If the value is not a base64 string or decodes to nothing.
        """

        if isinstance(value, bytes):
            return value
        if not isinstance(value, str):
            raise ValueError("csv must be a base64-encoded string.")
        try:
            decoded = base64.b64decode(value, validate=True)
        except binascii.Error as error:
            raise ValueError(f"Invalid base64 csv: {error}") from error
        if not decoded:
            raise ValueError("csv base64 payload is empty.")
        return decoded


class PresetInfo(BaseModel):
    """
    Metadata about a named dataset / model preset.

    ``available`` is true only when a trained artifact exists for the
    preset; the feature schema is then read from that artifact so the
    dashboard can render the exact fields a doctor must provide.

    Parameters
    ----------
    name : str
        Preset name (``"diabetes"`` / ``"heart"`` / ``"kidney"`` /
        ``"sepsis"``).
    dataset : str
        Source CSV file name.
    target : str
        Target column name.
    available : bool
        Whether a trained artifact exists for this preset.
    feature_names : list[str] | None
        Feature columns the served preset model expects (when available).
    classes : list[str] | None
        Class labels of the preset model (when available).
    """

    name: str
    dataset: str
    target: str
    available: bool = False
    feature_names: list[str] | None = None
    classes: list[str] | None = None


class AnalyzeImageRequest(BaseModel):
    """
    An image-based clinical analysis request.

    Parameters
    ----------
    patient : PatientInfo
        Patient context (name, study id, age, notes).
    image : bytes
        Raw image file bytes (PNG / JPEG) for the image model.
    markers : dict[str, float] | None
        Optional raw clinical markers feeding the risk assessment.
    recommendations : list[str] | None
        Optional recommendation strings for the report.
    """

    patient: PatientInfo = Field(default_factory=PatientInfo)
    image: bytes
    markers: dict[str, float] | None = None
    recommendations: list[str] | None = None

    @field_validator("image", mode="before")
    @classmethod
    def decode_base64_image(cls, value: Any) -> bytes:
        """
        Decode a base64-encoded JSON string into raw image bytes.

        Parameters
        ----------
        value : Any
            Raw JSON value (a base64 string or already bytes).

        Returns
        -------
        bytes
            Decoded image bytes.

        Raises
        ------
        ValueError
            If the value is not a base64 string or decodes to nothing.
        """

        if isinstance(value, bytes):
            return value
        if not isinstance(value, str):
            raise ValueError("image must be a base64-encoded string.")
        try:
            decoded = base64.b64decode(value, validate=True)
        except binascii.Error as error:
            raise ValueError(f"Invalid base64 image: {error}") from error
        if not decoded:
            raise ValueError("image base64 payload is empty.")
        return decoded


class ModelInfo(BaseModel):
    """
    Metadata about the configured prediction models.

    Parameters
    ----------
    available : bool
        Whether any model (tabular or image) is configured.
    model_type : str | None
        ``"tabular"`` / ``"image"`` / ``"tabular_and_image"`` or None.
    model_name : str | None
        Name of the primary model (tabular when present, else image).
    classes : list[str] | None
        Class labels of the primary model.
    feature_names : list[str] | None
        Expected feature columns (tabular model only).
    preset : str | None
        Dataset preset the served tabular model was trained on, when
        known (None for a model loaded from ``API_MODEL_PATH``).
    """

    available: bool = False
    model_type: str | None = None
    model_name: str | None = None
    classes: list[str] | None = None
    feature_names: list[str] | None = None
    preset: str | None = None


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


class FederationRound(BaseModel):
    """
    Per-round global metrics from a federation run.

    Parameters
    ----------
    round_index : int
        1-based round number.
    accuracy : float | None
        Global count-weighted accuracy.
    log_loss : float | None
        Global count-weighted log loss.
    n_clients : int | None
        Number of participating clients.
    bytes_exchanged : int | None
        Estimated bytes exchanged in the round.
    duration_s : float | None
        Wall-clock duration of the round.
    """

    round_index: int
    accuracy: float | None = None
    log_loss: float | None = None
    n_clients: int | None = None
    bytes_exchanged: int | None = None
    duration_s: float | None = None


class FederationRun(BaseModel):
    """
    Metadata about one distributed federation run.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    preset : str
        Dataset preset federated.
    n_hospitals : int
        Number of participating hospital sites.
    n_rounds : int
        Number of federated rounds.
    secure_aggregation : bool
        Whether client updates were masked.
    differential_privacy : bool
        Whether clients applied DP-SGD.
    status : str
        ``"running"`` / ``"completed"``.
    created_at : str
        Run creation timestamp.
    completed_at : str | None
        Completion timestamp (None while running).
    """

    run_id: str
    preset: str
    n_hospitals: int
    n_rounds: int
    secure_aggregation: bool
    differential_privacy: bool
    status: str
    created_at: str
    completed_at: str | None = None


class FederationModel(BaseModel):
    """
    A versioned global model artifact registered by a federation run.

    Parameters
    ----------
    id : int
        Registry row id.
    run_id : str
        The run that produced the model.
    preset : str
        Dataset preset the model predicts.
    version : int
        Monotonic version for the preset.
    model_path : str
        Path to the persisted model artifact.
    accuracy : float | None
        Hold-out accuracy.
    roc_auc : float | None
        Hold-out ROC-AUC.
    epsilon : float | None
        Worst-case DP epsilon (DP runs only).
    secure_aggregation : bool | None
        Whether the producing run masked client updates (None for the
        flat model list, where run details live in :class:`FederationRun`).
    differential_privacy : bool | None
        Whether the producing run applied DP-SGD (see above).
    created_at : str
        Registration timestamp.
    """

    id: int
    run_id: str
    preset: str
    version: int
    model_path: str
    accuracy: float | None = None
    roc_auc: float | None = None
    epsilon: float | None = None
    secure_aggregation: bool | None = None
    differential_privacy: bool | None = None
    created_at: str


class FederationPreset(BaseModel):
    """
    A preset's registry summary: preset metadata plus latest global model.

    Parameters
    ----------
    name : str
        Preset name.
    dataset : str
        Source CSV file name.
    target : str
        Target column name.
    available : bool
        Whether at least one global model is registered for the preset.
    feature_names : list[str] | None
        Feature columns of the latest served model (always None for
        registry entries; artifacts are authoritative).
    classes : list[str] | None
        Class labels of the latest model (always None here).
    latest_model : FederationModel | None
        The most recent registered global model for the preset.
    """

    name: str
    dataset: str
    target: str
    available: bool = False
    feature_names: list[str] | None = None
    classes: list[str] | None = None
    latest_model: FederationModel | None = None


class FederationStatus(BaseModel):
    """
    Overview of the federation model registry.

    Parameters
    ----------
    registry_path : str | None
        Path to the SQLite registry database (None when not configured).
    n_runs : int
        Total recorded runs.
    n_models : int
        Total registered global models.
    presets : list[FederationPreset]
        Per-preset summary with the latest registered model.
    """

    registry_path: str | None = None
    n_runs: int = 0
    n_models: int = 0
    presets: list[FederationPreset] = []


__all__ = [
    "AnalyzeCSVRequest",
    "AnalyzeImageRequest",
    "AnalyzeRequest",
    "ClinicalReport",
    "DatasetPreset",
    "EvidenceItem",
    "FederationModel",
    "FederationPreset",
    "FederationRound",
    "FederationRun",
    "FederationStatus",
    "HealthResponse",
    "ModelInfo",
    "PatientInfo",
    "PredictRequest",
    "PredictionResult",
    "PresetInfo",
    "RetrieveRequest",
    "TrainRequest",
    "TrainResponse",
]
