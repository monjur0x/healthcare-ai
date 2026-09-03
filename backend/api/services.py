"""
Service layer for the FastAPI module.

All business logic lives here; routes only validate and delegate. The
``AnalysisService`` orchestrates the prediction model, the CrewAI
clinical crew, and the RAG pipeline, translating domain exceptions into
typed ``APIError`` subclasses at the service boundary.
"""

from __future__ import annotations

import os
import subprocess
import sys

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from CrewAI.orchestrator import ClinicalCrew
from CrewAI.orchestrator.exceptions import CrewError
from CrewAI.orchestrator.schemas import (
    ClinicalReport,
    EvidenceItem,
    PatientInfo,
    PredictionResult,
)
from CrewAI.orchestrator.services import retrieve_evidence, run_prediction
from evaluation import evaluate_classifier
from federated import FedAvgServer, FederatedClient, make_global_evaluator
from federated.config import settings as federation_settings
from federated.hospitals import PRESETS
from federated.privacy import (
    PrivacyConfig,
    membership_inference_auroc,
    privacy_metrics_summary,
)
from federated.registry import ModelRegistry
from feedback import FeedbackStore, FeedbackStoreError
from feedback import settings as feedback_settings
from feedback.schemas import FeedbackRecord, FeedbackStatus, FeedbackSummary
from models import (
    BaseModel,
    ImageClassifier,
    ModelLoadError,
    TabularClassifier,
    TorchMLPClassifier,
)
from preprocessing.csv import CSVPipeline
from preprocessing.image import ImagePipeline
from preprocessing.logger import get_logger
from rag import RAGPipeline
from rag.corpus import load_bundled_corpus, load_documents
from risk import RiskHistoryStore, RiskHistoryStoreError

from .config import APISettings
from .config import settings as default_settings
from .exceptions import InvalidInputError, ServiceUnavailableError

logger = get_logger(__name__)

DEFAULT_CORPUS: list[str] = [
    "diabetes mellitus is managed with metformin, lifestyle changes, and "
    "regular glucose monitoring",
    "chronic hypertension management combines dietary sodium reduction, "
    "exercise, and blood pressure lowering medication",
    "sepsis is life-threatening organ dysfunction from infection and "
    "requires broad-spectrum antibiotics within one hour of recognition",
]


def _normalize_token(value: str) -> str:
    """Lowercase and normalize a column name to the pipeline convention."""
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase snake_case (pipeline convention)."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for column in dataframe.columns:
        name = _normalize_token(str(column))
        if name in seen:
            name = f"{name}_{len(seen)}"
        seen.add(name)
        cleaned.append(name)
    dataframe.columns = cleaned
    return dataframe


def prepare_tabular_data(
    dataset: Path, target: str, max_rows: int | None
) -> tuple[pd.DataFrame, pd.Series, dict[str, object], dict[str, object]]:
    """
    Load and preprocess a CSV into a feature frame and encoded labels.

    Parameters
    ----------
    dataset : Path
        Path to the source CSV.
    target : str
        Target column name.
    max_rows : int | None
        Optional cap on the number of rows used.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, dict[str, object], dict[str, object]]
        Engineered feature frame, integer-encoded label series aligned
        by index, the fitted scaler's serializable parameters, and the
        fitted encoder's serializable parameters.

    Raises
    ------
    InvalidInputError
        If the target column is missing or the pipeline yields no data.
    """

    raw = pd.read_csv(dataset)
    if max_rows is not None:
        raw = raw.head(max_rows)
    raw = _normalize_columns(raw)
    target = _normalize_token(target)

    if target not in raw.columns:
        raise InvalidInputError(f"Target column '{target}' not found in {dataset}.")

    y_raw = raw[target]
    feature_frame = raw.drop(columns=[target])
    for column in ("id", "subject_id"):
        if column in feature_frame.columns:
            feature_frame = feature_frame.drop(columns=[column])

    valid = y_raw.notna()
    feature_frame = feature_frame.loc[valid]
    y_raw = y_raw.loc[valid]

    pipeline = CSVPipeline(input_columns=tuple(feature_frame.columns))
    result = pipeline.run(feature_frame)
    features = result.dataframe
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise InvalidInputError("Pipeline produced no usable features.")

    labels = y_raw.loc[features.index]
    if pd.api.types.is_string_dtype(labels):
        labels = labels.str.strip()
        labels = pd.Series(
            LabelEncoder().fit_transform(labels), index=labels.index, name=target
        )
    else:
        labels = pd.to_numeric(labels).astype(int)
    logger.info(
        "Prepared %d samples, %d features, %d classes from %s",
        features.shape[0],
        features.shape[1],
        labels.nunique(),
        dataset,
    )
    return features, labels, pipeline.scaler_params(), pipeline.encoder_params()


def _partition_shards(
    features: np.ndarray,
    labels: np.ndarray,
    n_clients: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition training data into class-balanced client shards."""
    counts = np.bincount(labels)
    if counts.min() < n_clients:
        raise InvalidInputError(
            f"Rarest class has {counts.min()} samples, fewer than {n_clients} "
            "clients. Reduce 'clients' or provide a larger dataset."
        )
    splitter = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=seed)
    return [
        (features[index], labels[index])
        for index, _ in splitter.split(features, labels)
    ]


def load_predictive_model(path: str | Path) -> TabularClassifier:
    """
    Load a persisted ``TabularClassifier`` artifact.

    Parameters
    ----------
    path : str | Path
        Path to the joblib artifact.

    Returns
    -------
    TabularClassifier
        The loaded model.

    Raises
    ------
    ServiceUnavailableError
        If the artifact cannot be loaded.
    """

    try:
        model = TabularClassifier.load(path)
    except ModelLoadError as error:
        raise ServiceUnavailableError(
            f"Could not load model from {path}: {error}"
        ) from error
    logger.info("Loaded model from %s", path)
    return model


def load_image_model(path: str | Path) -> ImageClassifier:
    """
    Load a persisted ``ImageClassifier`` artifact.

    Parameters
    ----------
    path : str | Path
        Path to the torch artifact.

    Returns
    -------
    ImageClassifier
        The loaded model.

    Raises
    ------
    ServiceUnavailableError
        If the artifact cannot be loaded.
    """

    try:
        model = ImageClassifier.load(path)
    except ModelLoadError as error:
        raise ServiceUnavailableError(
            f"Could not load image model from {path}: {error}"
        ) from error
    logger.info("Loaded image model from %s", path)
    return model


def build_rag_pipeline(corpus_dir: str | Path | None = None) -> RAGPipeline:
    """
    Build an ingested RAG pipeline from a corpus directory or the bundled corpus.

    Parameters
    ----------
    corpus_dir : str | Path | None
        Directory of ``.txt`` / ``.md`` documents. When None (or empty),
        the repository's bundled medical corpus (``rag/corpus/``) is
        ingested.

    Returns
    -------
    RAGPipeline
        Ingested retrieval pipeline.

    Raises
    ------
    ServiceUnavailableError
        If a corpus directory is given but contains no supported documents.
    """

    pipeline = RAGPipeline()
    if corpus_dir:
        directory = Path(corpus_dir)
        documents = load_documents(directory)
        if not documents:
            raise ServiceUnavailableError(
                f"No .txt/.md documents found under {directory}."
            )
        pipeline.ingest_documents(documents)
        logger.info("Ingested %d documents from %s", len(documents), directory)
    else:
        documents = load_bundled_corpus()
        if documents:
            pipeline.ingest_documents(documents)
            logger.info(
                "Ingested the bundled medical corpus (%d documents)", len(documents)
            )
        else:
            pipeline.ingest_texts(DEFAULT_CORPUS)
            logger.info(
                "Bundled corpus unavailable; ingested the built-in fallback (%d texts)",
                len(DEFAULT_CORPUS),
            )
    return pipeline


@dataclass
class TrainResult:
    """
    Outcome of a training run.

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "model_path": self.model_path,
            "dataset": self.dataset,
            "target": self.target,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "f1": self.f1,
            "federated": self.federated,
            "federated_metrics": self.federated_metrics,
        }


@dataclass
class RetrainResult:
    """
    Outcome of a feedback-triggered retrain.

    Parameters
    ----------
    train : TrainResult
        The underlying training result.
    feedback_consumed : int
        Number of feedback samples folded into the retrain.
    pending_remaining : int
        Number of feedback samples left pending for the preset.
    """

    train: TrainResult
    feedback_consumed: int
    pending_remaining: int


@dataclass
class AnalysisService:
    """
    Facade exposing the clinical analysis pipeline to the API.

    Parameters
    ----------
    model : TabularClassifier | None
        Fitted model for the prediction step, or None to skip prediction.
    image_model : ImageClassifier | None
        Fitted CNN for image-based analysis, or None to skip it.
    rag_pipeline : RAGPipeline | None
        Ingested retrieval pipeline, or None to skip evidence retrieval.
    artifacts_dir : Path
        Base directory for model artifacts written by :meth:`train`.
    dataset_dir : Path
        Base directory for preset datasets resolved by :meth:`train`.
    """

    model: TabularClassifier | None = None
    image_model: ImageClassifier | None = None
    rag_pipeline: RAGPipeline | None = None
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    dataset_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATASET_DIR", "."))
    )
    #: Dataset preset the in-memory tabular model was trained on, when
    #: known (None for a model loaded from ``API_MODEL_PATH``).
    active_preset: str | None = None
    #: Persistent clinician-feedback store for the retrain loop.
    feedback_store: FeedbackStore | None = None
    #: Persistent risk history store for longitudinal monitoring.
    risk_history_store: RiskHistoryStore | None = None

    @classmethod
    def from_settings(cls, cfg: APISettings | None = None) -> AnalysisService:
        """
        Build a service from API settings (lazy model load, corpus ingest).

        Parameters
        ----------
        cfg : APISettings | None
            Settings to use; defaults to the module-level ``settings``.

        Returns
        -------
        AnalysisService
            Configured service.
        """

        cfg = cfg or default_settings
        model = load_predictive_model(cfg.MODEL_PATH) if cfg.MODEL_PATH else None
        image_model = (
            load_image_model(cfg.IMAGE_MODEL_PATH) if cfg.IMAGE_MODEL_PATH else None
        )
        rag_pipeline = build_rag_pipeline(cfg.CORPUS_DIR or None)
        dataset_dir = Path(cfg.DATASET_DIR or os.environ.get("DATASET_DIR", "."))
        artifacts_dir = Path(cfg.ARTIFACTS_DIR)
        feedback_db = artifacts_dir / "feedback.db"
        risk_history_db = artifacts_dir / "risk_history.db"
        return cls(
            model=model,
            image_model=image_model,
            rag_pipeline=rag_pipeline,
            artifacts_dir=artifacts_dir,
            dataset_dir=dataset_dir,
            active_preset=cfg.ACTIVE_PRESET or None,
            feedback_store=FeedbackStore(feedback_db),
            risk_history_store=RiskHistoryStore(risk_history_db),
        )

    def train(
        self,
        preset: str | None = None,
        dataset: str | None = None,
        target: str | None = None,
        model: Literal["mlp", "logistic"] = "mlp",
        test_size: float = 0.25,
        seed: int = 42,
        max_rows: int | None = None,
        federated: bool = False,
        distributed: bool = False,
        clients: int = 3,
        rounds: int = 3,
        differential_privacy: bool = False,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        privacy_delta: float = 1e-5,
        secure_aggregation: bool = False,
        tls_enabled: bool = False,
        tls_ca_cert: str | None = None,
        tls_server_cert: str | None = None,
        tls_server_key: str | None = None,
        tls_client_cert: str | None = None,
        tls_client_key: str | None = None,
    ) -> TrainResult:
        """
        Preprocess a dataset, fit a tabular model, evaluate, and persist it.

        The fitted model replaces the service model so subsequent
        prediction / analysis calls use it without a restart. When
        ``federated`` is true the model is aggregated through the
        synchronous FedAvg path.

        Parameters
        ----------
        preset : str | None
            Named dataset preset resolved against ``dataset_dir``.
        dataset : str | None
            Explicit CSV path (requires ``target``).
        target : str | None
            Target column name.
        model : Literal["mlp", "logistic"]
            Model family to fit.
        test_size : float
            Hold-out fraction.
        seed : int
            Reproducibility seed.
        max_rows : int | None
            Optional row cap.
        federated : bool
            Use the federated FedAvg path when true.
        distributed : bool
            Run hospitals as separate processes over Flower gRPC when
            true (requires ``federated`` and a preset).
        clients : int
            Number of hospital clients (federated path).
        rounds : int
            Number of federated rounds (federated path).
        differential_privacy : bool
            Apply Opacus DP-SGD to local client steps (federated path,
            requires a torch-backed model).
        noise_multiplier : float
            DP-SGD noise multiplier.
        max_grad_norm : float
            DP-SGD per-sample gradient clipping norm.
        privacy_delta : float
            Target privacy delta for the epsilon audit.
        secure_aggregation : bool
            Mask client updates with the pairwise one-time-pad
            aggregator (federated path).
        tls_enabled : bool
            Enable TLS for gRPC connections (distributed path).
        tls_ca_cert : str | None
            Path to CA certificate PEM (distributed path, requires tls_enabled).
        tls_server_cert : str | None
            Path to server certificate PEM (distributed path, requires tls_enabled).
        tls_server_key : str | None
            Path to server private key PEM (distributed path, requires tls_enabled).
        tls_client_cert : str | None
            Path to client certificate PEM for mutual TLS (distributed path).
        tls_client_key : str | None
            Path to client private key PEM for mutual TLS (distributed path).

        Returns
        -------
        TrainResult
            Artifact path and hold-out metrics.

        Raises
        ------
        InvalidInputError
            If no preset / dataset is given, the target is missing, or
            the data cannot support the requested partition.
        ServiceUnavailableError
            If training fails for another reason.
        """

        dataset_path, target = self._resolve_dataset(preset, dataset, target)
        try:
            features, labels, scaler_params, encoder_params = prepare_tabular_data(
                dataset_path, target, max_rows
            )
        except (OSError, ValueError) as error:
            raise InvalidInputError(str(error)) from error

        train_x, test_x, train_y, test_y = train_test_split(
            features,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=seed,
        )

        try:
            if federated:
                if model != "mlp":
                    raise InvalidInputError(
                        "The federated path supports model_name='mlp' only "
                        "(incremental local steps)."
                    )
                if distributed:
                    fitted, fed_metrics = self._train_distributed(
                        preset=preset,
                        clients=clients,
                        rounds=rounds,
                        seed=seed,
                        differential_privacy=differential_privacy,
                        noise_multiplier=noise_multiplier,
                        max_grad_norm=max_grad_norm,
                        privacy_delta=privacy_delta,
                        secure_aggregation=secure_aggregation,
                        tls_enabled=tls_enabled,
                        tls_ca_cert=tls_ca_cert,
                        tls_server_cert=tls_server_cert,
                        tls_server_key=tls_server_key,
                        tls_client_cert=tls_client_cert,
                        tls_client_key=tls_client_key,
                    )
                else:
                    fitted, fed_metrics = self._train_federated(
                        train_x,
                        train_y,
                        test_x,
                        test_y,
                        model,
                        clients,
                        rounds,
                        seed,
                        differential_privacy=differential_privacy,
                        noise_multiplier=noise_multiplier,
                        max_grad_norm=max_grad_norm,
                        privacy_delta=privacy_delta,
                        secure_aggregation=secure_aggregation,
                    )
                if hasattr(fitted, "set_scaler_params"):
                    fitted.set_scaler_params(scaler_params)
                if hasattr(fitted, "set_encoder_params"):
                    fitted.set_encoder_params(encoder_params)
            else:
                fitted = TabularClassifier(model_name=model).fit(train_x, train_y)
                fitted.set_scaler_params(scaler_params)
                fitted.set_encoder_params(encoder_params)
                fed_metrics = None
        except InvalidInputError:
            raise
        except (ValueError, RuntimeError) as error:
            raise InvalidInputError(str(error)) from error

        metrics = evaluate_classifier(fitted, test_x, test_y)
        out_dir = self.artifacts_dir / (preset or dataset_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "global_model.joblib"
        fitted.save(model_path)

        self.model = fitted
        self.active_preset = preset
        logger.info(
            "Trained %s model on %s (federated=%s): accuracy=%.4f artifact=%s",
            model,
            dataset_path,
            federated,
            metrics.accuracy,
            model_path,
        )
        return TrainResult(
            model_path=str(model_path),
            dataset=str(dataset_path),
            target=target,
            accuracy=float(metrics.accuracy),
            roc_auc=float(metrics.roc_auc) if metrics.roc_auc is not None else None,
            f1=float(metrics.f1_macro),
            federated=federated,
            federated_metrics=fed_metrics,
        )

    def _resolve_dataset(
        self, preset: str | None, dataset: str | None, target: str | None
    ) -> tuple[Path, str]:
        """
        Resolve a (preset or explicit) dataset path and target column.

        An explicit ``dataset`` path wins over the preset's bundled file
        when both are supplied, while the preset still supplies the
        default target and output directory naming.
        """
        if preset is not None:
            if preset not in PRESETS:
                raise InvalidInputError(
                    f"Unknown preset '{preset}'. Choose from {sorted(PRESETS)}."
                )
            file_name, preset_target = PRESETS[preset]
            dataset_path = Path(dataset) if dataset else self.dataset_dir / file_name
            return dataset_path, target or preset_target
        if dataset is None or target is None:
            raise InvalidInputError(
                "Provide a 'preset' or both 'dataset' and 'target'."
            )
        return Path(dataset), target

    def record_feedback(
        self,
        preset: str,
        patient_id: str,
        features: dict[str, float],
        confirmed_label: int,
        predicted_label: int | None = None,
        confidence: float | None = None,
    ) -> FeedbackRecord:
        """
        Record one clinician-confirmed feedback sample.

        Parameters
        ----------
        preset : str
            Dataset preset the feedback refers to.
        patient_id : str
            Patient study id.
        features : dict[str, float]
            Feature row that was analyzed.
        confirmed_label : int
            Clinician-confirmed outcome label (0/1).
        predicted_label : int | None
            Model prediction at analysis time (when recorded).
        confidence : float | None
            Model confidence at analysis time (when recorded).

        Returns
        -------
        FeedbackRecord
            The persisted record.

        Raises
        ------
        InvalidInputError
            If the preset is unknown.
        ServiceUnavailableError
            If the feedback store is unavailable.
        """

        self._validate_preset(preset)
        if self.feedback_store is None:
            raise ServiceUnavailableError(
                "Feedback store is not configured on this service."
            )
        try:
            record = self.feedback_store.add(
                preset=preset,
                patient_id=patient_id,
                features=features,
                confirmed_label=confirmed_label,
                predicted_label=predicted_label,
                confidence=confidence,
            )
        except FeedbackStoreError as error:
            raise ServiceUnavailableError(str(error)) from error
        logger.info(
            "Recorded feedback id=%d preset=%s patient=%s label=%d",
            record.id,
            preset,
            patient_id,
            confirmed_label,
        )
        return record

    def feedback_status(self) -> FeedbackStatus:
        """
        Summarize accumulated feedback across all presets.

        Returns
        -------
        FeedbackStatus
            Per-preset pending counts, thresholds, and readiness flags.
        """

        summaries: list[FeedbackSummary] = []
        if self.feedback_store is not None:
            for preset in sorted(PRESETS):
                file_name, target = PRESETS[preset]
                summaries.append(
                    FeedbackSummary(
                        preset=preset,
                        dataset=file_name,
                        target=target,
                        pending=self.feedback_store.count_pending(preset),
                        threshold=feedback_settings.RETRAIN_THRESHOLD,
                        ready=(
                            self.feedback_store.count_pending(preset)
                            >= feedback_settings.RETRAIN_THRESHOLD
                        ),
                        recent=self.feedback_store.recent(preset),
                    )
                )
        return FeedbackStatus(
            retrain_enabled=feedback_settings.RETRAIN_ENABLED,
            presets=summaries,
        )

    def retrain_from_feedback(
        self,
        preset: str,
        model: Literal["mlp", "logistic"] = "mlp",
        test_size: float = 0.25,
        seed: int = 42,
    ) -> RetrainResult:
        """
        Retrain a preset model on the base dataset augmented with pending
        feedback rows.

        The retrained artifact replaces the served model (written to the
        preset's artifact directory) and the consumed feedback rows are
        marked so they are not reused.

        Parameters
        ----------
        preset : str
            Dataset preset to retrain.
        model : Literal["mlp", "logistic"]
            Model family to fit.
        test_size : float
            Hold-out fraction.
        seed : int
            Reproducibility seed.

        Returns
        -------
        RetrainResult
            The training result plus feedback consumption counts.

        Raises
        ------
        InvalidInputError
            If the preset is unknown or feedback is below the threshold.
        ServiceUnavailableError
            If retraining is disabled or the feedback store is missing.
        """

        self._validate_preset(preset)
        if not feedback_settings.RETRAIN_ENABLED:
            raise ServiceUnavailableError(
                "Feedback-driven retraining is disabled on this deployment."
            )
        if self.feedback_store is None:
            raise ServiceUnavailableError(
                "Feedback store is not configured on this service."
            )

        pending = self.feedback_store.list_pending(preset)
        if len(pending) < feedback_settings.RETRAIN_THRESHOLD:
            raise InvalidInputError(
                f"Not enough pending feedback for '{preset}' "
                f"({len(pending)} < {feedback_settings.RETRAIN_THRESHOLD})."
            )

        dataset_path, target = self._resolve_dataset(preset, None, None)
        augmented_path = self._write_augmented_dataset(
            preset, dataset_path, target, pending
        )

        try:
            result = self.train(
                preset=preset,
                dataset=str(augmented_path),
                target=target,
                model=model,
                test_size=test_size,
                seed=seed,
            )
        except InvalidInputError:
            raise
        except Exception as error:
            raise ServiceUnavailableError(
                f"Feedback retrain failed: {error}"
            ) from error

        consumed = self.feedback_store.mark_consumed([record.id for record in pending])
        remaining = self.feedback_store.count_pending(preset)
        logger.info(
            "Retrained %s from feedback: consumed=%d pending=%d artifact=%s",
            preset,
            consumed,
            remaining,
            result.model_path,
        )
        return RetrainResult(
            train=result,
            feedback_consumed=consumed,
            pending_remaining=remaining,
        )

    def _validate_preset(self, preset: str) -> None:
        """Raise if ``preset`` is not a known dataset preset."""
        if preset not in PRESETS:
            raise InvalidInputError(
                f"Unknown preset '{preset}'. Choose from {sorted(PRESETS)}."
            )

    def _write_augmented_dataset(
        self,
        preset: str,
        dataset_path: Path,
        target: str,
        pending: list[FeedbackRecord],
    ) -> Path:
        """
        Write a temp CSV of the base dataset augmented with feedback rows.

        The base rows are normalized to the pipeline column convention,
        then one row per pending feedback record is appended using its
        stored features plus the confirmed label in the target column.

        Parameters
        ----------
        preset : str
            Dataset preset.
        dataset_path : Path
            Path to the base dataset CSV.
        target : str
            Target column name.
        pending : list[FeedbackRecord]
            Pending feedback records to append.

        Returns
        -------
        Path
            Path to the written augmented CSV.

        Raises
        ------
        InvalidInputError
            If the base dataset cannot be read.
        """

        try:
            base = pd.read_csv(dataset_path)
        except (OSError, ValueError) as error:
            raise InvalidInputError(f"Could not read base dataset: {error}") from error
        base = _normalize_columns(base)
        target_token = _normalize_token(target)

        rows: list[dict[str, object]] = []
        for record in pending:
            row: dict[str, object] = dict(record.features)
            row[target_token] = record.confirmed_label
            rows.append(row)
        feedback_frame = pd.DataFrame(rows)

        columns = list(base.columns)
        for column in feedback_frame.columns:
            if column not in columns:
                columns.append(column)
        augmented = pd.concat(
            [base, feedback_frame.reindex(columns=columns)], ignore_index=True
        )

        out_dir = self.artifacts_dir / preset
        out_dir.mkdir(parents=True, exist_ok=True)
        augmented_path = out_dir / "feedback_augmented.csv"
        augmented.to_csv(augmented_path, index=False)
        logger.info(
            "Wrote augmented dataset with %d base + %d feedback rows: %s",
            base.shape[0],
            len(pending),
            augmented_path,
        )
        return augmented_path

    def _train_federated(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        test_x: pd.DataFrame,
        test_y: pd.Series,
        model_name: str,
        clients: int,
        rounds: int,
        seed: int,
        differential_privacy: bool = False,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        privacy_delta: float = 1e-5,
        secure_aggregation: bool = False,
    ) -> tuple[BaseModel, dict[str, Any]]:
        """
        Train through the synchronous FedAvg path and return model + metrics.

        When differential privacy is requested a torch-backed
        ``TorchMLPClassifier`` is used per client (Opacus requires a
        ``torch.nn.Module``), the per-round epsilons are collected, and
        a membership-inference audit plus leakage check are appended to
        the federated metrics block.

        Parameters
        ----------
        train_x : pd.DataFrame
            Training features.
        train_y : pd.Series
            Training labels.
        test_x : pd.DataFrame
            Hold-out features (global evaluator / MIA audit).
        test_y : pd.Series
            Hold-out labels (global evaluator / MIA audit).
        model_name : str
            Model family name (only ``"mlp"`` is supported).
        clients : int
            Number of simulated hospital clients.
        rounds : int
            Number of federated rounds.
        seed : int
            Reproducibility seed.
        differential_privacy : bool
            Apply Opacus DP-SGD to local steps.
        noise_multiplier : float
            DP-SGD noise multiplier.
        max_grad_norm : float
            DP-SGD gradient clipping norm.
        privacy_delta : float
            Privacy delta for the epsilon audit.
        secure_aggregation : bool
            Mask updates with the pairwise one-time-pad aggregator.

        Returns
        -------
        tuple[BaseModel, dict[str, Any]]
            Aggregated global model and federated metrics (including a
            ``"privacy"`` block when differential privacy is active).
        """

        train_n = train_x.to_numpy(dtype=np.float64)
        labels_n = train_y.to_numpy()
        test_n = test_x.to_numpy(dtype=np.float64)
        test_labels_n = test_y.to_numpy()

        n_features = train_n.shape[1]
        n_classes = len(np.unique(labels_n))

        def make_client_model() -> BaseModel:
            if differential_privacy:
                return TorchMLPClassifier(
                    n_features=n_features,
                    n_classes=n_classes,
                    seed=seed,
                )
            return TabularClassifier(model_name=model_name)

        privacy_config = PrivacyConfig(
            enabled=differential_privacy,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            delta=privacy_delta,
        )

        shards = _partition_shards(train_n, labels_n, clients, seed)
        member_x = np.concatenate([shard for shard, _ in shards], axis=0)
        federated_clients = [
            FederatedClient(
                make_client_model,
                shard_x,
                shard_y,
                test_n,
                test_labels_n,
                privacy=privacy_config,
            )
            for shard_x, shard_y in shards
        ]
        evaluator = make_global_evaluator(make_client_model, test_n, test_labels_n)
        server = FedAvgServer(
            clients=federated_clients,
            num_rounds=rounds,
            evaluate_fn=evaluator,
            secure_aggregation=secure_aggregation,
        ).run()

        global_model = make_client_model()
        global_model.set_parameters(server.global_parameters)

        fed_metrics = server.metrics.to_dict()
        if differential_privacy:
            epsilon = server.max_epsilon or 0.0
            proba_member = global_model.predict_proba(member_x)[:, 1]
            proba_holdout = global_model.predict_proba(test_n)[:, 1]
            mia_auroc = membership_inference_auroc(proba_member, proba_holdout)

            # ── Measure data leakage from actual federation payloads ──
            from federated.privacy import inspect_federation_payloads

            payload_inspection = None
            if server._payload_inspection:
                payload_inspection = server._payload_inspection
                leakage_rate = payload_inspection["leakage_rate"]
            else:
                # Fallback: no secure aggregation → no masked payloads to
                # inspect; use the raw client updates as evidence.
                raw_updates = [
                    client.get_parameters({}) for client in federated_clients
                ]
                payload_inspection = inspect_federation_payloads(
                    raw_updates,
                    feature_names=list(train_x.columns),
                )
                leakage_rate = payload_inspection["leakage_rate"]

            mia_counts = {
                "train_members": len(member_x),
                "holdout_nonmembers": len(test_n),
            }

            fed_metrics["privacy"] = privacy_metrics_summary(
                epsilon=epsilon,
                delta=privacy_delta,
                mia_auroc=mia_auroc,
                leakage_rate=leakage_rate,
                num_samples=len(train_n),
                epsilon_target=PrivacyConfig().epsilon_target,
                secure_aggregation=secure_aggregation,
                per_round_epsilons=server.per_round_epsilons or None,
                epsilon_composition_method=(
                    "naive_sum_upper_bound"
                    if len(server.per_round_epsilons) > 1
                    else "single_round"
                ),
                mia_sample_counts=mia_counts,
                payload_inspection=payload_inspection,
            )
        if hasattr(global_model, "_feature_names"):
            global_model._feature_names = list(train_x.columns)
        return global_model, fed_metrics

    def _train_distributed(
        self,
        preset: str,
        clients: int,
        rounds: int,
        seed: int,
        differential_privacy: bool = False,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        privacy_delta: float = 1e-5,
        secure_aggregation: bool = False,
        tls_enabled: bool = False,
        tls_ca_cert: str | None = None,
        tls_server_cert: str | None = None,
        tls_server_key: str | None = None,
        tls_client_cert: str | None = None,
        tls_client_key: str | None = None,
    ) -> tuple[BaseModel, dict[str, Any]]:
        """
        Train through the distributed Flower gRPC deployment.

        Each hospital runs as its own subprocess (one per ``clients``),
        loading its own local data slice from the federation hospitals
        directory, connecting to a Flower server process, and exchanging
        weights over gRPC. The run is recorded in the SQLite model
        registry and the global model artifact is loaded back.

        Parameters
        ----------
        preset : str
            Dataset preset to federate.
        clients : int
            Number of hospital client processes.
        rounds : int
            Number of federated rounds.
        seed : int
            Reproducibility seed.
        differential_privacy : bool
            Apply Opacus DP-SGD on each hospital client.
        noise_multiplier : float
            DP-SGD noise multiplier.
        max_grad_norm : float
            DP-SGD gradient clipping norm.
        privacy_delta : float
            Privacy delta for the epsilon audit.
        secure_aggregation : bool
            Mask client updates with the pairwise one-time-pad aggregator.

        Returns
        -------
        tuple[BaseModel, dict[str, Any]]
            Loaded global model and federated metrics from the registry.

        Raises
        ------
        InvalidInputError
            If the distributed run fails or registers no model.
        """

        if preset is None:
            raise InvalidInputError("Distributed federation requires a 'preset'.")

        env = os.environ.copy()
        env["DATASET_DIR"] = str(self.dataset_dir)
        artifacts_root = Path(self.artifacts_dir)
        # Hospital slices live under <backend>/data/hospitals (single
        # source of truth shared with the federated CLI default
        # FED_HOSPITALS_DIR). Fall back to the legacy sibling only when
        # the standard location is absent (non-repo deployments).
        hospitals_dir = Path("data/hospitals")
        if not hospitals_dir.is_dir():
            hospitals_dir = artifacts_root.parent / "hospitals"
        env["FED_HOSPITALS_DIR"] = str(hospitals_dir)
        env["FED_REGISTRY_PATH"] = str(artifacts_root / "federation.db")
        env["FED_ARTIFACTS_DIR"] = str(artifacts_root)

        command = [
            sys.executable,
            "-m",
            "federated",
            "run",
            "--preset",
            preset,
            "--hospitals",
            str(clients),
            "--rounds",
            str(rounds),
            "--address",
            federation_settings.SERVER_ADDRESS,
            "--seed",
            str(seed),
        ]
        if secure_aggregation:
            command.append("--secure-aggregation")
        if differential_privacy:
            command += [
                "--differential-privacy",
                "--noise-multiplier",
                str(noise_multiplier),
                "--max-grad-norm",
                str(max_grad_norm),
                "--privacy-delta",
                str(privacy_delta),
            ]
        if tls_enabled:
            command.append("--tls-enabled")
            if tls_ca_cert:
                command += ["--tls-ca-cert", tls_ca_cert]
            if tls_server_cert:
                command += ["--tls-server-cert", tls_server_cert]
            if tls_server_key:
                command += ["--tls-server-key", tls_server_key]
            if tls_client_cert:
                command += ["--tls-client-cert", tls_client_cert]
            if tls_client_key:
                command += ["--tls-client-key", tls_client_key]

        logger.info("Launching distributed federation: %s", " ".join(command))
        try:
            subprocess.run(command, capture_output=True, text=True, env=env, check=True)
        except subprocess.CalledProcessError as error:
            logger.error(
                "Distributed federation failed: %s\n%s",
                error,
                error.stderr[-4000:] if error.stderr else "",
            )
            raise InvalidInputError(
                f"Distributed federation failed: {error}"
            ) from error

        registry = ModelRegistry(env["FED_REGISTRY_PATH"])
        try:
            latest = registry.latest_model(preset)
        finally:
            registry.close()
        if latest is None:
            raise InvalidInputError(
                f"Distributed federation registered no model for '{preset}'."
            )

        try:
            global_model = TabularClassifier.load(latest["model_path"])
        except ModelLoadError:
            global_model = TorchMLPClassifier.load(latest["model_path"])

        fed_metrics = {
            "distributed": True,
            "run_id": latest["run_id"],
            "version": latest["version"],
            "n_hospitals": clients,
            "n_rounds": rounds,
            "model_path": latest["model_path"],
            "accuracy": latest["accuracy"],
            "roc_auc": latest["roc_auc"],
            "epsilon": latest["epsilon"],
        }
        logger.info(
            "Distributed federation complete: run=%s version=%d artifact=%s",
            latest["run_id"],
            latest["version"],
            latest["model_path"],
        )
        if hasattr(global_model, "_feature_names"):
            from federated.hospitals import PRESETS
            from preprocessing.loader import load_classification_frame

            dataset_path = self.dataset_dir / PRESETS[preset][0]
            features, _, _ = load_classification_frame(dataset_path, PRESETS[preset][1])
            global_model._feature_names = list(features.columns)
        return global_model, fed_metrics

    def predict(self, features: Mapping[str, float]) -> PredictionResult:
        """
        Predict the class and probabilities for a single feature row.

        Parameters
        ----------
        features : Mapping[str, float]
            Feature values keyed by column name.

        Returns
        -------
        PredictionResult
            Predicted class, probabilities, and confidence.

        Raises
        ------
        ServiceUnavailableError
            If no model is configured.
        InvalidInputError
            If the features cannot be aligned to the model.
        """

        if self.model is None:
            raise ServiceUnavailableError(
                "No prediction model is configured (set API_MODEL_PATH)."
            )
        try:
            result = run_prediction(self.model, features)
        except CrewError as error:
            raise InvalidInputError(str(error)) from error
        logger.info("API prediction: %s", result.predicted_class)
        return result

    def retrieve(self, query: str, top_k: int | None = None) -> list[EvidenceItem]:
        """
        Retrieve evidence chunks for a query.

        Parameters
        ----------
        query : str
            Query text.
        top_k : int | None
            Number of results; defaults to the orchestrator RAG top-k.

        Returns
        -------
        list[EvidenceItem]
            Retrieved evidence ordered by descending score.

        Raises
        ------
        ServiceUnavailableError
            If no RAG pipeline is configured.
        InvalidInputError
            If the query is empty or the corpus is empty.
        """

        if self.rag_pipeline is None:
            raise ServiceUnavailableError("No retrieval pipeline is configured.")
        try:
            return retrieve_evidence(self.rag_pipeline, query, top_k=top_k)
        except CrewError as error:
            raise InvalidInputError(str(error)) from error

    def model_info(self) -> dict[str, Any]:
        """
        Describe the configured prediction models.

        Returns
        -------
        dict[str, Any]
            ``{available, model_type, model_name, classes, feature_names,
            preset}`` keyed for the ``ModelInfo`` schema.
        """

        tabular = self.model
        image = self.image_model
        model_type = None
        if tabular is not None and image is not None:
            model_type = "tabular_and_image"
        elif tabular is not None:
            model_type = "tabular"
        elif image is not None:
            model_type = "image"

        primary = tabular if tabular is not None else image
        classes = None
        feature_names = None
        if primary is not None:
            classes = [str(label) for label in primary.classes_]
            if tabular is not None:
                feature_names = list(tabular.feature_names or [])
        model_name = (
            tabular.model_name
            if tabular is not None
            else ("image-cnn" if image is not None else None)
        )
        return {
            "available": primary is not None,
            "model_type": model_type,
            "model_name": model_name,
            "classes": classes,
            "feature_names": feature_names,
            "preset": self.active_preset if tabular is not None else None,
        }

    def presets_info(self) -> list[dict[str, Any]]:
        """
        Describe the named dataset presets and their feature schemas.

        A preset is ``available`` only when a trained artifact exists
        under ``artifacts_dir/<preset>/global_model.joblib``; its feature
        schema and classes are then read from that artifact so the
        dashboard can render exactly the fields the model expects.

        Returns
        -------
        list[dict[str, Any]]
            One ``PresetInfo``-shaped dict per preset, ordered by name.
        """

        infos: list[dict[str, Any]] = []
        for name, (file_name, target) in sorted(PRESETS.items()):
            info: dict[str, Any] = {
                "name": name,
                "dataset": file_name,
                "target": target,
                "available": False,
                "feature_names": None,
                "classes": None,
            }
            artifact = self.artifacts_dir / name / "global_model.joblib"
            if artifact.exists():
                try:
                    preset_model = TabularClassifier.load(artifact)
                except ModelLoadError as error:
                    logger.warning(
                        "Could not read preset artifact %s: %s", artifact, error
                    )
                    preset_model = None
                if preset_model is not None and preset_model.is_fitted:
                    info["available"] = True
                    info["feature_names"] = list(preset_model.feature_names or [])
                    info["classes"] = [str(label) for label in preset_model.classes_]
            infos.append(info)
        logger.info("Reported %d presets", len(infos))
        return infos

    def _registry_path(self) -> Path | None:
        """
        Resolve the federation registry database path.

        The registry lives next to the artifacts directory
        (``<artifacts_dir>/federation.db``), matching where
        :meth:`_train_distributed` writes it. Returns ``None`` when the
        database does not exist.

        Returns
        -------
        Path | None
            The registry database path, or ``None`` if absent.
        """

        path = self.artifacts_dir / "federation.db"
        return path if path.is_file() else None

    def federation_status(self) -> dict[str, Any]:
        """
        Summarize the federation registry for the dashboard.

        Returns
        -------
        dict[str, Any]
            ``{registry_path, n_runs, n_models, presets}`` where each
            preset entry carries the preset metadata plus the latest
            registered model row (or ``None``).
        """

        path = self._registry_path()
        if path is None:
            return {
                "registry_path": None,
                "n_runs": 0,
                "n_models": 0,
                "presets": [],
            }
        registry = ModelRegistry(path)
        try:
            runs = registry.list_runs()
            models = registry.list_models()
            run_by_id = {run["run_id"]: run for run in runs}
            presets: list[dict[str, Any]] = []
            for name, (file_name, target) in sorted(PRESETS.items()):
                latest = registry.latest_model(name)
                if latest is not None:
                    run = run_by_id.get(latest["run_id"], {})
                    latest = {
                        **latest,
                        "secure_aggregation": bool(run.get("secure_aggregation", 0)),
                        "differential_privacy": bool(
                            run.get("differential_privacy", 0)
                        ),
                    }
                presets.append(
                    {
                        "name": name,
                        "dataset": file_name,
                        "target": target,
                        "available": latest is not None,
                        "feature_names": None,
                        "classes": None,
                        "latest_model": latest,
                    }
                )
            return {
                "registry_path": str(path),
                "n_runs": len(runs),
                "n_models": len(models),
                "presets": presets,
            }
        finally:
            registry.close()

    def federation_runs(self, preset: str | None = None) -> list[dict[str, Any]]:
        """
        List federation runs, newest first.

        Parameters
        ----------
        preset : str | None
            Restrict to a preset when given.

        Returns
        -------
        list[dict[str, Any]]
            Run rows ordered by creation time (descending).
        """

        path = self._registry_path()
        if path is None:
            return []
        registry = ModelRegistry(path)
        try:
            return registry.list_runs(preset)
        finally:
            registry.close()

    def federation_models(self, preset: str | None = None) -> list[dict[str, Any]]:
        """
        List registered global models, newest first.

        Parameters
        ----------
        preset : str | None
            Restrict to a preset when given.

        Returns
        -------
        list[dict[str, Any]]
            Model rows ordered by registration time (descending).
        """

        path = self._registry_path()
        if path is None:
            return []
        registry = ModelRegistry(path)
        try:
            return registry.list_models(preset)
        finally:
            registry.close()

    def federation_rounds(self, run_id: str) -> list[dict[str, Any]]:
        """
        Return the per-round metrics of a specific run.

        Parameters
        ----------
        run_id : str
            The run id.

        Returns
        -------
        list[dict[str, Any]]
            Round rows ordered by round index; empty when the registry is
            absent or the run is unknown.
        """

        path = self._registry_path()
        if path is None:
            return []
        registry = ModelRegistry(path)
        try:
            return registry.run_rounds(run_id)
        finally:
            registry.close()

    def analyze_image(
        self,
        patient: PatientInfo,
        image: bytes,
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
    ) -> ClinicalReport:
        """
        Preprocess an uploaded image and run the clinical crew on it.

        The crew always runs through the CrewAI agentic path when
        ``CREW_LLM_API_KEY`` is configured (preferred); it falls back to
        the deterministic pipeline only when the LLM is unavailable or its
        output cannot be parsed.

        Parameters
        ----------
        patient : PatientInfo
            Patient context.
        image : bytes
            Raw image file bytes (PNG / JPEG).
        markers : Mapping[str, float] | None
            Optional raw clinical markers for the risk assessment.
        recommendations : list[str] | None
            Optional recommendation strings.

        Returns
        -------
        ClinicalReport
            The assembled structured report.

        Raises
        ------
        ServiceUnavailableError
            If no image model is configured.
        InvalidInputError
            If the image cannot be preprocessed.
        """

        if self.image_model is None:
            raise ServiceUnavailableError(
                "No image model is configured (set API_IMAGE_MODEL_PATH)."
            )
        try:
            result = ImagePipeline().transform(image)
        except Exception as error:
            raise InvalidInputError(f"Image preprocessing failed: {error}") from error

        crew = ClinicalCrew(
            patient=patient,
            input_type="image",
            image_model=self.image_model,
            image=result.image,
            rag_pipeline=self.rag_pipeline,
            markers=markers,
            recommendations=recommendations,
        )
        try:
            report = crew.run()
        except CrewError as error:
            raise InvalidInputError(str(error)) from error
        logger.info("API image analysis complete for patient %s", patient.id)
        return report

    def analyze_csv(
        self,
        patient: PatientInfo,
        csv: bytes,
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
        input_type: str = "csv",
    ) -> ClinicalReport:
        """
        Run the clinical crew on the first row of an uploaded CSV.

        Preprocessing stays in ``preprocessing.csv.CSVPipeline`` (the
        reusable inference path, ADR-003); the dashboard never parses or
        transforms the CSV itself. The first data row is aligned to the
        served model's ``feature_names`` and fed through the standard
        analysis pipeline.

        Parameters
        ----------
        patient : PatientInfo
            Patient context.
        csv : bytes
            Raw UTF-8 CSV bytes.
        markers : Mapping[str, float] | None
            Optional raw clinical markers for the risk assessment.
        recommendations : list[str] | None
            Optional recommendation strings.
        input_type : str
            Data modality analyzed (default ``"csv"``).

        Returns
        -------
        ClinicalReport
            The assembled structured report.

        Raises
        ------
        ServiceUnavailableError
            If no tabular model is configured.
        InvalidInputError
            If the CSV cannot be preprocessed or is missing the columns
            the served model requires.
        """

        if self.model is None:
            raise ServiceUnavailableError(
                "No tabular model is configured for CSV analysis (set "
                "API_MODEL_PATH or train one)."
            )
        names = list(self.model.feature_names or [])
        if not names:
            raise InvalidInputError(
                "The served model has no recorded feature columns; CSV "
                "analysis cannot align the data."
            )
        try:
            pipeline = CSVPipeline(
                scaler_params=self.model.scaler_params,
                encoder_params=getattr(self.model, "encoder_params", None),
            )
            result = pipeline.run(csv)
        except Exception as error:
            raise InvalidInputError(f"CSV preprocessing failed: {error}") from error
        frame = result.dataframe
        if frame.shape[0] == 0:
            raise InvalidInputError("The uploaded CSV contains no data rows.")
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise InvalidInputError(
                "The uploaded CSV is missing columns required by the model: "
                f"{', '.join(missing)}. Expected: {', '.join(names)}."
            )
        row = frame.iloc[0]
        features = {name: float(row[name]) for name in names}
        effective_markers = markers if markers is not None else features
        logger.info(
            "API CSV analysis for patient %s (row 0 of %d, %d features)",
            patient.id,
            frame.shape[0],
            len(features),
        )
        return self.analyze(
            patient=patient,
            features=features,
            markers=effective_markers,
            recommendations=recommendations,
            input_type=input_type,
            preprocessed=True,
        )

    def analyze(
        self,
        patient: PatientInfo,
        features: Mapping[str, float],
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
        input_type: str = "csv",
        preprocessed: bool = False,
    ) -> ClinicalReport:
        """
        Run the clinical crew and return the report.

        The crew always runs through the CrewAI agentic path when
        ``CREW_LLM_API_KEY`` is configured (preferred); it falls back to
        the deterministic pipeline only when the LLM is unavailable or its
        output cannot be parsed.

        Parameters
        ----------
        patient : PatientInfo
            Patient context.
        features : Mapping[str, float]
            Preprocessed feature row for the prediction step.
        markers : Mapping[str, float] | None
            Optional raw clinical markers for the risk assessment.
        recommendations : list[str] | None
            Optional recommendation strings.
        input_type : str
            Data modality analyzed (``"csv"`` / ``"image"`` / ...).
        preprocessed : bool
            True when ``features`` were already transformed by the
            training pipeline (CSV path); False applies the model's
            persisted scaler to raw feature values.

        Returns
        -------
        ClinicalReport
            The assembled structured report.

        Raises
        ------
        InvalidInputError
            If the analysis inputs are inconsistent.
        """

        crew = ClinicalCrew(
            patient=patient,
            input_type=input_type,
            model=self.model,
            features=features,
            rag_pipeline=self.rag_pipeline,
            markers=markers,
            recommendations=recommendations,
            preprocessed=preprocessed,
            disease=self.active_preset,
        )
        try:
            report = crew.run()
        except CrewError as error:
            raise InvalidInputError(str(error)) from error

        if report.risk and self.risk_history_store:
            self._persist_risk_history(
                patient_id=patient.id,
                preset=self.active_preset or "unknown",
                report=report,
                markers=markers,
            )

        logger.info("API analysis complete for patient %s", patient.id)
        return report

    def _persist_risk_history(
        self,
        patient_id: str,
        preset: str,
        report: ClinicalReport,
        markers: Mapping[str, float] | None,
    ) -> None:
        """Persist the risk assessment from a clinical report."""
        if not self.risk_history_store or not report.risk:
            return
        risk = report.risk
        prediction = report.prediction.predicted_class if report.prediction else None
        confidence = report.prediction.confidence if report.prediction else None
        try:
            self.risk_history_store.add(
                patient_id=patient_id,
                preset=preset,
                risk_score=risk.risk_score,
                risk_level=risk.risk_level,
                prediction=int(prediction) if prediction is not None else None,
                confidence=confidence,
                markers=dict(markers) if markers else None,
            )
        except RiskHistoryStoreError as error:
            logger.warning(
                "Failed to persist risk history for %s: %s", patient_id, error
            )


__all__ = [
    "DEFAULT_CORPUS",
    "PRESETS",
    "AnalysisService",
    "TrainResult",
    "build_rag_pipeline",
    "load_image_model",
    "load_predictive_model",
    "prepare_tabular_data",
]
