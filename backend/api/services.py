"""
Service layer for the FastAPI module.

All business logic lives here; routes only validate and delegate. The
``AnalysisService`` orchestrates the prediction model, the CrewAI
clinical crew, and the RAG pipeline, translating domain exceptions into
typed ``APIError`` subclasses at the service boundary.
"""

from __future__ import annotations

import os

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
from models import ModelLoadError, TabularClassifier
from preprocessing.csv import CSVPipeline
from preprocessing.logger import get_logger
from rag import RAGPipeline
from rag.documents import Document

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

#: Named dataset presets mapping a name to ``(file name, target column)``.
PRESETS: dict[str, tuple[str, str]] = {
    "diabetes": ("diabetes.csv", "Outcome"),
    "heart": ("heart_disease_uci.csv", "num"),
    "kidney": ("kidney_disease.csv", "classification"),
    "sepsis": ("sepsis_icu_synthetic.csv", "sepsis_label"),
}


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
) -> tuple[pd.DataFrame, pd.Series]:
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
    tuple[pd.DataFrame, pd.Series]
        Engineered feature frame and integer-encoded label series aligned
        by index.

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

    result = CSVPipeline(input_columns=tuple(feature_frame.columns)).run(feature_frame)
    features = result.dataframe
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise InvalidInputError("Pipeline produced no usable features.")

    labels = y_raw.loc[features.index]
    if pd.api.types.is_string_dtype(labels):
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
    return features, labels


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


def build_rag_pipeline(corpus_dir: str | Path | None = None) -> RAGPipeline:
    """
    Build an ingested RAG pipeline from a corpus directory or a default corpus.

    Parameters
    ----------
    corpus_dir : str | Path | None
        Directory of ``.txt`` / ``.md`` documents. When None (or empty),
        a small built-in medical corpus is ingested.

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
        documents = [
            Document(id=path.name, text=path.read_text(encoding="utf-8"))
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        ]
        if not documents:
            raise ServiceUnavailableError(
                f"No .txt/.md documents found under {directory}."
            )
        pipeline.ingest_documents(documents)
        logger.info("Ingested %d documents from %s", len(documents), directory)
    else:
        pipeline.ingest_texts(DEFAULT_CORPUS)
        logger.info(
            "Ingested the built-in medical corpus (%d texts)", len(DEFAULT_CORPUS)
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
class AnalysisService:
    """
    Facade exposing the clinical analysis pipeline to the API.

    Parameters
    ----------
    model : TabularClassifier | None
        Fitted model for the prediction step, or None to skip prediction.
    rag_pipeline : RAGPipeline | None
        Ingested retrieval pipeline, or None to skip evidence retrieval.
    artifacts_dir : Path
        Base directory for model artifacts written by :meth:`train`.
    dataset_dir : Path
        Base directory for preset datasets resolved by :meth:`train`.
    """

    model: TabularClassifier | None = None
    rag_pipeline: RAGPipeline | None = None
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    dataset_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("DATASET_DIR", "."))
    )

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
        rag_pipeline = build_rag_pipeline(cfg.CORPUS_DIR or None)
        dataset_dir = Path(cfg.DATASET_DIR or os.environ.get("DATASET_DIR", "."))
        return cls(
            model=model,
            rag_pipeline=rag_pipeline,
            artifacts_dir=Path(cfg.ARTIFACTS_DIR),
            dataset_dir=dataset_dir,
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
        clients: int = 3,
        rounds: int = 3,
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
        clients : int
            Number of simulated hospital clients (federated path).
        rounds : int
            Number of federated rounds (federated path).

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
            features, labels = prepare_tabular_data(dataset_path, target, max_rows)
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
                fitted, fed_metrics = self._train_federated(
                    train_x, train_y, test_x, test_y, model, clients, rounds, seed
                )
            else:
                fitted = TabularClassifier(model_name=model).fit(train_x, train_y)
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
        """Resolve a (preset or explicit) dataset path and target column."""
        if preset is not None:
            if preset not in PRESETS:
                raise InvalidInputError(
                    f"Unknown preset '{preset}'. Choose from {sorted(PRESETS)}."
                )
            file_name, preset_target = PRESETS[preset]
            return self.dataset_dir / file_name, target or preset_target
        if dataset is None or target is None:
            raise InvalidInputError(
                "Provide a 'preset' or both 'dataset' and 'target'."
            )
        return Path(dataset), target

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
    ) -> tuple[TabularClassifier, dict[str, Any]]:
        """Train through the synchronous FedAvg path and return model + metrics."""
        train_n = train_x.to_numpy(dtype=np.float64)
        labels_n = train_y.to_numpy()
        test_n = test_x.to_numpy(dtype=np.float64)
        test_labels_n = test_y.to_numpy()

        shards = _partition_shards(train_n, labels_n, clients, seed)
        federated_clients = [
            FederatedClient(
                lambda: TabularClassifier(model_name=model_name),
                shard_x,
                shard_y,
                test_n,
                test_labels_n,
            )
            for shard_x, shard_y in shards
        ]
        evaluator = make_global_evaluator(
            lambda: TabularClassifier(model_name=model_name), test_n, test_labels_n
        )
        server = FedAvgServer(
            clients=federated_clients, num_rounds=rounds, evaluate_fn=evaluator
        ).run()

        global_model = TabularClassifier(model_name=model_name)
        global_model.set_parameters(server.global_parameters)
        return global_model, server.metrics.to_dict()

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

    def analyze(
        self,
        patient: PatientInfo,
        features: Mapping[str, float],
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
        input_type: str = "csv",
    ) -> ClinicalReport:
        """
        Run the deterministic clinical crew and return the report.

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
        )
        try:
            report = crew.run_analysis()
        except CrewError as error:
            raise InvalidInputError(str(error)) from error
        logger.info("API analysis complete for patient %s", patient.id)
        return report


__all__ = [
    "DEFAULT_CORPUS",
    "PRESETS",
    "AnalysisService",
    "TrainResult",
    "build_rag_pipeline",
    "load_predictive_model",
    "prepare_tabular_data",
]
