"""
Service layer for the FastAPI module.

All business logic lives here; routes only validate and delegate. The
``AnalysisService`` orchestrates the prediction model, the CrewAI
clinical crew, and the RAG pipeline, translating domain exceptions into
typed ``APIError`` subclasses at the service boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from CrewAI.orchestrator import ClinicalCrew
from CrewAI.orchestrator.exceptions import CrewError
from CrewAI.orchestrator.schemas import (
    ClinicalReport,
    EvidenceItem,
    PatientInfo,
    PredictionResult,
)
from CrewAI.orchestrator.services import retrieve_evidence, run_prediction
from models import ModelLoadError, TabularClassifier
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
class AnalysisService:
    """
    Facade exposing the clinical analysis pipeline to the API.

    Parameters
    ----------
    model : TabularClassifier | None
        Fitted model for the prediction step, or None to skip prediction.
    rag_pipeline : RAGPipeline | None
        Ingested retrieval pipeline, or None to skip evidence retrieval.
    """

    model: TabularClassifier | None = None
    rag_pipeline: RAGPipeline | None = None

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
        return cls(model=model, rag_pipeline=rag_pipeline)

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
    "AnalysisService",
    "build_rag_pipeline",
    "load_predictive_model",
]
