"""
Deterministic orchestration services.

These are the pure, LLM-free functions behind the crew's tools:
prediction from a fitted model, risk assessment from a prediction and
clinical markers, evidence retrieval from the RAG pipeline, and the
final clinical report assembly. Keeping them framework-free lets every
step run and be tested without an LLM API key (ADR-008).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from models import ImageClassifier
from models.csv.tabular import TabularClassifier
from preprocessing.csv.scaler import CSVScaler
from preprocessing.logger import get_logger
from rag import RAGPipeline
from rag.exceptions import EmptyCorpusError, EmptyQueryError

from .config import settings
from .exceptions import (
    PredictionToolError,
    ReportError,
    RetrievalToolError,
    RiskToolError,
)
from .schemas import (
    ClinicalReport,
    EvidenceItem,
    PatientInfo,
    PredictionResult,
    RiskResult,
)

logger = get_logger(__name__)

MONITORING_SCHEDULES: dict[str, list[dict[str, str]]] = {
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


def run_prediction(
    model: TabularClassifier,
    features: Mapping[str, float],
    preprocessed: bool = False,
) -> PredictionResult:
    """
    Predict the class and probabilities for a single feature row.

    Parameters
    ----------
    model : TabularClassifier
        A fitted classifier.
    features : Mapping[str, float]
        Feature values. Ordered by ``model.feature_names`` when the model
        captured column names during fit, otherwise by insertion order.
    preprocessed : bool
        True when the features were already transformed by the training
        pipeline (e.g. the CSV inference path); False applies the model's
        persisted scaler to raw values (manual entry, n8n structured
        input).

    Returns
    -------
    PredictionResult
        Predicted class, per-class probabilities, and confidence.

    Raises
    ------
    PredictionToolError
        If the model is unfitted or the features cannot be aligned.
    """

    names = model.feature_names
    keys = list(features) if names is None else names
    if names is not None and not all(name in features for name in names):
        missing = [name for name in names if name not in features]
        raise PredictionToolError(f"Missing feature values for columns: {missing}.")
    try:
        row = np.array([float(features[key]) for key in keys], dtype=np.float64)
        if not preprocessed and getattr(model, "scaler_params", None) is not None:
            scaled = CSVScaler.from_params(model.scaler_params).transform(
                pd.DataFrame([features])
            )[0]
            row = np.array(
                [float(scaled.iloc[0][key]) for key in keys], dtype=np.float64
            )
        probabilities = model.predict_proba(row.reshape(1, -1))[0]
    except Exception as error:
        raise PredictionToolError(f"Prediction failed: {error}") from error

    classes = [str(label) for label in model.classes_]
    probability_map = {
        label: float(probability)
        for label, probability in zip(classes, probabilities, strict=True)
    }
    predicted = classes[int(np.argmax(probabilities))]
    result = PredictionResult(
        predicted_class=predicted,
        probabilities=probability_map,
        confidence=float(np.max(probabilities)),
        model_name=model.model_name,
    )
    logger.info("Predicted %s with confidence %.4f", predicted, result.confidence)
    return result


def run_image_prediction(
    image_model: ImageClassifier, image: np.ndarray
) -> PredictionResult:
    """
    Predict the class and probabilities for a single preprocessed image.

    Parameters
    ----------
    image_model : ImageClassifier
        A fitted CNN image classifier.
    image : np.ndarray
        Preprocessed image array (``(H, W, C)`` channels-last float32).

    Returns
    -------
    PredictionResult
        Predicted class, per-class probabilities, and confidence.

    Raises
    ------
    PredictionToolError
        If the model is unfitted or the image array is malformed.
    """

    array = np.asarray(image)
    if array.ndim != 3:
        raise PredictionToolError(
            f"Expected a preprocessed image with 3 dimensions (H, W, C), "
            f"got {array.ndim}."
        )
    if not image_model.is_fitted:
        raise PredictionToolError("The image model must be fitted before use.")
    try:
        probabilities = image_model.predict_proba(array[np.newaxis, ...])[0]
    except Exception as error:
        raise PredictionToolError(f"Image prediction failed: {error}") from error

    classes = [str(label) for label in image_model.classes_]
    probability_map = {
        label: float(probability)
        for label, probability in zip(classes, probabilities, strict=True)
    }
    predicted = classes[int(np.argmax(probabilities))]
    result = PredictionResult(
        predicted_class=predicted,
        probabilities=probability_map,
        confidence=float(np.max(probabilities)),
        model_name="image-cnn",
    )
    logger.info("Image predicted %s with confidence %.4f", predicted, result.confidence)
    return result


def _positive_class_probability(prediction: PredictionResult) -> float:
    """
    Return the probability of the positive (disease) class.

    The risk score must reflect the chance of the condition, not the
    model's confidence in whichever class it predicted. A confident
    prediction of the healthy class therefore scores *low* risk, while a
    confident prediction of the disease class scores *high* risk.

    For binary models the positive class is the highest class label (the
    disease outcome, conventionally ``"1"``). For multi-class models the
    most abnormal class cannot be inferred from the label alone, so the
    max-class confidence is used as a conservative fallback.

    Parameters
    ----------
    prediction : PredictionResult
        Prediction whose positive-class probability is wanted.

    Returns
    -------
    float
        Probability in ``[0, 1]`` of the positive class.
    """

    labels = [str(label) for label in prediction.probabilities]
    if len(labels) == 2 and "1" in labels:
        return float(prediction.probabilities["1"])
    return prediction.confidence


def assess_risk(
    prediction: PredictionResult,
    markers: Mapping[str, float] | None = None,
) -> RiskResult:
    """
    Score risk from prediction confidence and elevated clinical markers.

    Parameters
    ----------
    prediction : PredictionResult
        Model prediction to score.
    markers : Mapping[str, float] | None
        Optional numeric clinical markers (e.g. glucose, bmi) compared
        against ``settings.MARKER_THRESHOLDS``.

    Returns
    -------
    RiskResult
        Risk score, level, contributing factors, and monitoring plan.

    Raises
    ------
    RiskToolError
        If any marker value is not numeric.
    """

    score = _positive_class_probability(prediction)
    if score < settings.RISK_LOW_THRESHOLD:
        level = "low"
    elif score < settings.RISK_MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "high"

    factors: list[str] = []
    for marker, threshold in settings.MARKER_THRESHOLDS.items():
        if markers is None or marker not in markers:
            continue
        try:
            value = float(markers[marker])
        except (TypeError, ValueError) as error:
            raise RiskToolError(
                f"Marker '{marker}' must be numeric, got {markers[marker]!r}."
            ) from error
        if value > threshold:
            factors.append(f"Elevated {marker} ({value:.1f} > {threshold:.0f})")

    result = RiskResult(
        risk_score=round(score, 4),
        risk_level=level,
        risk_factors=factors,
        monitoring_schedule=list(
            MONITORING_SCHEDULES.get(level, MONITORING_SCHEDULES["medium"])
        ),
    )
    logger.info("Risk level %s (score %.4f)", level, score)
    return result


def retrieve_evidence(
    pipeline: RAGPipeline, query: str, top_k: int | None = None
) -> list[EvidenceItem]:
    """
    Retrieve evidence chunks for a query from a RAG pipeline.

    Parameters
    ----------
    pipeline : RAGPipeline
        Ingested retrieval pipeline.
    query : str
        Query text.
    top_k : int | None
        Number of results; defaults to ``settings.RAG_TOP_K``.

    Returns
    -------
    list[EvidenceItem]
        Retrieved evidence ordered by descending score.

    Raises
    ------
    RetrievalToolError
        If the query is empty or the corpus is empty.
    """

    limit = settings.RAG_TOP_K if top_k is None else int(top_k)
    try:
        results = pipeline.retrieve(query, top_k=limit)
    except (EmptyCorpusError, EmptyQueryError) as error:
        raise RetrievalToolError(str(error)) from error

    evidence = [
        EvidenceItem(
            document_id=result.chunk.document_id,
            source=result.chunk.source,
            score=result.score,
            text=result.chunk.text,
        )
        for result in results
    ]
    logger.info("Retrieved %d evidence items", len(evidence))
    return evidence


def assemble_clinical_report(
    patient: PatientInfo,
    input_type: str = "csv",
    prediction: PredictionResult | None = None,
    risk: RiskResult | None = None,
    evidence: list[EvidenceItem] | None = None,
    recommendations: list[str] | None = None,
    agent_metrics: Mapping[str, float] | None = None,
) -> ClinicalReport:
    """
    Assemble the final structured clinical report from tool outputs.

    Parameters
    ----------
    patient : PatientInfo
        Patient context.
    input_type : str
        Data modality analyzed (e.g. ``"csv"`` / ``"image"``).
    prediction : PredictionResult | None
        Optional model prediction.
    risk : RiskResult | None
        Optional risk assessment.
    evidence : list[EvidenceItem] | None
        Optional retrieved evidence.
    recommendations : list[str] | None
        Optional treatment / monitoring recommendations.
    agent_metrics : Mapping[str, float] | None
        Optional agent-level metrics block (e.g. from
        ``compute_agent_metrics``).

    Returns
    -------
    ClinicalReport
        The structured report.

    Raises
    ------
    ReportError
        If a prediction is present without a risk assessment (or vice
        versa), which would make the report internally inconsistent.
    """

    items = evidence or []
    if (prediction is None) != (risk is None):
        raise ReportError(
            "Prediction and risk must be provided together for a consistent report."
        )

    summary_parts = [
        f"Analysis completed for patient {patient.name or patient.id}",
        f"using {input_type} input.",
    ]
    if prediction is not None:
        summary_parts.append(
            f"Primary prediction: {prediction.predicted_class} "
            f"(confidence {prediction.confidence:.2f})."
        )
    if risk is not None:
        summary_parts.append(f"Overall risk: {risk.risk_level}.")

    report = ClinicalReport(
        patient=patient,
        input_type=input_type,
        patient_summary=" ".join(summary_parts),
        prediction=prediction,
        risk=risk,
        evidence=items,
        recommendations=list(recommendations or []),
        agent_metrics=dict(agent_metrics) if agent_metrics else None,
    )
    logger.info("Assembled clinical report for patient %s", patient.id)
    return report


__all__ = [
    "assemble_clinical_report",
    "assess_risk",
    "retrieve_evidence",
    "run_image_prediction",
    "run_prediction",
]
