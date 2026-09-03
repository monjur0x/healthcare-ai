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

# ---------------------------------------------------------------------------
# Disease / target resolver
# ---------------------------------------------------------------------------

#: Clinical context per supported dataset preset. ``labels`` maps raw
#: class values to human-readable outcomes; the positive class is the
#: disease outcome. ``rag_topic`` is the corpus topic tag used to
#: prioritize disease-relevant evidence.
DISEASE_REGISTRY: dict[str, dict[str, object]] = {
    "diabetes": {
        "disease": "diabetes",
        "positive_class": "1",
        "labels": {"0": "No Diabetes", "1": "Diabetes"},
        "rag_topic": "diabetes",
    },
    "heart": {
        "disease": "heart_disease",
        "positive_class": "1",
        "labels": {"0": "No Heart Disease", "1": "Heart Disease"},
        "rag_topic": "heart_failure",
    },
    "kidney": {
        "disease": "chronic_kidney_disease",
        "positive_class": "1",
        "labels": {"0": "No Chronic Kidney Disease", "1": "Chronic Kidney Disease"},
        "rag_topic": "chronic_kidney_disease",
    },
    "sepsis": {
        "disease": "sepsis",
        "positive_class": "1",
        "labels": {"0": "No Sepsis", "1": "Sepsis"},
        "rag_topic": "sepsis",
    },
}


def resolve_disease(preset: str | None) -> dict[str, object] | None:
    """
    Resolve a dataset preset to its clinical disease context.

    Parameters
    ----------
    preset : str | None
        Dataset preset name (e.g. ``"diabetes"``); may be ``None``.

    Returns
    -------
    dict[str, object] | None
        The registry entry, or ``None`` when the preset has no disease
        mapping (custom CSVs, image models).
    """

    if not preset:
        return None
    return DISEASE_REGISTRY.get(str(preset).strip().lower())


def enrich_prediction(
    prediction: PredictionResult,
    disease_context: dict[str, object] | None,
) -> PredictionResult:
    """
    Attach resolved disease context to a model prediction.

    Builds the structured prediction object: raw class plus human
    label, per-class probabilities split into positive (disease) and
    negative probabilities. The returned object is a copy; the input is
    not mutated.

    Parameters
    ----------
    prediction : PredictionResult
        Raw model prediction.
    disease_context : dict[str, object] | None
        Entry from :data:`DISEASE_REGISTRY`, or ``None``.

    Returns
    -------
    PredictionResult
        Prediction with ``disease``, ``predicted_label``,
        ``positive_probability``, and ``negative_probability`` filled.
    """

    if not disease_context:
        return prediction.model_copy(
            update={"predicted_label": prediction.predicted_class}
        )

    labels = disease_context["labels"]
    assert isinstance(labels, dict)
    predicted_label = str(labels.get(prediction.predicted_class, ""))
    positive_class = str(disease_context["positive_class"])
    positive_prob = (
        float(prediction.probabilities[positive_class])
        if positive_class in prediction.probabilities
        else None
    )
    negative_prob = None
    if positive_prob is not None and len(prediction.probabilities) == 2:
        negative_prob = 1.0 - positive_prob

    return prediction.model_copy(
        update={
            "disease": str(disease_context["disease"]),
            "predicted_label": predicted_label or prediction.predicted_class,
            "positive_probability": positive_prob,
            "negative_probability": negative_prob,
        }
    )


def _marker_query_terms(markers: Mapping[str, float] | None) -> list[str]:
    """
    Format elevated clinical markers as retrieval query terms.

    Only markers listed in ``settings.MARKER_THRESHOLDS`` that exceed
    their threshold contribute; non-numeric values are skipped (the
    risk-assessment path is the one that validates marker types).
    Bounded to the first five terms so the query stays retrieval-friendly.

    Parameters
    ----------
    markers : Mapping[str, float] | None
        Raw clinical markers (e.g. ``{"glucose": 210.0}``).

    Returns
    -------
    list[str]
        Terms like ``"glucose 210.0"`` for elevated markers.
    """

    if not markers:
        return []
    terms: list[str] = []
    for marker, threshold in settings.MARKER_THRESHOLDS.items():
        if marker not in markers:
            continue
        try:
            value = float(markers[marker])
        except (TypeError, ValueError):
            continue
        if value > threshold:
            terms.append(f"{marker} {value:.1f}")
    return terms[:5]


def build_disease_query(
    prediction: PredictionResult | None,
    markers: Mapping[str, float] | None = None,
) -> str:
    """
    Build a disease-specific RAG query from a prediction and markers.

    The query always carries the clinical condition name — never a raw
    class integer — so retrieval stays anchored to the predicted
    disease whether the outcome is positive or negative. Elevated
    clinical markers (e.g. ``glucose 210.0``) are appended so the
    retrieved evidence matches the patient's actual presentation
    instead of the condition name alone.

    Parameters
    ----------
    prediction : PredictionResult | None
        Enriched prediction (with ``disease`` / ``predicted_label``).
    markers : Mapping[str, float] | None
        Raw clinical markers; elevated ones become query terms.

    Returns
    -------
    str
        Query text for the evidence-retrieval step.
    """

    if prediction is None:
        return "clinical evidence and management recommendations"
    if prediction.disease:
        positive = (
            prediction.positive_probability is not None
            and prediction.negative_probability is not None
            and prediction.positive_probability >= prediction.negative_probability
        )
        if positive:
            query = (
                f"{prediction.disease} clinical guidelines diagnosis "
                f"management treatment"
            )
        else:
            query = f"{prediction.disease} prevention risk factors screening guidelines"
    else:
        query = f"clinical evidence for {prediction.predicted_label} management"
    marker_terms = _marker_query_terms(markers)
    if marker_terms:
        query = f"{query} {' '.join(marker_terms)}"
    return query


def build_rag_topic(prediction: PredictionResult | None) -> str | None:
    """
    Map an enriched prediction to its corpus topic tag.

    Parameters
    ----------
    prediction : PredictionResult | None
        Enriched prediction.

    Returns
    -------
    str | None
        Topic tag (e.g. ``"diabetes"``), or ``None`` when unknown.
    """

    if prediction is None or not prediction.disease:
        return None
    topic_by_disease = {
        str(ctx["disease"]): str(ctx["rag_topic"]) for ctx in DISEASE_REGISTRY.values()
    }
    return topic_by_disease.get(prediction.disease)


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

#: Disease-specific monitoring layered over the risk-level schedule.
#: Keyed by ``(disease, level)``; missing combinations fall back to the
#: level-only generic schedule above.
DISEASE_MONITORING: dict[tuple[str, str], list[dict[str, str]]] = {
    ("diabetes", "low"): [
        {"test": "Fasting glucose or HbA1c screening", "frequency": "Annually"},
        {"test": "Weight and BMI review", "frequency": "Annually"},
        {"test": "Physical activity and diet review", "frequency": "Annually"},
    ],
    ("diabetes", "medium"): [
        {"test": "HbA1c", "frequency": "Every 3-6 months"},
        {"test": "Fasting glucose", "frequency": "Every 3-6 months"},
        {"test": "Foot examination", "frequency": "Annually"},
        {"test": "Dilated eye examination", "frequency": "Annually"},
    ],
    ("diabetes", "high"): [
        {"test": "HbA1c", "frequency": "Every 3 months"},
        {"test": "Home blood glucose monitoring", "frequency": "Daily"},
        {"test": "Foot examination", "frequency": "Every visit"},
        {"test": "Dilated eye examination", "frequency": "Annually"},
        {"test": "Urine albumin-to-creatinine ratio", "frequency": "Annually"},
    ],
    ("heart_disease", "low"): [
        {"test": "Blood pressure check", "frequency": "Annually"},
        {"test": "Lipid panel", "frequency": "Every 4-6 years"},
    ],
    ("heart_disease", "medium"): [
        {"test": "Lipid panel", "frequency": "Every 6-12 months"},
        {"test": "Blood pressure monitoring", "frequency": "Monthly"},
        {"test": "ECG review", "frequency": "As advised by clinician"},
    ],
    ("heart_disease", "high"): [
        {"test": "Cardiology consultation", "frequency": "Every 3 months"},
        {"test": "BNP / NT-proBNP", "frequency": "As directed by cardiologist"},
        {"test": "Blood pressure monitoring", "frequency": "Weekly"},
    ],
    ("chronic_kidney_disease", "low"): [
        {"test": "Serum creatinine / eGFR", "frequency": "Annually"},
        {"test": "Urine albumin-to-creatinine ratio", "frequency": "Annually"},
    ],
    ("chronic_kidney_disease", "medium"): [
        {"test": "eGFR", "frequency": "Every 3-6 months"},
        {"test": "Urine albumin-to-creatinine ratio", "frequency": "Every 3-6 months"},
        {"test": "Serum potassium", "frequency": "Every 3-6 months"},
    ],
    ("chronic_kidney_disease", "high"): [
        {"test": "Nephrology consultation", "frequency": "Every 1-3 months"},
        {"test": "eGFR", "frequency": "Every 1-3 months"},
        {"test": "Electrolyte panel", "frequency": "Monthly"},
    ],
    ("sepsis", "low"): [
        {"test": "Infection surveillance education", "frequency": "At discharge"},
    ],
    ("sepsis", "medium"): [
        {
            "test": "Clinical review for persistent infection signs",
            "frequency": "Weekly until resolved",
        },
        {"test": "Inflammatory markers (CRP)", "frequency": "As directed by clinician"},
    ],
    ("sepsis", "high"): [
        {"test": "Immediate escalation to acute care", "frequency": "Immediately"},
        {"test": "Lactate", "frequency": "Per sepsis protocol"},
        {"test": "Blood cultures before antibiotics", "frequency": "Immediately"},
    ],
}


def _normalize_feature_key(key: str) -> str:
    """Normalize a feature key for tolerant matching (lowercase, no separators)."""
    return "".join(character for character in key.lower() if character.isalnum())


def _align_feature_keys(
    features: Mapping[str, float], names: list[str]
) -> dict[str, float] | None:
    """
    Re-key an incoming feature mapping to match a model's feature names.

    Retrained models capture pipeline-normalized column names (e.g.
    ``bloodpressure``, ``diabetespedigreefunction``) while the manual /
    n8n path sends snake_case keys (``blood_pressure``,
    ``diabetes_pedigree_function``). This builds a tolerant lookup so
    both spellings align to the model's expected names.

    Parameters
    ----------
    features : Mapping[str, float]
        Incoming feature values.
    names : list[str]
        The model's expected feature names.

    Returns
    -------
    dict[str, float] | None
        Re-keyed features aligned to ``names``, or None when a required
        feature has no matching key.
    """

    lookup: dict[str, str] = {_normalize_feature_key(name): name for name in names}
    aligned: dict[str, float] = {}
    for key, value in features.items():
        canonical = lookup.get(_normalize_feature_key(key))
        if canonical is not None:
            aligned[canonical] = value
        else:
            aligned[key] = value
    if not all(name in aligned for name in names):
        return None
    return aligned


def summarize_patient(
    patient: PatientInfo,
    features: Mapping[str, float] | None,
    markers: Mapping[str, float] | None,
    input_type: str = "csv",
) -> str:
    """
    Summarize patient context for downstream agents.

    Parameters
    ----------
    patient : PatientInfo
        Patient identity (name/id/age used; nothing sensitive logged).
    features : Mapping[str, float] | None
        Raw input features (scanned for outlier magnitudes only).
    markers : Mapping[str, float] | None
        Clinical markers echoed into the summary.
    input_type : str
        Input modality label.

    Returns
    -------
    str
        Semicolon-joined patient summary.
    """

    parts = [
        f"Patient {patient.name} ({patient.id})",
        f"age {patient.age}" if patient.age else "",
        f"input type: {input_type}",
    ]
    if markers:
        marker_strs = [f"{k}={v}" for k, v in sorted(markers.items())]
        parts.append("markers: " + ", ".join(marker_strs))
    if features:
        abnormal = [
            key
            for key, value in features.items()
            if isinstance(value, (int, float)) and (value > 200 or value < 0)
        ]
        if abnormal:
            parts.append(f"outlier features: {abnormal}")
    return "; ".join(part for part in parts if part)


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
    if names is not None:
        aligned = _align_feature_keys(features, names)
        if aligned is None:
            missing = [name for name in names if name not in features]
            raise PredictionToolError(f"Missing feature values for columns: {missing}.")
        keys = list(names)
        features = aligned
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


def _marker_evidence(
    markers: Mapping[str, float] | None,
) -> tuple[float, list[str]]:
    """
    Quantify marker-based risk evidence.

    A marker contributes its normalized elevation
    ``clamp(value / threshold - 1, 0, 1)`` — zero at its threshold,
    one at twice the threshold — and the returned evidence score is the
    maximum elevation across the configured markers present in
    ``markers``, so a single severely elevated marker is not diluted by
    the count of normal ones.

    Parameters
    ----------
    markers : Mapping[str, float] | None
        Raw clinical markers compared against
        ``settings.MARKER_THRESHOLDS``.

    Returns
    -------
    tuple[float, list[str]]
        Maximum normalized elevation in ``[0, 1]`` and the factor
        strings for every elevated marker.

    Raises
    ------
    RiskToolError
        If any marker value is not numeric.
    """

    if not markers:
        return 0.0, []
    max_elevation = 0.0
    factors: list[str] = []
    for marker, threshold in settings.MARKER_THRESHOLDS.items():
        if marker not in markers:
            continue
        try:
            value = float(markers[marker])
        except (TypeError, ValueError) as error:
            raise RiskToolError(
                f"Marker '{marker}' must be numeric, got {markers[marker]!r}."
            ) from error
        if value > threshold:
            factors.append(f"Elevated {marker} ({value:.1f} > {threshold:.0f})")
            max_elevation = max(max_elevation, min(value / threshold - 1.0, 1.0))
    return max_elevation, factors


def assess_risk(
    prediction: PredictionResult,
    markers: Mapping[str, float] | None = None,
    disease_context: dict[str, object] | None = None,
) -> RiskResult:
    """
    Score risk from positive-class probability and elevated markers.

    The base score is the probability of the disease (positive) class —
    not the model's confidence in whichever class it predicted. Marker
    evidence can only *raise* the score: each elevated marker
    contributes its normalized elevation (``value / threshold - 1``,
    capped at 1) scaled by ``settings.RISK_MARKER_WEIGHT``, so the
    numeric score never contradicts the reported ``risk_factors`` (a
    flagged marker always lifts the score into at least the medium
    band when the elevation is severe enough). Markers never lower the
    score. The monitoring schedule is disease-specific when a disease
    context is available, falling back to the generic risk-level
    schedule.

    Parameters
    ----------
    prediction : PredictionResult
        Model prediction to score.
    markers : Mapping[str, float] | None
        Optional numeric clinical markers (e.g. glucose, bmi) compared
        against ``settings.MARKER_THRESHOLDS``.
    disease_context : dict[str, object] | None
        Entry from :data:`DISEASE_REGISTRY` for disease-specific
        monitoring; ``None`` keeps the generic schedule.

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
    max_elevation, factors = _marker_evidence(markers)
    if max_elevation > 0.0:
        score = max(score, settings.RISK_MARKER_WEIGHT * max_elevation)

    if score < settings.RISK_LOW_THRESHOLD:
        level = "low"
    elif score < settings.RISK_MEDIUM_THRESHOLD:
        level = "medium"
    else:
        level = "high"

    schedule = list(MONITORING_SCHEDULES.get(level, MONITORING_SCHEDULES["medium"]))
    if disease_context is not None:
        disease_key = str(disease_context["disease"])
        specific = DISEASE_MONITORING.get((disease_key, level))
        if specific:
            schedule = [dict(item) for item in specific]

    result = RiskResult(
        risk_score=round(score, 4),
        risk_level=level,
        risk_factors=factors,
        monitoring_schedule=schedule,
    )
    logger.info("Risk level %s (score %.4f)", level, score)
    return result


def build_evidence_query(features: Mapping[str, float] | None) -> str:
    """
    Build a retrieval query from raw feature values.

    Used when no prediction is available yet (per-agent evidence step):
    elevated markers select disease-anchored guideline queries,
    otherwise a generic management query is returned.

    Parameters
    ----------
    features : Mapping[str, float] | None
        Raw input features.

    Returns
    -------
    str
        Retrieval query text.
    """

    query = "clinical management and monitoring recommendations"
    if not features:
        return query
    glucose = features.get("glucose", 0)
    bmi = features.get("bmi", 0)
    creatinine = features.get("creatinine", 0)
    topic_parts = []
    if isinstance(glucose, (int, float)) and glucose > 126:
        topic_parts.append("diabetes hyperglycemia")
    if isinstance(creatinine, (int, float)) and creatinine > 1.5:
        topic_parts.append("chronic kidney disease creatinine")
    if isinstance(bmi, (int, float)) and bmi > 30:
        topic_parts.append("obesity metabolic health")
    if topic_parts:
        query = " ".join(topic_parts) + " treatment guidelines"
    return query


def build_explanation(
    prediction: PredictionResult | None,
    features: Mapping[str, float],
) -> tuple[str, list[str]]:
    """
    Explain a prediction via its top-magnitude features.

    Parameters
    ----------
    prediction : PredictionResult | None
        Model prediction (None yields an empty explanation).
    features : Mapping[str, float]
        Raw input features ranked by absolute value.

    Returns
    -------
    tuple[str, list[str]]
        Explanation text ("" when no prediction) and the top-3
        contributing feature names.
    """

    top_features = sorted(
        features.items(),
        key=lambda x: abs(x[1] if isinstance(x[1], (int, float)) else 0),
        reverse=True,
    )[:3]
    contributing = [key for key, _ in top_features]
    if prediction is None:
        return "", contributing
    explanation = (
        "Prediction driven primarily by: "
        + ", ".join(f"{k}={v}" for k, v in top_features)
        + f". Model confidence: {prediction.confidence:.1%}."
    )
    return explanation, contributing


def retrieve_evidence(
    pipeline: RAGPipeline,
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
) -> list[EvidenceItem]:
    """
    Retrieve evidence chunks for a query from the RAG pipeline.

    Parameters
    ----------
    pipeline : RAGPipeline
        Ingested retrieval pipeline.
    query : str
        Query text (should carry the disease context).
    top_k : int | None
        Number of results; defaults to ``settings.RAG_TOP_K``.
    topic : str | None
        Clinical topic tag to prioritize (e.g. ``"diabetes"``).

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
        results = pipeline.retrieve(query, top_k=limit, topic=topic)
    except (EmptyCorpusError, EmptyQueryError) as error:
        raise RetrievalToolError(str(error)) from error

    evidence = [
        EvidenceItem(
            document_id=result.chunk.document_id,
            source=result.chunk.source,
            score=result.score,
            text=result.chunk.text,
            topics=list((result.chunk.metadata or {}).get("topics", [])),
        )
        for result in results
    ]
    logger.info("Retrieved %d evidence items", len(evidence))
    return evidence


# ---------------------------------------------------------------------------
# Disease-specific treatment playbooks
# ---------------------------------------------------------------------------

#: Deterministic treatment recommendations keyed by ``(disease,
#: predicted_positive, risk_level)`` with a per-disease fallback. These
#: are clinical decision-support prompts for clinician review, not
#: prescriptions; they keep recommendations anchored to the assessed
#: condition instead of echoing whatever evidence chunks were retrieved.
TREATMENT_PLAYBOOKS: dict[tuple[str, bool], dict[str, list[str]]] = {
    ("diabetes", False): {
        "low": [
            "Maintain regular physical activity (>=150 min/week moderate "
            "intensity) and a balanced diet.",
            "Repeat diabetes screening (fasting glucose or HbA1c) annually, "
            "or earlier if symptoms appear.",
            "Keep BMI within the recommended range; discuss weight-management "
            "strategies with your clinician.",
        ],
        "medium": [
            "Discuss prediabetes-range results with your clinician; consider "
            "a structured lifestyle program.",
            "Repeat HbA1c in 3-6 months to confirm trend before any diagnosis.",
            "Review diet, activity, and sleep habits that affect glycemic control.",
        ],
        "high": [
            "Elevated diabetes probability despite class assignment - request "
            "clinician review of glycemic markers.",
            "Consider confirmatory HbA1c and fasting plasma glucose testing.",
        ],
    },
    ("diabetes", True): {
        "low": [
            "Confirm the diagnosis with HbA1c on a separate day before "
            "starting pharmacotherapy.",
            "Start lifestyle modification (diet, activity, weight target) as "
            "first-line therapy.",
        ],
        "medium": [
            "First-line management: medical nutrition therapy plus physical "
            "activity; metformin per clinician.",
            "Monitor HbA1c every 3-6 months until stable at target.",
            "Screen for diabetic complications (feet, eyes, kidneys) at baseline.",
        ],
        "high": [
            "Prompt clinician review required; assess for symptomatic hyperglycemia.",
            "Initiate guideline-based glucose-lowering therapy per ADA "
            "Standards of Care.",
            "Baseline complication screen: retinal exam, foot exam, urine "
            "albumin, lipid panel.",
        ],
    },
}

#: Generic fallback when no disease-specific playbook matches.
GENERIC_PLAYBOOK: dict[str, list[str]] = {
    "low": [
        "Continue routine preventive care and annual review with your clinician.",
    ],
    "medium": [
        "Schedule a clinical consultation to review the assessment findings.",
    ],
    "high": [
        "Seek prompt clinical review of this assessment by a physician.",
    ],
}


def build_treatment_recommendations(
    prediction: PredictionResult | None,
    risk: RiskResult | None,
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Produce disease-specific treatment recommendations and monitoring.

    Parameters
    ----------
    prediction : PredictionResult | None
        Enriched prediction carrying disease context.
    risk : RiskResult | None
        Risk assessment providing the risk level.

    Returns
    -------
    tuple[list[str], list[dict[str, str]]]
        Recommendation strings and the monitoring schedule.
    """

    level = risk.risk_level if risk else "low"
    monitoring = list(risk.monitoring_schedule) if risk else []

    playbook: dict[str, list[str]] = GENERIC_PLAYBOOK
    if prediction and prediction.disease:
        positive = (
            prediction.positive_probability is not None
            and prediction.negative_probability is not None
            and prediction.positive_probability >= prediction.negative_probability
        )
        playbook = TREATMENT_PLAYBOOKS.get(
            (prediction.disease, positive), GENERIC_PLAYBOOK
        )
    recommendations = [str(item) for item in playbook.get(level, [])]
    return recommendations, monitoring


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
        outcome = prediction.predicted_label or prediction.predicted_class
        if prediction.disease:
            disease_name = prediction.disease.replace("_", " ")
            prob_text = (
                f"; {disease_name} probability {prediction.positive_probability:.1%}"
                if prediction.positive_probability is not None
                else ""
            )
            summary_parts.append(f"Predicted condition: {outcome}{prob_text}.")
        else:
            summary_parts.append(
                f"Predicted condition: {outcome} "
                f"(model confidence {prediction.confidence:.2f})."
            )
    if risk is not None:
        summary_parts.append(f"Overall risk: {risk.risk_level.upper()}.")

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
    "DISEASE_REGISTRY",
    "assemble_clinical_report",
    "assess_risk",
    "build_disease_query",
    "build_rag_topic",
    "build_treatment_recommendations",
    "enrich_prediction",
    "resolve_disease",
    "retrieve_evidence",
    "run_image_prediction",
    "run_prediction",
]
