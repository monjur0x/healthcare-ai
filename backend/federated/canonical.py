"""
Canonical cross-hospital feature schema (per the research proposal).

The proposal federates four hospitals that own *different* disease
datasets:

* Hospital A — Pima Diabetes
* Hospital B — UCI Heart Disease
* Hospital C — Chronic Kidney Disease
* Hospital D — MIMIC-IV style ICU records (sepsis)

FedAvg can only average weights when every client's model has the same
shape, so each hospital maps its local columns onto one shared canonical
schema (the proposal's "Expected Inputs") and a single binary target
(``has_disease``). Missing canonical features are zero-filled by
:func:`ModelSpec.align_features` at the client, exactly as the existing
canonical-schema mechanism expects.

No raw rows leave a hospital; only the mapped numeric frame is used for
local training.
"""

from __future__ import annotations

import re

from collections.abc import Callable

import pandas as pd

from preprocessing.logger import get_logger

logger = get_logger(__name__)

#: Shared feature schema from the proposal's "Expected Inputs" section.
CANONICAL_FEATURES: tuple[str, ...] = (
    "age",
    "gender",
    "bmi",
    "blood_pressure",
    "heart_rate",
    "spo2",
    "glucose",
    "creatinine",
    "cholesterol",
    "hemoglobin",
    "albumin",
)

#: Canonical binary target column name.
TARGET_COLUMN = "has_disease"


def _norm(value: object) -> str:
    """Normalize a column name to lowercase snake_case."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _pick(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    """Return the first matching (normalized) column series, else None."""
    lookup = {_norm(column): column for column in frame.columns}
    for name in names:
        column = lookup.get(_norm(name))
        if column is not None:
            return pd.to_numeric(frame[column], errors="coerce")
    return None


def _gender01(series: pd.Series | None) -> pd.Series | None:
    """Map a gender/sex column to binary 1=male / 0=female."""
    if series is None:
        return None
    text = series.astype(str).str.strip().str.upper()
    mapping = {"M": 1.0, "MALE": 1.0, "1": 1.0, "F": 0.0, "FEMALE": 0.0, "0": 0.0}
    return text.map(mapping)


def _binary_from_threshold(
    series: pd.Series | None, threshold: float
) -> pd.Series | None:
    """Return ``series > threshold`` as float labels."""
    if series is None:
        return None
    return (pd.to_numeric(series, errors="coerce").fillna(0) > threshold).astype(float)


def _binary_from_text(series: pd.Series | None, positive: str) -> pd.Series | None:
    """Return 1.0 where stripped text equals ``positive``, else 0.0."""
    if series is None:
        return None
    text = series.astype(str).str.strip().str.lower()
    return (text == positive).astype(float)


def _assemble(
    frame: pd.DataFrame,
    columns: dict[str, pd.Series | None],
    label: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the canonical feature frame and label series.

    Missing canonical features are added as zeros so every hospital
    yields an identical schema; the client-side aligner then sees an
    already-aligned frame.
    """

    data: dict[str, pd.Series] = {}
    for name in CANONICAL_FEATURES:
        series = columns.get(name)
        if series is None:
            data[name] = pd.Series(0.0, index=frame.index)
        else:
            data[name] = pd.to_numeric(series, errors="coerce").fillna(0.0)

    features = pd.DataFrame(data, index=frame.index)
    if label is None:
        raise ValueError("Canonical adapter requires a label column.")
    labels = pd.to_numeric(label, errors="coerce").fillna(0.0).astype(int)
    labels.name = TARGET_COLUMN
    return features, labels


def adapt_diabetes(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Hospital A: Pima Indians Diabetes → canonical schema."""
    return _assemble(
        raw,
        {
            "age": _pick(raw, "age"),
            "bmi": _pick(raw, "bmi"),
            "blood_pressure": _pick(raw, "blood_pressure", "bp"),
            "glucose": _pick(raw, "glucose"),
            "insulin_extra": _pick(raw, "insulin"),  # ignored by schema
        },
        _binary_from_threshold(_pick(raw, "outcome", "class"), 0),
    )


def adapt_heart(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Hospital B: UCI Heart Disease → canonical schema."""
    return _assemble(
        raw,
        {
            "age": _pick(raw, "age"),
            "gender": _gender01(_pick(raw, "sex", "gender")),
            "blood_pressure": _pick(raw, "trestbps", "resting_bp"),
            "cholesterol": _pick(raw, "chol", "cholesterol"),
            # thalach = max heart rate achieved
            "heart_rate": _pick(raw, "thalach", "max_hr"),
        },
        _binary_from_threshold(_pick(raw, "num", "target"), 0),
    )


def adapt_kidney(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Hospital C: UCI Chronic Kidney Disease → canonical schema."""
    classification = None
    lookup = {_norm(column): column for column in raw.columns}
    column = lookup.get("classification") or lookup.get("class")
    if column is not None:
        classification = raw[column]
    return _assemble(
        raw,
        {
            "age": _pick(raw, "age"),
            "blood_pressure": _pick(raw, "bp", "blood_pressure"),
            "creatinine": _pick(raw, "sc", "serum_creatinine"),
            "hemoglobin": _pick(raw, "hemo", "hemoglobin"),
            "albumin": _pick(raw, "al", "albumin"),
            "glucose": _pick(raw, "bgr", "bu"),  # random blood glucose
        },
        _binary_from_text(classification, "ckd"),
    )


def adapt_sepsis(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Hospital D: MIMIC-IV style ICU sepsis data → canonical schema."""
    return _assemble(
        raw,
        {
            "age": _pick(raw, "age"),
            "gender": _gender01(_pick(raw, "gender", "sex")),
            "bmi": _pick(raw, "bmi"),
            "blood_pressure": _pick(raw, "sbp_mean", "map_mean"),
            "heart_rate": _pick(raw, "hr_mean"),
            "spo2": _pick(raw, "spo2_mean", "spo2_min"),
            "glucose": _pick(raw, "glucose"),
            "creatinine": _pick(raw, "creatinine"),
            "hemoglobin": _pick(raw, "hemoglobin"),
        },
        _binary_from_threshold(_pick(raw, "sepsis_label", "mortality"), 0),
    )


#: Per-preset canonical adapters keyed by hospital preset name.
ADAPTERS: dict[str, Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.Series]]] = {
    "diabetes": adapt_diabetes,
    "heart": adapt_heart,
    "kidney": adapt_kidney,
    "sepsis": adapt_sepsis,
}

#: Hospital id → dataset preset owning that specialty.
HOSPITAL_PRESETS: dict[str, str] = {
    "hospital_A": "diabetes",
    "hospital_B": "heart",
    "hospital_C": "kidney",
    "hospital_D": "sepsis",
}


def load_canonical_frame(
    csv_path: str, preset: str | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load one hospital CSV and map it to the canonical schema.

    Parameters
    ----------
    csv_path : str
        Path to the hospital's local CSV.
    preset : str | None
        Adapter name; inferred from the file path / hospital id when omitted.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Canonical feature frame and integer ``has_disease`` labels.
    """

    if preset is None:
        preset = next(
            (
                name
                for hospital, name in HOSPITAL_PRESETS.items()
                if hospital_key(csv_path, hospital)
            ),
            None,
        )
    if preset is None or preset not in ADAPTERS:
        raise ValueError(f"No canonical adapter for preset {preset!r} ({csv_path}).")

    raw = pd.read_csv(csv_path)
    features, labels = ADAPTERS[preset](raw)
    logger.info(
        "Mapped %s (%s): %d rows, %d features, %d/%d positives",
        csv_path,
        preset,
        len(labels),
        features.shape[1],
        int(labels.sum()),
        int((labels == 0).sum()),
    )
    return features, labels


def hospital_key(path: str, hospital_id: str) -> bool:
    """True when ``path`` belongs to ``hospital_id``."""
    return hospital_id in str(path)


def canonical_spec() -> tuple[int, int, tuple[str, ...]]:
    """Return ``(n_features, n_classes, feature_names)`` for the shared model."""
    return len(CANONICAL_FEATURES), 2, CANONICAL_FEATURES


__all__ = [
    "ADAPTERS",
    "CANONICAL_FEATURES",
    "HOSPITAL_PRESETS",
    "TARGET_COLUMN",
    "canonical_spec",
    "load_canonical_frame",
]
