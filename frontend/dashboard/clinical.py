"""
Pure clinical-domain helpers for the Streamlit dashboard.

These functions group model features into the research-defined clinical
sections, build analysis payloads, and derive doctor-facing summaries
(pipeline stages, explanation, output availability) from a clinical
report. They contain no Streamlit or HTTP code so they stay unit-testable.

Only fields actually supported by the backend model are surfaced: the
grouping is driven by ``/api/v1/model`` feature names, and every
derivation reads real report fields (nothing is fabricated here).
"""

from __future__ import annotations

import math

from collections.abc import Mapping, Sequence

PATIENT_INFO = {
    "age",
    "sex",
    "gender",
    "ethnicity",
    "insurance",
}

VITAL_SIGNS = {
    "bloodpressure",
    "trestbps",
    "bp",
    "sbp_mean",
    "sbp_max",
    "sbp_min",
    "sbp_std",
    "dbp_mean",
    "dbp_max",
    "dbp_min",
    "dbp_std",
    "map_mean",
    "hr_mean",
    "hr_max",
    "hr_min",
    "hr_std",
    "thalch",
    "spo2_mean",
    "spo2_min",
    "spo2_max",
    "spo2_std",
    "respiratory_rate_mean",
    "respiratory_rate_max",
    "respiratory_rate_min",
    "respiratory_rate_std",
    "temp_celsius_mean",
    "temp_celsius_max",
    "temp_celsius_min",
    "temp_celsius_std",
    "fio2_percent",
    "pao2_fio2_ratio",
    "gcs_total",
}

CLINICAL_MEASUREMENTS = {
    "bmi",
    "glucose",
    "cholesterol",
    "chol",
    "creatinine",
    "sc",
    "bgr",
    "bu",
    "sod",
    "sodium",
    "pot",
    "potassium",
    "chloride",
    "bicarbonate",
    "hemo",
    "hemoglobin",
    "pcv",
    "hematocrit",
    "wbc",
    "wc",
    "rc",
    "platelet_count",
    "lactate_mmol",
    "bilirubin_total",
    "inr",
    "ph_arterial",
    "skinthickness",
    "insulin",
    "diabetespedigreefunction",
    "oldpeak",
    "sg",
    "al",
    "su",
    "fbs",
    "restecg",
    "slope",
    "ca",
    "thal",
    "cp",
    "sofa_score",
    "apache_iv",
    "qsofa",
    "sirs_criteria",
}

MEDICAL_HISTORY = {
    "pregnancies",
    "diabetes",
    "hypertension",
    "chf",
    "copd",
    "chronic_kidney_disease",
    "liver_disease",
    "immunosuppression",
    "cad",
    "atrial_fibrillation",
    "cancer_active",
    "dm",
    "htn",
    "exang",
    "smoking",
    "readmission_30day",
    "vasopressors_flag",
    "mechanical_ventilation",
    "antibiotics_24h",
    "insulin_infusion_flag",
    "fluids_ml_24h",
    "sedation_score",
    "vasopressor_dose_mcg_kg_min",
}

#: Ordered group labels; any unlisted feature lands in the last group.
GROUP_ORDER = [
    "Patient Information",
    "Vital Signs",
    "Clinical Measurements",
    "Medical History",
    "Additional Model Features",
]

FEATURE_GROUPS: dict[str, set[str]] = {
    "Patient Information": PATIENT_INFO,
    "Vital Signs": VITAL_SIGNS,
    "Clinical Measurements": CLINICAL_MEASUREMENTS,
    "Medical History": MEDICAL_HISTORY,
}

#: Doctor-friendly labels for the model column names. Raw dataset column
#: names are never shown to doctors; unknown names fall back to a
#: Title-cased version of the normalized name.
DISPLAY_LABELS: dict[str, str] = {
    # Shared demographics
    "age": "Age",
    "sex": "Sex",
    "gender": "Gender",
    "ethnicity": "Ethnicity",
    "insurance": "Insurance",
    # PIMA diabetes
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "bloodpressure": "Blood Pressure",
    "skinthickness": "Skin Thickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetespedigreefunction": "Diabetes Pedigree Function",
    # UCI heart disease
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "thalch": "Max Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression",
    "restecg": "Resting ECG",
    "cp": "Chest Pain Type",
    "fbs": "Fasting Blood Sugar >120",
    "slope": "ST Slope",
    "ca": "Major Vessels Count",
    "thal": "Thalassemia",
    # UCI chronic kidney disease
    "sc": "Serum Creatinine",
    "bu": "Blood Urea",
    "bgr": "Random Blood Glucose",
    "pcv": "Packed Cell Volume",
    "rc": "Red Blood Cells",
    "wc": "White Blood Cells",
    "hemo": "Hemoglobin",
    "al": "Albumin",
    "su": "Sugar in Urine",
    "sg": "Specific Gravity",
    "rbc": "Red Blood Cells",
    "pc": "Pus Cells",
    "pcc": "Pus Cell Clumps",
    "ba": "Bacteria",
    "sod": "Serum Sodium",
    "pot": "Serum Potassium",
    "appet": "Appetite",
    "pe": "Pedal Edema",
    "ane": "Anemia",
    "bp": "Blood Pressure",
    "htn": "Hypertension",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    # Sepsis ICU (synthetic)
    "sbp_mean": "Systolic BP Mean",
    "sbp_max": "Systolic BP Max",
    "sbp_min": "Systolic BP Min",
    "dbp_mean": "Diastolic BP Mean",
    "dbp_max": "Diastolic BP Max",
    "dbp_min": "Diastolic BP Min",
    "map_mean": "Mean Arterial Pressure",
    "hr_mean": "Heart Rate Mean",
    "hr_max": "Heart Rate Max",
    "hr_min": "Heart Rate Min",
    "spo2_mean": "SpO₂ Mean",
    "spo2_min": "SpO₂ Min",
    "spo2_max": "SpO₂ Max",
    "respiratory_rate_mean": "Resp. Rate Mean",
    "respiratory_rate_max": "Resp. Rate Max",
    "respiratory_rate_min": "Resp. Rate Min",
    "temp_celsius_mean": "Temperature Mean",
    "temp_celsius_max": "Temperature Max",
    "temp_celsius_min": "Temperature Min",
    "gcs_total": "Glasgow Coma Scale",
    "lactate_mmol": "Lactate",
    "platelet_count": "Platelet Count",
    "bilirubin_total": "Total Bilirubin",
    "ph_arterial": "Arterial pH",
    "inr": "INR",
    "fio2_percent": "FiO₂",
    "pao2_fio2_ratio": "PaO₂/FiO₂ Ratio",
    "sofa_score": "SOFA Score",
    "apache_iv": "APACHE IV Score",
    "qsofa": "qSOFA Score",
    "sirs_criteria": "SIRS Criteria Count",
    "vasopressors_flag": "Vasopressors Required",
    "mechanical_ventilation": "Mechanical Ventilation",
    "antibiotics_24h": "Antibiotics Within 24h",
    "insulin_infusion_flag": "Insulin Infusion",
    "fluids_ml_24h": "IV Fluids (24h)",
    "sedation_score": "Sedation Score",
    "vasopressor_dose_mcg_kg_min": "Vasopressor Dose",
    "readmission_30day": "Readmission Within 30 Days",
    # General / fallback-friendly
    "cholesterol": "Cholesterol",
    "creatinine": "Creatinine",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "chloride": "Chloride",
    "bicarbonate": "Bicarbonate",
    "hemoglobin": "Hemoglobin",
    "hematocrit": "Hematocrit",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "chf": "Congestive Heart Failure",
    "copd": "COPD",
    "chronic_kidney_disease": "Chronic Kidney Disease",
    "liver_disease": "Liver Disease",
    "immunosuppression": "Immunosuppression",
    "atrial_fibrillation": "Atrial Fibrillation",
    "cancer_active": "Active Cancer",
    "smoking": "Smoking",
}

#: Verified units for the documented datasets. Only units the project
#: data / documentation supports are listed; anything else is left blank
#: rather than guessed.
FEATURE_UNITS: dict[str, str] = {
    "pregnancies": "count",
    "glucose": "mg/dL",
    "bloodpressure": "mmHg",
    "skinthickness": "mm",
    "insulin": "µU/mL",
    "bmi": "kg/m²",
    "age": "years",
    "trestbps": "mmHg",
    "chol": "mg/dL",
    "thalch": "bpm",
    "bp": "mmHg",
    "sbp_mean": "mmHg",
    "sbp_max": "mmHg",
    "sbp_min": "mmHg",
    "dbp_mean": "mmHg",
    "dbp_max": "mmHg",
    "dbp_min": "mmHg",
    "map_mean": "mmHg",
    "hr_mean": "bpm",
    "hr_max": "bpm",
    "hr_min": "bpm",
    "spo2_mean": "%",
    "spo2_min": "%",
    "spo2_max": "%",
    "respiratory_rate_mean": "breaths/min",
    "respiratory_rate_max": "breaths/min",
    "respiratory_rate_min": "breaths/min",
    "temp_celsius_mean": "°C",
    "temp_celsius_max": "°C",
    "temp_celsius_min": "°C",
    "lactate_mmol": "mmol/L",
    "sod": "mEq/L",
    "pot": "mEq/L",
    "bgr": "mg/dL",
    "bu": "mg/dL",
    "sc": "mg/dL",
    "hemo": "g/dL",
}

#: Feature names that must be whole numbers (counts / scores).
INTEGER_FEATURES = frozenset(
    {
        "pregnancies",
        "ca",
        "wc",
        "rc",
        "gcs_total",
        "sofa_score",
        "qsofa",
        "sirs_criteria",
    }
)

#: Feature names that are binary flags; rendered as a checkbox in the form.
FLAG_FEATURES = frozenset(
    {
        "fbs",
        "exang",
        "htn",
        "dm",
        "cad",
        "ane",
        "pe",
        "appet",
        "diabetes",
        "hypertension",
        "chf",
        "copd",
        "chronic_kidney_disease",
        "liver_disease",
        "immunosuppression",
        "atrial_fibrillation",
        "cancer_active",
        "smoking",
        "readmission_30day",
        "mechanical_ventilation",
    }
)

#: Feature names with an upper bound on a reasonable clinical input range.
BOUNDED_FEATURES: dict[str, tuple[float, float]] = {
    "age": (0.0, 120.0),
    "spo2_mean": (0.0, 100.0),
    "spo2_min": (0.0, 100.0),
    "spo2_max": (0.0, 100.0),
    "spo2_std": (0.0, 100.0),
}

#: The research-defined expected outputs, in display order.
RESEARCH_OUTPUTS = [
    "Disease Risk Score",
    "Mortality Risk",
    "Readmission Risk",
    "Treatment Recommendation",
    "Clinical Evidence",
    "Explainable Decision Report",
]


def normalize_feature_name(name: str) -> str:
    """
    Normalize a column name to the pipeline convention.

    Parameters
    ----------
    name : str
        Raw column name.

    Returns
    -------
    str
        Lowercase snake_case name.
    """
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def feature_group(name: str) -> str:
    """
    Map a feature name to its research-defined clinical group.

    Parameters
    ----------
    name : str
        Model feature column name.

    Returns
    -------
    str
        Group label (last group is the catch-all).
    """
    normalized = normalize_feature_name(name)
    for group in GROUP_ORDER:
        if normalized in FEATURE_GROUPS.get(group, set()):
            return group
    return GROUP_ORDER[-1]


def group_features(feature_names: Sequence[str]) -> list[tuple[str, list[str]]]:
    """
    Organize model feature names into ordered clinical groups.

    Parameters
    ----------
    feature_names : Sequence[str]
        Feature columns reported by ``/api/v1/model``.

    Returns
    -------
    list[tuple[str, list[str]]]
        ``(group label, [feature names])`` pairs, groups with no members
        are omitted.
    """
    grouped: dict[str, list[str]] = {}
    for name in feature_names:
        grouped.setdefault(feature_group(name), []).append(name)
    return [
        (group, grouped.get(group, [])) for group in GROUP_ORDER if grouped.get(group)
    ]


def feature_label(name: str) -> str:
    """
    Return a doctor-friendly display label for a feature.

    Parameters
    ----------
    name : str
        Model feature column name.

    Returns
    -------
    str
        Known display label, or a Title-cased version of the raw name.
    """
    normalized = normalize_feature_name(name)
    if normalized in DISPLAY_LABELS:
        return DISPLAY_LABELS[normalized]
    return normalized.replace("_", " ").title()


def is_flag_feature(name: str) -> bool:
    """
    Whether a feature should be rendered as a binary (on/off) input.

    Parameters
    ----------
    name : str
        Model feature column name.

    Returns
    -------
    bool
        True for known flag columns or names ending in ``_flag``.
    """
    normalized = normalize_feature_name(name)
    return normalized in FLAG_FEATURES or normalized.endswith("_flag")


def feature_bounds(name: str) -> tuple[float, float] | None:
    """
    Reasonable numeric bounds for a feature input, when known.

    Parameters
    ----------
    name : str
        Model feature column name.

    Returns
    -------
    tuple[float, float] | None
        ``(min, max)`` for bounded features, else None.
    """
    return BOUNDED_FEATURES.get(normalize_feature_name(name))


def feature_unit(name: str) -> str | None:
    """
    Verified display unit for a feature, when known.

    Parameters
    ----------
    name : str
        Model feature column name.

    Returns
    -------
    str | None
        Unit string (e.g. ``"mg/dL"``), or None when the unit is not
        verified by the project data / documentation.
    """
    return FEATURE_UNITS.get(normalize_feature_name(name))


def is_integer_feature(name: str) -> bool:
    """
    Whether a feature must be a whole number (counts / scores).

    Parameters
    ----------
    name : str
        Model feature column name.

    Returns
    -------
    bool
        True for known integer columns.
    """
    return normalize_feature_name(name) in INTEGER_FEATURES


def parse_blood_pressure(raw: str) -> float | None:
    """
    Parse a blood-pressure entry into the model's ``bloodpressure`` value.

    The model feature ``bloodpressure`` corresponds to the diastolic
    reading (the PIMA diabetes "Blood Pressure (mm Hg)" column), so a
    ``"SYS/DIA"`` entry maps to the diastolic component. A lone number is
    used as-is.

    Parameters
    ----------
    raw : str
        User entry, e.g. ``"120/90"``, ``"120 / 90"``, or ``"90"``.

    Returns
    -------
    float | None
        The diastolic value (or the single value), or None when the entry
        is not parseable.
    """
    text = raw.strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split("/")]
    try:
        if len(parts) == 2:
            systolic = float(parts[0])
            diastolic = float(parts[1])
            if systolic <= 0 or diastolic <= 0 or diastolic > systolic:
                return None
            return diastolic
        if len(parts) == 1:
            value = float(parts[0])
            return value if value > 0 else None
        return None
    except ValueError:
        return None


def build_analyze_payload(
    patient: Mapping[str, object],
    features: Mapping[str, float],
    markers: Mapping[str, float] | None = None,
    recommendations: Sequence[str] | None = None,
    input_type: str = "csv",
) -> dict[str, object]:
    """
    Build the shared analysis request body for the backend and n8n.

    Parameters
    ----------
    patient : Mapping[str, object]
        Patient context (``name``, ``id``, ``age``, ``notes``).
    features : Mapping[str, float]
        Feature row for the prediction step.
    markers : Mapping[str, float] | None
        Raw clinical markers for the risk assessment.
    recommendations : Sequence[str] | None
        Recommendation strings for the report.
    input_type : str
        Data modality analyzed (``"csv"`` / ``"image"`` / ...).

    Returns
    -------
    dict[str, object]
        JSON-request body compatible with ``POST /api/v1/analyze``.
    """
    payload: dict[str, object] = {
        "patient": dict(patient),
        "features": dict(features),
        "input_type": input_type,
    }
    if markers is not None:
        payload["markers"] = dict(markers)
    if recommendations:
        payload["recommendations"] = list(recommendations)
    return payload


def validate_feature_values(
    values: Mapping[str, float | None],
    schema: Sequence[str],
    patient_age: int | None = None,
) -> list[str]:
    """
    Validate entered feature values against the model schema.

    Checks required fields, numeric validity, integer-only features, and
    documented bounds. The backend remains the final authority for
    validation; this pass only gives the doctor clear, early messages.

    Parameters
    ----------
    values : Mapping[str, float | None]
        Feature name to entered value (``None`` marks an unparseable
        entry such as an invalid blood-pressure string).
    schema : Sequence[str]
        Feature columns the selected model requires.
    patient_age : int | None
        Patient-context age (fills the model ``age`` feature when the
        schema requires it).

    Returns
    -------
    list[str]
        Human-readable validation messages (empty when valid).
    """
    errors: list[str] = []
    for name in schema:
        normalized = normalize_feature_name(name)
        label = feature_label(name)
        if normalized == "age":
            if patient_age is None or patient_age <= 0:
                errors.append("Patient age is required for this assessment type.")
                continue
            bounds = feature_bounds(name)
            if bounds:
                low, high = bounds
                if patient_age < low or patient_age > high:
                    errors.append(f"{label} must be between {low:g} and {high:g}.")
            continue
        value = values.get(name)
        if value is None:
            errors.append(f"{label} is required.")
            continue
        if not math.isfinite(float(value)):
            errors.append(f"{label} must be a valid number.")
            continue
        if is_integer_feature(name) and float(value) != float(int(float(value))):
            errors.append(f"{label} must be a whole number.")
        bounds = feature_bounds(name)
        if bounds:
            low, high = bounds
            if value < low or value > high:
                errors.append(f"{label} must be between {low:g} and {high:g}.")
    return errors


def assessment_summary(
    patient: Mapping[str, object],
    preset_label: str,
    schema: Sequence[str],
    values: Mapping[str, float | None],
    notes_provided: bool,
    patient_age: int | None = None,
) -> list[tuple[str, str]]:
    """
    Build the review-before-analysis summary rows.

    The feature count is computed from the actual model schema, never
    hardcoded.

    Parameters
    ----------
    patient : Mapping[str, object]
        Patient context (``name``, ``id``, ``age``, ``notes``).
    preset_label : str
        Doctor-friendly label of the selected assessment type.
    schema : Sequence[str]
        Feature columns the selected model requires.
    values : Mapping[str, float | None]
        Entered feature values (``None`` marks unparseable entries).
    notes_provided : bool
        Whether optional clinical notes were entered.
    patient_age : int | None
        Patient-context age (counted as entered when the model requires
        ``age``).

    Returns
    -------
    list[tuple[str, str]]
        ``(label, value)`` rows for the summary.
    """
    entered = 0
    for name in schema:
        if normalize_feature_name(name) == "age":
            if patient_age is not None and patient_age > 0:
                entered += 1
        elif values.get(name) is not None:
            entered += 1
    return [
        ("Patient", str(patient.get("name") or "Unknown")),
        ("Patient ID", str(patient.get("id") or "Unknown")),
        ("Assessment", preset_label),
        ("Clinical data", f"{entered} / {len(schema)} required features"),
        ("Medical image", "Not provided"),
        ("Clinical notes", "Provided" if notes_provided else "Not provided"),
    ]


def analysis_stages(report: Mapping[str, object]) -> list[dict[str, object]]:
    """
    Derive the pipeline stages that actually completed for a report.

    The stages mirror the backend clinical pipeline (prediction -> risk ->
    RAG evidence -> report); each is marked complete only when the report
    really carries the corresponding output.

    Parameters
    ----------
    report : Mapping[str, object]
        ``ClinicalReport`` payload.

    Returns
    -------
    list[dict[str, object]]
        ``{"label", "done", "detail"}`` entries.
    """
    prediction = report.get("prediction")
    risk = report.get("risk")
    evidence = report.get("evidence") or []
    recommendations = report.get("recommendations") or []

    def prediction_detail() -> str:
        if not isinstance(prediction, Mapping):
            return "Skipped — no prediction model configured"
        predicted = prediction.get("predicted_class")
        confidence = prediction.get("confidence", 0.0)
        return f"Predicted {predicted} at {confidence:.0%} confidence"

    return [
        {
            "label": "Patient data received",
            "done": True,
            "detail": f"Patient {report.get('patient', {}).get('id', 'unknown')}",
        },
        {
            "label": "Prediction completed",
            "done": bool(prediction),
            "detail": prediction_detail(),
        },
        {
            "label": "Risk assessment completed",
            "done": bool(risk),
            "detail": f"Risk {risk.get('risk_level')}" if risk else "Not available",
        },
        {
            "label": "Clinical evidence retrieved",
            "done": bool(evidence),
            "detail": (
                f"{len(evidence)} evidence item(s)" if evidence else "No evidence"
            ),
        },
        {
            "label": "Treatment recommendation generated",
            "done": bool(recommendations),
            "detail": (
                f"{len(recommendations)} recommendation(s)"
                if recommendations
                else "Not generated"
            ),
        },
        {
            "label": "Analysis complete",
            "done": True,
            "detail": "",
        },
    ]


def explanation_sections(report: Mapping[str, object]) -> list[dict[str, str]]:
    """
    Build a doctor-friendly explanation from actual model outputs.

    No chain-of-thought or internal reasoning is exposed; the explanation
    is limited to what the model actually produced (prediction,
    confidence, risk, contributing factors, monitoring plan).

    Parameters
    ----------
    report : Mapping[str, object]
        ``ClinicalReport`` payload.

    Returns
    -------
    list[dict[str, str]]
        ``{"title", "body"}`` explanation entries.
    """
    sections: list[dict[str, str]] = []
    prediction = report.get("prediction")
    risk = report.get("risk")

    if isinstance(prediction, Mapping):
        sections.append(
            {
                "title": "Predicted condition",
                "body": (
                    f"The model-estimated primary condition is "
                    f"**{prediction.get('predicted_class')}** with an estimated "
                    f"confidence of {prediction.get('confidence', 0.0):.0%}."
                ),
            }
        )

    if isinstance(risk, Mapping):
        factors = risk.get("risk_factors") or []
        factor_text = (
            "No clinical markers were flagged as elevated."
            if not factors
            else "The following clinical markers exceeded their reference thresholds: "
            + "; ".join(str(factor) for factor in factors)
            + "."
        )
        sections.append(
            {
                "title": "Risk assessment",
                "body": (
                    f"The overall model-estimated risk level is "
                    f"**{risk.get('risk_level', 'unknown')}** "
                    f"(score {risk.get('risk_score', 0.0):.2f}). {factor_text}"
                ),
            }
        )
        schedule = risk.get("monitoring_schedule") or []
        if schedule:
            plan = "; ".join(
                (
                    f"{item.get('test')} ({item.get('frequency')})"
                    if isinstance(item, Mapping)
                    else str(item)
                )
                for item in schedule
            )
            sections.append(
                {
                    "title": "Suggested monitoring",
                    "body": plan + ".",
                }
            )

    if isinstance(prediction, Mapping):
        probabilities = prediction.get("probabilities") or {}
        if probabilities:
            ranked = sorted(
                probabilities.items(), key=lambda item: float(item[1]), reverse=True
            )[:3]
            top = ", ".join(
                f"{label} {float(probability):.0%}" for label, probability in ranked
            )
            sections.append(
                {
                    "title": "Model certainty",
                    "body": f"Highest-confidence outcomes: {top}.",
                }
            )

    if not sections:
        sections.append(
            {
                "title": "No model output",
                "body": (
                    "No prediction or risk output was produced for this analysis "
                    "(no prediction model was configured)."
                ),
            }
        )
    return sections


def output_availability(report: Mapping[str, object]) -> dict[str, bool]:
    """
    Report which research-defined outputs are present for a report.

    Parameters
    ----------
    report : Mapping[str, object]
        ``ClinicalReport`` payload.

    Returns
    -------
    dict[str, bool]
        Output title to availability flag.
    """
    prediction = bool(report.get("prediction"))
    risk = bool(report.get("risk"))
    return {
        "Disease Risk Score": risk,
        "Mortality Risk": False,
        "Readmission Risk": False,
        "Treatment Recommendation": bool(report.get("recommendations")),
        "Clinical Evidence": bool(report.get("evidence")),
        "Explainable Decision Report": prediction or risk,
    }


__all__ = [
    "RESEARCH_OUTPUTS",
    "analysis_stages",
    "assessment_summary",
    "build_analyze_payload",
    "explanation_sections",
    "feature_bounds",
    "feature_group",
    "feature_label",
    "feature_unit",
    "group_features",
    "is_flag_feature",
    "is_integer_feature",
    "normalize_feature_name",
    "output_availability",
    "parse_blood_pressure",
    "validate_feature_values",
]
