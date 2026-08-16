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

#: Doctor-friendly labels for the more cryptic model column names.
DISPLAY_LABELS: dict[str, str] = {
    "trestbps": "Resting Blood Pressure (trestbps)",
    "chol": "Cholesterol (chol)",
    "thalch": "Max Heart Rate (thalch)",
    "exang": "Exercise-Induced Angina (exang)",
    "oldpeak": "ST Depression (oldpeak)",
    "restecg": "Resting ECG (restecg)",
    "cp": "Chest Pain Type (cp)",
    "fbs": "Fasting Blood Sugar >120 (fbs)",
    "slope": "ST Slope (slope)",
    "ca": "Major Vessels Count (ca)",
    "thal": "Thalassemia (thal)",
    "sc": "Serum Creatinine (sc)",
    "bu": "Blood Urea (bu)",
    "bgr": "Random Blood Glucose (bgr)",
    "pcv": "Packed Cell Volume (pcv)",
    "rc": "Red Blood Cells (rc)",
    "wc": "White Blood Cells (wc)",
    "hemo": "Hemoglobin (hemo)",
    "al": "Albumin (al)",
    "su": "Sugar in Urine (su)",
    "sg": "Specific Gravity (sg)",
    "bp": "Blood Pressure (bp)",
    "sbp_mean": "Systolic BP Mean (sbp_mean)",
    "sbp_max": "Systolic BP Max (sbp_max)",
    "sbp_min": "Systolic BP Min (sbp_min)",
    "dbp_mean": "Diastolic BP Mean (dbp_mean)",
    "dbp_max": "Diastolic BP Max (dbp_max)",
    "dbp_min": "Diastolic BP Min (dbp_min)",
    "map_mean": "Mean Arterial Pressure (map_mean)",
    "hr_mean": "Heart Rate Mean (hr_mean)",
    "hr_max": "Heart Rate Max (hr_max)",
    "hr_min": "Heart Rate Min (hr_min)",
    "spo2_mean": "SpO₂ Mean (spo2_mean)",
    "spo2_min": "SpO₂ Min (spo2_min)",
    "spo2_max": "SpO₂ Max (spo2_max)",
    "respiratory_rate_mean": "Resp. Rate Mean (respiratory_rate_mean)",
    "temp_celsius_mean": "Temperature Mean (temp_celsius_mean)",
    "pao2_fio2_ratio": "PaO₂/FiO₂ Ratio (pao2_fio2_ratio)",
    "lactate_mmol": "Lactate mmol/L (lactate_mmol)",
    "platelet_count": "Platelet Count (platelet_count)",
    "bilirubin_total": "Total Bilirubin (bilirubin_total)",
    "ph_arterial": "Arterial pH (ph_arterial)",
    "inr": "INR (inr)",
    "skinthickness": "Skin Thickness (skinthickness)",
    "insulin": "Insulin (insulin)",
    "diabetespedigreefunction": "Diabetes Pedigree Function (diabetespedigreefunction)",
    "pregnancies": "Pregnancies (pregnancies)",
}

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
    "build_analyze_payload",
    "explanation_sections",
    "feature_bounds",
    "feature_group",
    "feature_label",
    "group_features",
    "is_flag_feature",
    "normalize_feature_name",
    "output_availability",
    "parse_blood_pressure",
]
