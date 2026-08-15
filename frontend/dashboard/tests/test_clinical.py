"""
Tests for the pure clinical-domain helpers (feature grouping, payload
building, stage / explanation / output-availability derivation).
"""

from __future__ import annotations

from dashboard.clinical import (
    analysis_stages,
    build_analyze_payload,
    explanation_sections,
    feature_bounds,
    feature_group,
    feature_label,
    group_features,
    is_flag_feature,
    normalize_feature_name,
    output_availability,
)

REPORT = {
    "patient": {"id": "p-1", "name": "Patient", "age": 54, "notes": ""},
    "input_type": "csv",
    "patient_summary": "Analysis completed.",
    "prediction": {
        "predicted_class": "1",
        "probabilities": {"0": 0.3, "1": 0.7},
        "confidence": 0.7,
        "model_name": "tabular",
    },
    "risk": {
        "risk_score": 0.7,
        "risk_level": "high",
        "risk_factors": ["Elevated glucose (148.0 > 126)"],
        "monitoring_schedule": [
            {"test": "Medical consultation", "frequency": "Monthly"}
        ],
    },
    "evidence": [
        {
            "document_id": "diabetes.txt",
            "source": "protocols",
            "score": 0.9,
            "text": "diabetes is managed with metformin",
        }
    ],
    "context": "",
    "recommendations": ["Review with a physician."],
    "limitations": "AI analysis has inherent limitations.",
    "doctor_notice": "AI-assisted.",
}


def test_normalize_feature_name():
    assert normalize_feature_name(" Blood Pressure ") == "blood_pressure"
    assert normalize_feature_name("heart-rate") == "heart_rate"


def test_feature_group_maps_research_sections():
    assert feature_group("age") == "Patient Information"
    assert feature_group("sex") == "Patient Information"
    assert feature_group("bloodpressure") == "Vital Signs"
    assert feature_group("hr_mean") == "Vital Signs"
    assert feature_group("spo2_mean") == "Vital Signs"
    assert feature_group("glucose") == "Clinical Measurements"
    assert feature_group("bmi") == "Clinical Measurements"
    assert feature_group("creatinine") == "Clinical Measurements"
    assert feature_group("cholesterol") == "Clinical Measurements"
    assert feature_group("dm") == "Medical History"
    assert feature_group("exang") == "Medical History"
    assert feature_group("unknown_column") == "Additional Model Features"


def test_group_features_orders_and_falls_back():
    groups = group_features(["bmi", "age", "unknown_x", "glucose", "spo2_mean"])
    labels = [label for label, _ in groups]
    assert labels == [
        "Patient Information",
        "Vital Signs",
        "Clinical Measurements",
        "Additional Model Features",
    ]
    by_label = dict(groups)
    assert by_label["Patient Information"] == ["age"]
    assert by_label["Vital Signs"] == ["spo2_mean"]
    assert by_label["Clinical Measurements"] == ["bmi", "glucose"]
    assert by_label["Additional Model Features"] == ["unknown_x"]


def test_feature_label_humanizes():
    assert feature_label("trestbps").startswith("Resting Blood Pressure")
    assert feature_label("glucose") == "Glucose"
    assert feature_label("heart_rate") == "Heart Rate"


def test_flag_and_bounds_detection():
    assert is_flag_feature("dm")
    assert is_flag_feature("mechanical_ventilation")
    assert is_flag_feature("any_flag")
    assert not is_flag_feature("glucose")
    assert feature_bounds("age") == (0.0, 120.0)
    assert feature_bounds("spo2_mean") == (0.0, 100.0)
    assert feature_bounds("glucose") is None


def test_build_analyze_payload():
    payload = build_analyze_payload(
        patient={"id": "p-1", "name": "P"},
        features={"glucose": 148.0},
        markers={"glucose": 148.0},
        recommendations=["Review."],
        input_type="csv",
    )
    assert payload["features"] == {"glucose": 148.0}
    assert payload["markers"] == {"glucose": 148.0}
    assert payload["recommendations"] == ["Review."]
    assert payload["input_type"] == "csv"
    assert payload["patient"]["id"] == "p-1"


def test_build_analyze_payload_omits_optional():
    payload = build_analyze_payload(patient={"id": "p-1"}, features={"glucose": 148.0})
    assert "markers" not in payload
    assert "recommendations" not in payload


def test_analysis_stages_reflect_actual_report():
    stages = analysis_stages(REPORT)
    by_label = {stage["label"]: stage for stage in stages}
    assert by_label["Patient data received"]["done"] is True
    assert by_label["Prediction completed"]["done"] is True
    assert by_label["Risk assessment completed"]["done"] is True
    assert by_label["Clinical evidence retrieved"]["done"] is True
    assert by_label["Treatment recommendation generated"]["done"] is True
    assert by_label["Analysis complete"]["done"] is True


def test_analysis_stages_mark_missing_as_undone():
    stages = analysis_stages({"patient": {"id": "p"}, "evidence": []})
    by_label = {stage["label"]: stage for stage in stages}
    assert by_label["Prediction completed"]["done"] is False
    assert by_label["Treatment recommendation generated"]["done"] is False
    assert by_label["Clinical evidence retrieved"]["done"] is False


def test_explanation_sections_from_model_outputs():
    sections = explanation_sections(REPORT)
    titles = [section["title"] for section in sections]
    assert "Predicted condition" in titles
    assert "Risk assessment" in titles
    assert "Suggested monitoring" in titles
    assert "Model certainty" in titles
    joined = " ".join(section["body"] for section in sections)
    assert "0.7" in joined or "70%" in joined
    assert "Elevated glucose" in joined


def test_explanation_sections_empty_report():
    sections = explanation_sections({"patient": {"id": "p"}})
    assert sections[0]["title"] == "No model output"


def test_output_availability_is_honest():
    availability = output_availability(REPORT)
    assert availability["Disease Risk Score"] is True
    assert availability["Treatment Recommendation"] is True
    assert availability["Clinical Evidence"] is True
    assert availability["Explainable Decision Report"] is True
    assert availability["Mortality Risk"] is False
    assert availability["Readmission Risk"] is False

    empty = output_availability({"patient": {"id": "p"}})
    assert empty["Disease Risk Score"] is False
    assert empty["Explainable Decision Report"] is False
