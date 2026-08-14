"""
Tests for the clinical report assembly service.
"""

from __future__ import annotations

import pytest

from CrewAI.orchestrator.exceptions import ReportError
from CrewAI.orchestrator.schemas import (
    ClinicalReport,
    EvidenceItem,
    PatientInfo,
    PredictionResult,
    RiskResult,
)
from CrewAI.orchestrator.services import assemble_clinical_report


def test_report_assembles_with_all_parts() -> None:
    report = assemble_clinical_report(
        patient=PatientInfo(name="A", id="p1", age=50),
        input_type="csv",
        prediction=PredictionResult(
            predicted_class="diabetes",
            probabilities={"healthy": 0.2, "diabetes": 0.8},
            confidence=0.8,
        ),
        risk=RiskResult(risk_score=0.8, risk_level="high"),
        evidence=[
            EvidenceItem(
                document_id="d1", source="guideline", text="metformin first line"
            )
        ],
        recommendations=["Confirm with physician"],
    )
    assert isinstance(report, ClinicalReport)
    assert report.prediction.predicted_class == "diabetes"
    assert report.risk.risk_level == "high"
    assert report.context == "[d1] (0.0000)\nmetformin first line"
    assert report.recommendations == ["Confirm with physician"]
    assert "diabetes" in report.patient_summary
    assert report.doctor_notice


def test_report_without_prediction_and_risk() -> None:
    report = assemble_clinical_report(patient=PatientInfo(id="p1"))
    assert report.prediction is None
    assert report.risk is None
    assert report.evidence == []


def test_report_requires_prediction_and_risk_together() -> None:
    with pytest.raises(ReportError):
        assemble_clinical_report(
            patient=PatientInfo(id="p1"),
            prediction=PredictionResult(
                predicted_class="x", probabilities={"x": 1.0}, confidence=1.0
            ),
        )
    with pytest.raises(ReportError):
        assemble_clinical_report(
            patient=PatientInfo(id="p1"),
            risk=RiskResult(risk_score=0.5, risk_level="medium"),
        )


def test_report_serializes_to_dict() -> None:
    report = assemble_clinical_report(patient=PatientInfo(id="p1"))
    payload = report.to_dict()
    assert payload["patient"]["id"] == "p1"
    assert "doctor_notice" in payload
