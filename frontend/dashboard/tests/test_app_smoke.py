"""
Headless smoke test of the Streamlit dashboard.

Uses Streamlit's ``AppTest`` harness to boot the app without a browser,
submit the analysis form with a mocked API, and assert the report
renders without exceptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app.py"

REPORT = {
    "patient": {"id": "p-1", "name": "Patient", "age": 54, "notes": ""},
    "input_type": "csv",
    "patient_summary": "Analysis completed for patient p-1 using csv input.",
    "prediction": {
        "predicted_class": "1",
        "probabilities": {"0": 0.3, "1": 0.7},
        "confidence": 0.7,
        "model_name": "tabular",
    },
    "risk": {
        "risk_score": 0.7,
        "risk_level": "high",
        "risk_factors": [],
        "monitoring_schedule": [],
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


@pytest.fixture()
def app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    return app


def test_dashboard_boots_without_exceptions(app):
    assert len(app.exception) == 0
    labels = [tab.label for tab in app.tabs]
    assert labels == ["Clinical Analysis", "Prediction", "Evidence Retrieval", "Info"]


def test_analysis_form_submission_renders_report(monkeypatch):
    def fake_analyze(
        self, patient, features, markers=None, recommendations=None, input_type="csv"
    ):
        return REPORT

    import dashboard.client as client_module

    monkeypatch.setattr(client_module.HealthcareAPIClient, "analyze", fake_analyze)

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    form = app.button[0]
    form.click().run()

    assert len(app.exception) == 0
    text = " ".join(
        [element.value for element in app.markdown]
        + [element.value for element in app.subheader]
    )
    assert "p-1" in text
    assert "Analysis completed" in text


def test_analysis_tab_renders_image_upload(monkeypatch):
    def fake_model_info(self):
        return {
            "available": True,
            "model_type": "image",
            "model_name": "image-cnn",
            "classes": ["glioma", "meningioma", "notumor", "pituitary"],
            "feature_names": None,
        }

    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", fake_model_info
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app.radio[0].set_value("Image (MRI upload)").run()
    assert len(app.exception) == 0
    assert len(app.get("file_uploader")) >= 1
