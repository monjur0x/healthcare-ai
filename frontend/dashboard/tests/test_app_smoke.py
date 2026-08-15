"""
Headless smoke tests of the Streamlit dashboard.

Uses Streamlit's ``AppTest`` harness to boot the app without a browser,
submit the analysis form with mocked API calls, and assert the clinical
results render without exceptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app.py"

TAB_LABELS = [
    "Overview",
    "Clinical Assessment",
    "Imaging",
    "Results",
    "System Status",
]

ROUTE_DIRECT = "Direct to FastAPI"
ROUTE_N8N = "Via n8n workflow"

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

TABULAR_MODEL = {
    "available": True,
    "model_type": "tabular",
    "model_name": "mlp",
    "classes": ["0", "1"],
    "feature_names": ["glucose", "bmi", "age"],
}

IMAGE_MODEL = {
    "available": True,
    "model_type": "image",
    "model_name": "image-cnn",
    "classes": ["glioma", "meningioma", "notumor", "pituitary"],
    "feature_names": None,
}


@pytest.fixture()
def app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    return app


def _texts(app: AppTest) -> list[str]:
    collected: list[str] = []
    for elements in (
        app.markdown,
        app.subheader,
        app.header,
        app.info,
        app.warning,
        app.success,
        app.error,
        app.caption,
    ):
        collected.extend(element.value for element in elements)
    return collected


def test_dashboard_boots_without_exceptions(app):
    assert len(app.exception) == 0
    labels = [tab.label for tab in app.tabs]
    assert labels == TAB_LABELS


def test_assessment_form_submission_renders_results(monkeypatch):
    import dashboard.client as client_module

    def fake_analyze(
        self,
        patient,
        features,
        markers=None,
        recommendations=None,
        input_type="csv",
    ):
        return REPORT

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", lambda self: TABULAR_MODEL
    )
    monkeypatch.setattr(
        client_module.HealthcareAPIClient,
        "analyze",
        fake_analyze,
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app.radio[0].set_value(ROUTE_DIRECT).run()

    app.button[0].click().run()

    assert len(app.exception) == 0
    text = " ".join(_texts(app))
    assert "Clinical Results" in text
    assert "Disease Risk Score" in text
    assert "Mortality Risk" in text
    assert "Readmission Risk" in text
    assert "Treatment Recommendation" in text
    assert "Clinical Evidence" in text
    assert "Explainable Decision Report" in text
    assert "Model-estimated" in text


def test_n8n_route_uses_webhook(monkeypatch):
    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", lambda self: TABULAR_MODEL
    )
    calls = {}

    def fake_n8n(
        self,
        n8n_base_url,
        patient,
        features,
        markers=None,
        recommendations=None,
        input_type="csv",
        webhook=None,
    ):
        calls["n8n_base_url"] = n8n_base_url
        calls["features"] = features
        return REPORT

    monkeypatch.setattr(client_module.HealthcareAPIClient, "analyze_via_n8n", fake_n8n)

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app.radio[0].set_value(ROUTE_N8N).run()

    app.button[0].click().run()

    assert len(app.exception) == 0
    assert calls["n8n_base_url"] == "http://localhost:5678"
    assert "glucose" in calls["features"]
    text = " ".join(_texts(app))
    assert "Clinical Results" in text
    assert "n8n end-to-end workflow" in text


def test_unconfigured_model_is_graceful(monkeypatch):
    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient,
        "model_info",
        lambda self: {
            "available": False,
            "model_type": None,
            "model_name": None,
            "classes": None,
            "feature_names": None,
        },
    )

    def fake_analyze(
        self,
        patient,
        features,
        markers=None,
        recommendations=None,
        input_type="csv",
    ):
        return REPORT

    monkeypatch.setattr(
        client_module.HealthcareAPIClient,
        "analyze",
        fake_analyze,
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app.radio[0].set_value(ROUTE_DIRECT).run()

    app.button[0].click().run()

    assert len(app.exception) == 0
    text = " ".join(_texts(app))
    assert "no tabular model with known features is configured" in text.lower()


def test_imaging_unavailable_when_no_image_model(monkeypatch):
    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", lambda self: TABULAR_MODEL
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    assert len(app.exception) == 0
    text = " ".join(_texts(app))
    assert "not currently available" in text.lower()


def test_imaging_upload_analyzes_image(monkeypatch):
    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", lambda self: IMAGE_MODEL
    )
    monkeypatch.setattr(
        client_module.HealthcareAPIClient,
        "analyze_image",
        lambda self, patient, image, markers=None, recommendations=None: REPORT,
    )

    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (8, 8), (120, 120, 120)).save(buffer, format="PNG")

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    uploader = app.get("file_uploader")[0]
    uploader.set_value(("scan.png", buffer.getvalue(), "image/png")).run()

    app.button[1].click().run()

    assert len(app.exception) == 0
    text = " ".join(_texts(app))
    assert "Clinical Results" in text
    assert "Image analysis" in text


def test_results_tab_prompts_before_first_analysis(app):
    text = " ".join(_texts(app))
    assert "No analysis has been run yet" in text
