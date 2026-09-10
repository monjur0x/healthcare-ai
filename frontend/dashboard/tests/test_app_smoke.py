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
    "Federation",
    "Risk Monitoring",
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

PRESETS = [
    {
        "name": "diabetes",
        "dataset": "diabetes.csv",
        "target": "Outcome",
        "available": True,
        "feature_names": ["glucose", "bmi", "age"],
        "classes": ["0", "1"],
    },
    {
        "name": "heart",
        "dataset": "heart_disease_uci.csv",
        "target": "num",
        "available": True,
        "feature_names": ["trestbps", "chol", "age"],
        "classes": ["0", "1"],
    },
    {
        "name": "kidney",
        "dataset": "kidney_disease.csv",
        "target": "class",
        "available": True,
        "feature_names": ["age", "blood_pressure", "albumin"],
        "classes": ["0", "1"],
    },
    {
        "name": "sepsis",
        "dataset": "sepsis.csv",
        "target": "sepsis_label",
        "available": True,
        "feature_names": ["age", "heart_rate", "glucose"],
        "classes": ["0", "1"],
    },
]


def _route_radio(app: AppTest):
    return next(radio for radio in app.radio if radio.label == "Analysis route")


def _assessment_submit(app: AppTest):
    return next(b for b in app.button if b.label == "Run Clinical Assessment")


def _set_features(app: AppTest, values: dict[str, float]) -> None:
    """Set keyed feature inputs via session_state (deterministic in AppTest)."""
    for name, value in values.items():
        app.session_state[name] = value
    app.run()


@pytest.fixture()
def app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    return app


def _route_radio(app: AppTest):
    return next(radio for radio in app.radio if radio.label == "Analysis route")


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
    # st.metric renders a label instead of markdown; include it.
    collected.extend(f"{element.label} {element.value}" for element in app.metric)
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
        client_module.HealthcareAPIClient, "presets", lambda self: PRESETS
    )
    monkeypatch.setattr(
        client_module.HealthcareAPIClient,
        "analyze",
        fake_analyze,
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    _route_radio(app).set_value(ROUTE_DIRECT).run()

    # Fill the required feature inputs with plausible values.
    _set_features(
        app,
        {
            "patient_age": 45,
            "feature_glucose": 150.0,
            "feature_bmi": 25.0,
        },
    )
    _assessment_submit(app).click().run()

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
    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "presets", lambda self: PRESETS
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
    _route_radio(app).set_value(ROUTE_N8N).run()

    _set_features(
        app,
        {
            "patient_age": 45,
            "feature_glucose": 150.0,
            "feature_bmi": 25.0,
        },
    )
    _assessment_submit(app).click().run()

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
            "preset": None,
        },
    )
    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "presets", lambda self: []
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

    image_button = next(
        (b for b in app.button if b.label == "Analyze Image"),
        None,
    )
    image_button.click().run()

    assert len(app.exception) == 0
    text = " ".join(_texts(app))
    assert "Clinical Results" in text
    assert "Image analysis" in text


def _set_csv_upload(app: AppTest, csv_bytes: bytes) -> None:
    """Switch the assessment to CSV mode and upload file bytes."""
    input_radio = next(r for r in app.radio if r.label == "Input method")
    input_radio.set_value("CSV Upload").run()
    uploader = next(f for f in app.file_uploader if f.label == "Clinical data CSV")
    uploader.set_value(("data.csv", csv_bytes, "text/csv")).run()


def test_assessment_csv_upload_routes_through_n8n(monkeypatch):
    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", lambda self: TABULAR_MODEL
    )
    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "presets", lambda self: PRESETS
    )
    calls: dict[str, object] = {}

    def fake_analyze_csv_via_n8n(
        self,
        n8n_base_url,
        patient,
        csv,
        markers=None,
        recommendations=None,
        input_type="csv",
        webhook="healthcare-endtoend",
        preset=None,
        train=False,
    ):
        calls["csv"] = csv
        calls["preset"] = preset
        calls["train"] = train
        return REPORT

    monkeypatch.setattr(
        client_module.HealthcareAPIClient,
        "analyze_csv_via_n8n",
        fake_analyze_csv_via_n8n,
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    _route_radio(app).set_value(ROUTE_N8N).run()
    _set_csv_upload(app, b"glucose,bmi,age\n150,25,45\n")

    _assessment_submit(app).click().run()

    assert len(app.exception) == 0
    assert calls["csv"] == b"glucose,bmi,age\n150,25,45\n"
    assert calls["train"] is False
    assert calls["preset"] is None
    text = " ".join(_texts(app))
    assert "Clinical Results" in text
    assert "n8n end-to-end workflow" in text


def test_results_tab_prompts_before_first_analysis(app):
    text = " ".join(_texts(app))
    assert "No analysis has been run yet" in text


def _number_labels(app: AppTest) -> list[str]:
    return [element.label for element in app.number_input]


def test_assessment_type_selector_adapts_form(monkeypatch):
    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", lambda self: TABULAR_MODEL
    )
    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "presets", lambda self: PRESETS
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    assert len(app.exception) == 0
    text = " ".join(_texts(app))
    assert "Assessment Type" in text
    assert "Glucose (mg/dL)" in _number_labels(app)
    assert "Resting Blood Pressure (mmHg)" not in _number_labels(app)

    app.selectbox[0].set_value("heart").run()

    assert "Resting Blood Pressure (mmHg)" in _number_labels(app)
    assert "Glucose (mg/dL)" not in _number_labels(app)
    text = " ".join(_texts(app))
    assert "train/serve that model first" in text


def test_assessment_train_on_demand_for_other_preset(monkeypatch):
    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", lambda self: TABULAR_MODEL
    )
    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "presets", lambda self: PRESETS
    )
    calls: dict[str, object] = {}

    def fake_train(self, preset, model="mlp"):
        calls["train"] = preset
        return {"status": "ok"}

    def fake_analyze(
        self,
        patient,
        features,
        markers=None,
        recommendations=None,
        input_type="csv",
    ):
        calls["features"] = features
        calls["patient"] = patient
        return REPORT

    monkeypatch.setattr(client_module.HealthcareAPIClient, "train", fake_train)
    monkeypatch.setattr(client_module.HealthcareAPIClient, "analyze", fake_analyze)

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    _route_radio(app).set_value(ROUTE_DIRECT).run()
    app.selectbox[0].set_value("heart").run()

    _set_features(
        app,
        {
            "patient_age": 45,
            "feature_trestbps": 120.0,
            "feature_chol": 200.0,
        },
    )
    _assessment_submit(app).click().run()

    assert len(app.exception) == 0
    assert calls["train"] == "heart"
    assert "trestbps" in calls["features"]
    assert calls["features"]["age"] == 45.0
    assert calls["patient"]["name"] == "Patient"
    assert calls["patient"]["age"] == 45
    assert "name" not in calls["features"]
    assert "id" not in calls["features"]


def test_assessment_csv_upload_analyzes(monkeypatch):
    import dashboard.client as client_module

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "model_info", lambda self: TABULAR_MODEL
    )
    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "presets", lambda self: PRESETS
    )
    calls: dict[str, object] = {}

    def fake_analyze_csv(self, patient, csv, markers=None, recommendations=None):
        calls["patient"] = patient
        calls["csv"] = csv
        return REPORT

    monkeypatch.setattr(
        client_module.HealthcareAPIClient, "analyze_csv", fake_analyze_csv
    )

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    _route_radio(app).set_value(ROUTE_DIRECT).run()
    _set_csv_upload(app, b"glucose,bmi,age\n150,25,45\n")

    _assessment_submit(app).click().run()

    assert len(app.exception) == 0
    assert calls["csv"] == b"glucose,bmi,age\n150,25,45\n"
    assert calls["patient"]["name"] == "Patient"
    text = " ".join(_texts(app))
    assert "Clinical Results" in text
    assert "directly to the FastAPI backend" in text
