"""
API route tests using FastAPI's TestClient with a hermetic fake service.

Routes must not contain business logic, so a fake service is enough to
verify validation, serialization, auth, and error mapping.
"""

from __future__ import annotations

import base64

import pytest

from fastapi.testclient import TestClient

from api.config import APISettings
from api.exceptions import ServiceUnavailableError
from api.main import create_app
from api.services import AnalysisService, TrainResult
from CrewAI.orchestrator.schemas import (
    ClinicalReport,
    EvidenceItem,
    PredictionResult,
)

FAKE_IMAGE = base64.b64encode(b"\x89PNG\r\n\x1a\nfakedata").decode("ascii")


class FakeService(AnalysisService):
    """Hermetic service returning fixed, valid results."""

    def __init__(self) -> None:
        super().__init__(model=None, rag_pipeline=None)

    def predict(self, features):
        return PredictionResult(
            predicted_class="1",
            probabilities={"0": 0.3, "1": 0.7},
            confidence=0.7,
            model_name="fake",
        )

    def retrieve(self, query, top_k=None):
        return [
            EvidenceItem(
                document_id="diabetes.txt",
                source="protocols",
                score=0.9,
                text="diabetes is managed with metformin",
            )
        ]

    def analyze(
        self, patient, features, markers=None, recommendations=None, input_type="csv"
    ):
        return ClinicalReport(
            patient=patient,
            input_type=input_type,
            patient_summary="Analysis completed.",
            prediction=PredictionResult(
                predicted_class="1",
                probabilities={"0": 0.3, "1": 0.7},
                confidence=0.7,
                model_name="fake",
            ),
            risk=None,
            evidence=[],
            recommendations=list(recommendations or []),
        )

    def train(self, preset=None, dataset=None, target=None, **kwargs):
        return TrainResult(
            model_path="/tmp/fake/global_model.joblib",
            dataset="diabetes.csv",
            target="outcome",
            accuracy=0.82,
            roc_auc=0.91,
            f1=0.78,
            federated=bool(kwargs.get("federated", False)),
            federated_metrics=None,
        )

    def analyze_image(
        self, patient, image, markers=None, recommendations=None, **kwargs
    ):
        return ClinicalReport(
            patient=patient,
            input_type="image",
            patient_summary="Image analysis completed.",
            prediction=PredictionResult(
                predicted_class="1",
                probabilities={"0": 0.2, "1": 0.8},
                confidence=0.8,
                model_name="image-cnn",
            ),
            risk=None,
            evidence=[],
            recommendations=list(recommendations or []),
        )

    def model_info(self):
        return {
            "available": True,
            "model_type": "tabular_and_image",
            "model_name": "mlp",
            "classes": ["0", "1"],
            "feature_names": ["glucose", "bmi", "age"],
            "preset": "diabetes",
        }

    def presets_info(self):
        return [
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
                "available": False,
                "feature_names": None,
                "classes": None,
            },
        ]

    def analyze_csv(self, patient, csv, markers=None, recommendations=None, **kwargs):
        return ClinicalReport(
            patient=patient,
            input_type="csv",
            patient_summary="CSV analysis completed.",
            prediction=PredictionResult(
                predicted_class="1",
                probabilities={"0": 0.2, "1": 0.8},
                confidence=0.8,
                model_name="fake",
            ),
            risk=None,
            evidence=[],
            recommendations=list(recommendations or []),
        )


@pytest.fixture()
def client() -> TestClient:
    app = create_app(cfg=APISettings(_env_file=None), service=FakeService())
    return TestClient(app)


def test_root_returns_metadata(client):
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["name"] == "Healthcare AI Backend"


def test_health_returns_healthy(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_returns_prediction(client):
    response = client.post("/api/v1/predict", json={"features": {"glucose": 150.0}})
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] == "1"
    assert payload["confidence"] == 0.7


def test_predict_missing_features_is_422(client):
    response = client.post("/api/v1/predict", json={})
    assert response.status_code == 422


def test_retrieve_returns_evidence(client):
    response = client.post("/api/v1/retrieve", json={"query": "diabetes"})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload[0]["document_id"] == "diabetes.txt"


def test_retrieve_empty_query_is_422(client):
    response = client.post("/api/v1/retrieve", json={"query": ""})
    assert response.status_code == 422


def test_analyze_returns_report(client):
    response = client.post(
        "/api/v1/analyze",
        json={
            "patient": {"name": "Test", "id": "p-1"},
            "features": {"glucose": 150.0},
            "recommendations": ["Review with a physician."],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["patient"]["id"] == "p-1"
    assert payload["prediction"]["predicted_class"] == "1"
    assert payload["recommendations"] == ["Review with a physician."]


def test_analyze_invalid_patient_is_422(client):
    response = client.post(
        "/api/v1/analyze",
        json={"patient": {"age": "not-an-int"}},
    )
    assert response.status_code == 422


def test_train_returns_metrics(client):
    response = client.post("/api/v1/train", json={"preset": "diabetes"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_path"].endswith("global_model.joblib")
    assert payload["dataset"] == "diabetes.csv"
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert payload["federated"] is False


def test_train_accepts_explicit_dataset_and_federated(client):
    response = client.post(
        "/api/v1/train",
        json={"dataset": "data.csv", "target": "outcome", "federated": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["federated"] is True


def test_train_invalid_preset_is_422(client):
    response = client.post("/api/v1/train", json={"preset": "unknown"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "preset"]


def test_train_bad_model_choice_is_422(client):
    response = client.post("/api/v1/train", json={"preset": "diabetes", "model": "svm"})
    assert response.status_code == 422


def test_train_rounds_out_of_range_is_422(client):
    response = client.post("/api/v1/train", json={"preset": "diabetes", "rounds": 0})
    assert response.status_code == 422


def test_service_error_maps_to_503():
    class UnavailableService(FakeService):
        def retrieve(self, query, top_k=None):
            raise ServiceUnavailableError("No retrieval pipeline is configured.")

    app = create_app(cfg=APISettings(_env_file=None), service=UnavailableService())
    response = TestClient(app).post("/api/v1/retrieve", json={"query": "diabetes"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "service_unavailable"


def test_analyze_image_returns_report(client):
    response = client.post(
        "/api/v1/analyze/image",
        json={
            "patient": {"id": "p-img", "name": "P", "age": 60},
            "image": FAKE_IMAGE,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["input_type"] == "image"
    assert payload["prediction"]["model_name"] == "image-cnn"


def test_analyze_image_missing_image_is_422(client):
    response = client.post("/api/v1/analyze/image", json={"patient": {"id": "p-img"}})
    assert response.status_code == 422


def test_analyze_image_error_maps_to_503():
    class UnavailableService(FakeService):
        def analyze_image(
            self, patient, image, markers=None, recommendations=None, **kwargs
        ):
            raise ServiceUnavailableError("No image model is configured.")

    app = create_app(cfg=APISettings(_env_file=None), service=UnavailableService())
    response = TestClient(app).post(
        "/api/v1/analyze/image",
        json={"patient": {"id": "p-img"}, "image": FAKE_IMAGE},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "service_unavailable"


def test_model_info_returns_metadata(client):
    response = client.get("/api/v1/model")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["model_type"] == "tabular_and_image"
    assert payload["feature_names"] == ["glucose", "bmi", "age"]
    assert payload["preset"] == "diabetes"


def test_model_info_returns_unavailable_when_no_model():
    class EmptyService(FakeService):
        def model_info(self):
            return {
                "available": False,
                "model_type": None,
                "model_name": None,
                "classes": None,
                "feature_names": None,
                "preset": None,
            }

    app = create_app(cfg=APISettings(_env_file=None), service=EmptyService())
    response = TestClient(app).get("/api/v1/model")
    assert response.status_code == 200
    assert response.json()["available"] is False


def test_presets_returns_feature_schemas(client):
    response = client.get("/api/v1/presets")
    assert response.status_code == 200
    payload = response.json()
    assert [item["name"] for item in payload] == ["diabetes", "heart"]
    diabetes = payload[0]
    assert diabetes["available"] is True
    assert diabetes["feature_names"] == ["glucose", "bmi", "age"]
    heart = payload[1]
    assert heart["available"] is False
    assert heart["feature_names"] is None


def test_analyze_csv_returns_report(client):
    response = client.post(
        "/api/v1/analyze/csv",
        json={
            "patient": {"id": "p-csv", "name": "P"},
            "csv": base64.b64encode(b"glucose,bmi,age\n150.0,25.0,55\n").decode(
                "ascii"
            ),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["patient"]["id"] == "p-csv"
    assert payload["input_type"] == "csv"
    assert payload["prediction"]["confidence"] == 0.8


def test_analyze_csv_missing_csv_is_422(client):
    response = client.post("/api/v1/analyze/csv", json={"patient": {"id": "p-csv"}})
    assert response.status_code == 422


def test_analyze_csv_invalid_base64_is_422(client):
    response = client.post(
        "/api/v1/analyze/csv",
        json={"patient": {"id": "p-csv"}, "csv": "not!base64!"},
    )
    assert response.status_code == 422


def test_analyze_csv_error_maps_to_503():
    class UnavailableService(FakeService):
        def analyze_csv(
            self, patient, csv, markers=None, recommendations=None, **kwargs
        ):
            raise ServiceUnavailableError("No tabular model is configured.")

    app = create_app(cfg=APISettings(_env_file=None), service=UnavailableService())
    response = TestClient(app).post(
        "/api/v1/analyze/csv",
        json={"patient": {"id": "p-csv"}, "csv": "Z2x1Y29zZQ=="},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "service_unavailable"


def test_token_required_when_configured():
    app = create_app(
        cfg=APISettings(_env_file=None, API_TOKEN="s3cret"),
        service=FakeService(),
    )
    client = TestClient(app)

    no_auth = client.post("/api/v1/predict", json={"features": {"glucose": 150.0}})
    assert no_auth.status_code == 401
    assert no_auth.json()["detail"]["code"] == "unauthorized"

    bad_auth = client.post(
        "/api/v1/predict",
        json={"features": {"glucose": 150.0}},
        headers={"Authorization": "Bearer wrong"},
    )
    assert bad_auth.status_code == 401

    good_auth = client.post(
        "/api/v1/predict",
        json={"features": {"glucose": 150.0}},
        headers={"Authorization": "Bearer s3cret"},
    )
    assert good_auth.status_code == 200

    assert client.get("/health").status_code == 200
