"""
API route tests using FastAPI's TestClient with a hermetic fake service.

Routes must not contain business logic, so a fake service is enough to
verify validation, serialization, auth, and error mapping.
"""

from __future__ import annotations

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
