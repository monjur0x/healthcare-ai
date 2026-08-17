"""
Tests for the dashboard API client using a mocked HTTP transport.

The client is tested against canned backend responses so the tests are
hermetic (no network, no running server).
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from dashboard.client import APIConfig, HealthcareAPIClient, HealthcareAPIError


def _client(handler) -> HealthcareAPIClient:
    transport = httpx.MockTransport(handler)
    return HealthcareAPIClient(
        config=APIConfig(base_url="http://test"), transport=transport
    )


def _report():
    return {
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


def test_health_parses_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200, json={"status": "healthy", "name": "Backend", "version": "1.0.0"}
        )

    client = _client(handler)
    try:
        result = client.health()
    finally:
        client.close()
    assert result["status"] == "healthy"
    assert result["version"] == "1.0.0"


def test_predict_sends_features_and_parses():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/predict"
        assert request.read() is not None
        payload = json.loads(request.content)
        assert payload["features"] == {"glucose": 148.0}
        return httpx.Response(200, json=_report()["prediction"])

    client = _client(handler)
    try:
        result = client.predict({"glucose": 148.0})
    finally:
        client.close()
    assert result["predicted_class"] == "1"


def test_model_info_parses_metadata():
    payload = {
        "available": True,
        "model_type": "tabular_and_image",
        "model_name": "mlp",
        "classes": ["0", "1"],
        "feature_names": ["glucose", "bmi", "age"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/model"
        return httpx.Response(200, json=payload)

    client = _client(handler)
    try:
        result = client.model_info()
    finally:
        client.close()
    assert result["available"] is True
    assert result["feature_names"] == ["glucose", "bmi", "age"]


def test_analyze_image_base64_encodes_and_parses():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/analyze/image"
        payload = json.loads(request.content)
        assert payload["patient"]["id"] == "p-img"
        assert payload["image"].startswith("iVBOR")
        return httpx.Response(200, json=_report())

    client = _client(handler)
    try:
        result = client.analyze_image(
            patient={"id": "p-img", "name": "P", "age": 60},
            image=base64.b64decode("iVBORw0KGgoAAAANSUhEUg=="),
        )
    finally:
        client.close()
    assert result["input_type"] == "csv"


def test_analyze_image_omits_optional_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "markers" not in payload
        assert "recommendations" not in payload
        return httpx.Response(200, json=_report())

    client = _client(handler)
    try:
        client.analyze_image(patient={"id": "p-img"}, image=b"\x00\x01")
    finally:
        client.close()


def test_retrieve_forwards_query_and_top_k():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["query"] == "diabetes"
        assert payload["top_k"] == 2
        return httpx.Response(200, json=[_report()["evidence"][0]])

    client = _client(handler)
    try:
        result = client.retrieve("diabetes", top_k=2)
    finally:
        client.close()
    assert isinstance(result, list)
    assert result[0]["document_id"] == "diabetes.txt"


def test_analyze_builds_full_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        captured["url"] = request.url.path
        return httpx.Response(200, json=_report())

    client = _client(handler)
    try:
        result = client.analyze(
            patient={"id": "p-1", "name": "Patient"},
            features={"glucose": 148.0, "bmi": 27.3},
            markers={"glucose": 148.0},
            recommendations=["Review."],
            input_type="csv",
        )
    finally:
        client.close()

    assert captured["url"] == "/api/v1/analyze"
    assert captured["json"]["patient"]["id"] == "p-1"
    assert captured["json"]["features"]["bmi"] == 27.3
    assert captured["json"]["markers"]["glucose"] == 148.0
    assert captured["json"]["recommendations"] == ["Review."]
    assert captured["json"]["input_type"] == "csv"
    assert result["prediction"]["predicted_class"] == "1"


def test_analyze_omits_optional_fields_when_unset():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_report())

    client = _client(handler)
    try:
        client.analyze(patient={"id": "p-1"}, features={"glucose": 148.0})
    finally:
        client.close()

    assert "markers" not in captured["json"]
    assert "recommendations" not in captured["json"]


def test_error_response_raises_typed_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "detail": {
                    "code": "service_unavailable",
                    "message": "No model configured.",
                }
            },
        )

    client = _client(handler)
    try:
        with pytest.raises(HealthcareAPIError) as excinfo:
            client.predict({"glucose": 148.0})
    finally:
        client.close()

    assert excinfo.value.status_code == 503
    assert excinfo.value.code == "service_unavailable"
    assert "No model configured" in excinfo.value.message


def test_token_header_sent_when_configured():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer s3cret"
        return httpx.Response(200, json={"status": "healthy"})

    transport = httpx.MockTransport(handler)
    client = HealthcareAPIClient(
        config=APIConfig(base_url="http://test", api_token="s3cret"),
        transport=transport,
    )
    try:
        client.health()
    finally:
        client.close()


def test_analyze_via_n8n_posts_to_webhook_and_parses_report():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"status": "success", "report": _report()},
        )

    client = _client(handler)
    try:
        result = client.analyze_via_n8n(
            n8n_base_url="http://n8n:5678/",
            patient={"id": "p-1", "name": "Patient"},
            features={"glucose": 148.0},
            markers={"glucose": 148.0},
            input_type="csv",
        )
    finally:
        client.close()

    assert captured["url"] == "http://n8n:5678/webhook/healthcare-endtoend"
    assert captured["json"]["features"]["glucose"] == 148.0
    assert captured["json"]["markers"]["glucose"] == 148.0
    assert captured["json"]["input_type"] == "csv"
    assert result["prediction"]["predicted_class"] == "1"


def test_analyze_via_n8n_raises_on_workflow_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "error",
                "stage": "analysis_call",
                "error_message": "No prediction model configured",
            },
        )

    client = _client(handler)
    try:
        with pytest.raises(HealthcareAPIError) as excinfo:
            client.analyze_via_n8n(
                n8n_base_url="http://n8n:5678",
                patient={"id": "p-1"},
                features={"glucose": 148.0},
            )
    finally:
        client.close()

    assert excinfo.value.code == "n8n_workflow_error"
    assert "No prediction model configured" in excinfo.value.message


def test_analyze_via_n8n_raises_when_report_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success"})

    client = _client(handler)
    try:
        with pytest.raises(HealthcareAPIError) as excinfo:
            client.analyze_via_n8n(
                n8n_base_url="http://n8n:5678",
                patient={"id": "p-1"},
                features={"glucose": 148.0},
            )
    finally:
        client.close()

    assert excinfo.value.code == "n8n_report_missing"


def test_analyze_via_n8n_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "webhook not active"})

    client = _client(handler)
    try:
        with pytest.raises(HealthcareAPIError) as excinfo:
            client.analyze_via_n8n(
                n8n_base_url="http://n8n:5678",
                patient={"id": "p-1"},
                features={"glucose": 148.0},
            )
    finally:
        client.close()

    assert excinfo.value.status_code == 404


def test_n8n_health_true_when_healthz_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/healthz"
        return httpx.Response(200, text="OK")

    client = _client(handler)
    try:
        assert client.n8n_health("http://n8n:5678") is True
    finally:
        client.close()


def test_presets_parses_schemas():
    payload = [
        {
            "name": "diabetes",
            "dataset": "diabetes.csv",
            "target": "Outcome",
            "available": True,
            "feature_names": ["glucose", "bmi", "age"],
            "classes": ["0", "1"],
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/presets"
        return httpx.Response(200, json=payload)

    client = _client(handler)
    try:
        result = client.presets()
    finally:
        client.close()
    assert result[0]["name"] == "diabetes"
    assert result[0]["feature_names"] == ["glucose", "bmi", "age"]


def test_train_posts_preset():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model_path": "/tmp/diabetes/global_model.joblib",
                "dataset": "diabetes.csv",
                "target": "Outcome",
                "accuracy": 0.82,
                "roc_auc": 0.91,
                "f1": 0.78,
                "federated": False,
                "federated_metrics": None,
            },
        )

    client = _client(handler)
    try:
        result = client.train("diabetes", model="mlp")
    finally:
        client.close()
    assert captured["json"] == {"preset": "diabetes", "model": "mlp"}
    assert result["accuracy"] == 0.82


def test_analyze_csv_base64_encodes_and_parses():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/analyze/csv"
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=_report())

    client = _client(handler)
    try:
        result = client.analyze_csv(
            patient={"id": "p-csv", "name": "P"},
            csv=b"glucose,bmi,age\n150.0,25.0,55\n",
        )
    finally:
        client.close()
    assert captured["json"]["patient"]["id"] == "p-csv"
    assert captured["json"]["csv"] == base64.b64encode(
        b"glucose,bmi,age\n150.0,25.0,55\n"
    ).decode("ascii")
    assert captured["json"]["input_type"] == "csv"
    assert "markers" not in captured["json"]
    assert result["prediction"]["predicted_class"] == "1"


def test_analyze_via_n8n_includes_train_and_preset_when_requested():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "success", "report": _report()})

    client = _client(handler)
    try:
        client.analyze_via_n8n(
            n8n_base_url="http://n8n:5678",
            patient={"id": "p-1"},
            features={"glucose": 148.0},
            preset="heart",
            train=True,
        )
    finally:
        client.close()
    assert captured["json"]["train"] is True
    assert captured["json"]["preset"] == "heart"


def test_analyze_via_n8n_omits_train_preset_when_unset():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "success", "report": _report()})

    client = _client(handler)
    try:
        client.analyze_via_n8n(
            n8n_base_url="http://n8n:5678",
            patient={"id": "p-1"},
            features={"glucose": 148.0},
        )
    finally:
        client.close()
    assert "train" not in captured["json"]
    assert "preset" not in captured["json"]


def test_analyze_csv_via_n8n_posts_base64_and_parses_report():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "success", "report": _report()})

    client = _client(handler)
    try:
        result = client.analyze_csv_via_n8n(
            n8n_base_url="http://n8n:5678/",
            patient={"id": "p-csv", "name": "Patient"},
            csv=b"glucose,bmi,age\n150.0,25.0,55\n",
            markers={"glucose": 150.0},
            input_type="csv",
        )
    finally:
        client.close()

    assert captured["url"] == "http://n8n:5678/webhook/healthcare-endtoend"
    assert captured["json"]["csv_b64"] == base64.b64encode(
        b"glucose,bmi,age\n150.0,25.0,55\n"
    ).decode("ascii")
    assert captured["json"]["patient"]["id"] == "p-csv"
    assert captured["json"]["markers"]["glucose"] == 150.0
    assert captured["json"]["input_type"] == "csv"
    assert result["prediction"]["predicted_class"] == "1"


def test_analyze_csv_via_n8n_includes_train_and_preset_when_requested():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "success", "report": _report()})

    client = _client(handler)
    try:
        client.analyze_csv_via_n8n(
            n8n_base_url="http://n8n:5678",
            patient={"id": "p-csv"},
            csv=b"glucose,bmi,age\n150.0,25.0,55\n",
            preset="heart",
            train=True,
        )
    finally:
        client.close()
    assert captured["json"]["train"] is True
    assert captured["json"]["preset"] == "heart"
    assert "csv_b64" in captured["json"]


def test_analyze_csv_via_n8n_raises_on_workflow_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "error",
                "stage": "analysis_call",
                "error_message": "CSV columns do not match the model",
            },
        )

    client = _client(handler)
    try:
        with pytest.raises(HealthcareAPIError) as excinfo:
            client.analyze_csv_via_n8n(
                n8n_base_url="http://n8n:5678",
                patient={"id": "p-csv"},
                csv=b"bad,csv\n1,2\n",
            )
    finally:
        client.close()

    assert excinfo.value.code == "n8n_workflow_error"
    assert "CSV columns do not match" in excinfo.value.message


def test_n8n_health_false_when_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    try:
        assert client.n8n_health("http://n8n:5678") is False
    finally:
        client.close()
