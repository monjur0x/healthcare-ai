"""
Service-layer tests for the FastAPI module.

These exercise the real ``AnalysisService`` wiring (model load + RAG
corpus ingest + crew analysis) without any network or LLM dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.config import APISettings
from api.exceptions import InvalidInputError, ServiceUnavailableError
from api.services import (
    AnalysisService,
    build_rag_pipeline,
    load_predictive_model,
    prepare_tabular_data,
)
from CrewAI.orchestrator.schemas import PatientInfo
from models import TabularClassifier


def _trained_model(tmp_path) -> TabularClassifier:
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {
            "glucose": rng.uniform(70, 200, n),
            "bmi": rng.uniform(18, 40, n),
            "age": rng.integers(20, 80, n),
        }
    )
    labels = pd.Series((frame["glucose"] > 126).astype(int), name="outcome")
    model = TabularClassifier(model_name="logistic").fit(frame, labels)
    model.save(tmp_path / "model.joblib")
    return model


def _corpus(tmp_path) -> None:
    (tmp_path / "diabetes.txt").write_text(
        "diabetes mellitus is managed with metformin, lifestyle changes, "
        "and regular glucose monitoring",
        encoding="utf-8",
    )
    (tmp_path / "hypertension.txt").write_text(
        "chronic hypertension management combines dietary sodium reduction "
        "and blood pressure lowering medication",
        encoding="utf-8",
    )


def test_from_settings_loads_model_and_corpus(tmp_path):
    _trained_model(tmp_path)
    _corpus(tmp_path)
    cfg = APISettings(
        _env_file=None,
        MODEL_PATH=str(tmp_path / "model.joblib"),
        CORPUS_DIR=str(tmp_path),
    )
    service = AnalysisService.from_settings(cfg)
    assert service.model is not None
    assert service.rag_pipeline is not None


def test_predict_returns_structured_result(tmp_path):
    model = _trained_model(tmp_path)
    service = AnalysisService(model=model)
    result = service.predict({"glucose": 150.0, "bmi": 25.0, "age": 55.0})
    assert result.predicted_class in {"0", "1"}
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == {"0", "1"}


def test_predict_without_model_raises_service_unavailable():
    service = AnalysisService(model=None)
    with pytest.raises(ServiceUnavailableError):
        service.predict({"glucose": 120.0})


def test_predict_with_missing_feature_raises_invalid_input(tmp_path):
    model = _trained_model(tmp_path)
    service = AnalysisService(model=model)
    with pytest.raises(InvalidInputError):
        service.predict({"glucose": 120.0})


def test_retrieve_returns_evidence(tmp_path):
    _corpus(tmp_path)
    service = AnalysisService(rag_pipeline=build_rag_pipeline(tmp_path))
    evidence = service.retrieve("diabetes management")
    assert len(evidence) >= 1
    assert all(item.text for item in evidence)


def test_retrieve_without_pipeline_raises_service_unavailable():
    service = AnalysisService(rag_pipeline=None)
    with pytest.raises(ServiceUnavailableError):
        service.retrieve("diabetes")


def test_analyze_returns_full_report(tmp_path):
    model = _trained_model(tmp_path)
    _corpus(tmp_path)
    service = AnalysisService(model=model, rag_pipeline=build_rag_pipeline(tmp_path))
    report = service.analyze(
        patient=PatientInfo(name="Test", id="p-1"),
        features={"glucose": 150.0, "bmi": 25.0, "age": 55.0},
        markers={"glucose": 150.0, "bmi": 25.0},
    )
    assert report.patient.id == "p-1"
    assert report.prediction is not None
    assert report.risk is not None
    assert report.evidence
    assert report.recommendations == []


def test_analyze_without_model_still_builds_report(tmp_path):
    _corpus(tmp_path)
    service = AnalysisService(rag_pipeline=build_rag_pipeline(tmp_path))
    report = service.analyze(
        patient=PatientInfo(name="Test", id="p-2"),
        features={"glucose": 120.0},
    )
    assert report.patient.id == "p-2"
    assert report.prediction is None
    assert report.risk is None


def test_load_predictive_model_missing_path_raises(tmp_path):
    with pytest.raises(ServiceUnavailableError):
        load_predictive_model(tmp_path / "missing.joblib")


def _write_csv(tmp_path, n=80, name="dataset.csv") -> Path:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "glucose": rng.uniform(70, 200, n),
            "bmi": rng.uniform(18, 40, n),
            "age": rng.integers(20, 80, n),
        }
    )
    frame["Outcome"] = (frame["glucose"] > 126).astype(int)
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_prepare_tabular_data_returns_features_and_labels(tmp_path):
    dataset = _write_csv(tmp_path)
    features, labels = prepare_tabular_data(dataset, "Outcome", max_rows=None)
    assert features.shape[0] == labels.shape[0] == 80
    assert "outcome" not in features.columns
    assert set(labels.unique()) == {0, 1}


def test_prepare_tabular_data_missing_target_raises(tmp_path):
    dataset = _write_csv(tmp_path)
    with pytest.raises(InvalidInputError):
        prepare_tabular_data(dataset, "missing_col", max_rows=None)


def test_train_central_fits_and_saves(tmp_path):
    dataset = _write_csv(tmp_path)
    service = AnalysisService(artifacts_dir=tmp_path / "artifacts")
    result = service.train(dataset=str(dataset), target="Outcome", model="logistic")
    assert result.model_path.endswith("global_model.joblib")
    assert result.federated is False
    assert 0.0 <= result.accuracy <= 1.0
    assert service.model is not None
    prediction = service.predict({"glucose": 150.0, "bmi": 25.0, "age": 55.0})
    assert prediction.predicted_class in {"0", "1"}


def test_train_preset_resolves_and_updates_service(tmp_path):
    _write_csv(tmp_path, name="diabetes.csv")
    service = AnalysisService(
        artifacts_dir=tmp_path / "artifacts",
        dataset_dir=tmp_path,
    )
    service.train(preset="diabetes", model="logistic")
    assert service.model is not None


def test_train_without_preset_or_dataset_raises(tmp_path):
    service = AnalysisService(artifacts_dir=tmp_path / "artifacts")
    with pytest.raises(InvalidInputError):
        service.train()


def test_train_unknown_preset_raises(tmp_path):
    service = AnalysisService(artifacts_dir=tmp_path / "artifacts")
    with pytest.raises(InvalidInputError):
        service.train(preset="unknown")


def test_train_missing_dataset_file_raises(tmp_path):
    service = AnalysisService(artifacts_dir=tmp_path / "artifacts")
    with pytest.raises(InvalidInputError):
        service.train(dataset=str(tmp_path / "nope.csv"), target="Outcome")


def test_train_federated_aggregates_global_model(tmp_path):
    dataset = _write_csv(tmp_path, n=120)
    service = AnalysisService(artifacts_dir=tmp_path / "artifacts")
    result = service.train(
        dataset=str(dataset),
        target="Outcome",
        model="mlp",
        federated=True,
        clients=3,
        rounds=2,
    )
    assert result.federated is True
    assert result.federated_metrics is not None
    assert service.model is not None
    prediction = service.predict({"glucose": 150.0, "bmi": 25.0, "age": 55.0})
    assert prediction.predicted_class in {"0", "1"}


def test_train_federated_rejects_non_mlp(tmp_path):
    dataset = _write_csv(tmp_path)
    service = AnalysisService(artifacts_dir=tmp_path / "artifacts")
    with pytest.raises(InvalidInputError):
        service.train(
            dataset=str(dataset), target="Outcome", model="logistic", federated=True
        )


def test_train_federated_with_differential_privacy_and_secure_aggregation(tmp_path):
    dataset = _write_csv(tmp_path, n=120)
    service = AnalysisService(artifacts_dir=tmp_path / "artifacts")
    result = service.train(
        dataset=str(dataset),
        target="Outcome",
        model="mlp",
        federated=True,
        clients=3,
        rounds=2,
        differential_privacy=True,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
        privacy_delta=1e-5,
        secure_aggregation=True,
    )
    assert result.federated is True
    assert result.federated_metrics is not None
    privacy = result.federated_metrics["privacy"]
    assert privacy["epsilon"] > 0.0
    assert privacy["secure_aggregation"] is True
    assert 0.0 <= privacy["attack_resistance_score"] <= 1.0
    assert privacy["data_leakage_rate"] == 0.0
    assert "DP-SGD" in privacy["mechanism"]
    assert "Secure Aggregation" in privacy["mechanism"]
    assert service.model is not None
    prediction = service.predict({"glucose": 150.0, "bmi": 25.0, "age": 55.0})
    assert prediction.predicted_class in {"0", "1"}


def test_train_federated_with_secure_aggregation_only(tmp_path):
    dataset = _write_csv(tmp_path, n=120)
    service = AnalysisService(artifacts_dir=tmp_path / "artifacts")
    result = service.train(
        dataset=str(dataset),
        target="Outcome",
        model="mlp",
        federated=True,
        clients=3,
        rounds=2,
        secure_aggregation=True,
    )
    assert result.federated_metrics["secure_aggregation"] is True
    assert "privacy" not in result.federated_metrics
