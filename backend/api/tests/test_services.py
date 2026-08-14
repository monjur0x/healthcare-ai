"""
Service-layer tests for the FastAPI module.

These exercise the real ``AnalysisService`` wiring (model load + RAG
corpus ingest + crew analysis) without any network or LLM dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.config import APISettings
from api.exceptions import InvalidInputError, ServiceUnavailableError
from api.services import (
    AnalysisService,
    build_rag_pipeline,
    load_predictive_model,
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
