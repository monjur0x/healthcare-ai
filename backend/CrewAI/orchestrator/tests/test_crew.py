"""
End-to-end tests for the deterministic clinical crew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CrewAI.orchestrator.crew import ClinicalCrew
from CrewAI.orchestrator.exceptions import OrchestrationError
from CrewAI.orchestrator.schemas import ClinicalReport, PatientInfo
from models import ImageClassifier, TabularClassifier
from rag import HashingEmbedder, RAGPipeline

CORPUS = [
    "diabetes is managed with metformin and lifestyle changes",
    "hypertension is treated with blood pressure lowering drugs",
    "sepsis requires broad-spectrum antibiotics within one hour",
]


@pytest.fixture
def model() -> TabularClassifier:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(150, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    return TabularClassifier(model_name="mlp").fit(
        pd.DataFrame(x, columns=["glucose", "bmi", "age"]), y
    )


@pytest.fixture
def pipeline() -> RAGPipeline:
    rag = RAGPipeline(embedder=HashingEmbedder(dims=64))
    rag.ingest_texts(CORPUS)
    return rag


def test_run_analysis_full_pipeline(model, pipeline) -> None:
    crew = ClinicalCrew(
        patient=PatientInfo(name="P", id="p1", age=60),
        model=model,
        features={"glucose": 1.0, "bmi": 2.0, "age": 0.5},
        rag_pipeline=pipeline,
        markers={"glucose": 140.0, "age": 60.0},
    )
    report = crew.run_analysis()
    assert isinstance(report, ClinicalReport)
    assert report.prediction is not None
    assert report.prediction.predicted_class
    assert report.risk is not None
    assert report.evidence
    assert report.context
    assert report.doctor_notice


def test_run_analysis_without_model_and_pipeline() -> None:
    crew = ClinicalCrew(patient=PatientInfo(id="p2"))
    report = crew.run_analysis()
    assert report.prediction is None
    assert report.risk is None
    assert report.evidence == []
    assert report.patient_summary


def test_run_prefers_deterministic_when_no_llm(model, monkeypatch) -> None:
    monkeypatch.setattr("CrewAI.orchestrator.crew.settings.LLM_API_KEY", "")
    crew = ClinicalCrew(
        patient=PatientInfo(id="p3"),
        model=model,
        features={"glucose": 1.0, "bmi": 2.0, "age": 0.5},
    )
    report = crew.run()
    assert report.prediction is not None


def test_run_analysis_requires_features_with_model(model) -> None:
    from CrewAI.orchestrator.exceptions import OrchestrationError

    crew = ClinicalCrew(patient=PatientInfo(id="p4"), model=model)
    with pytest.raises(OrchestrationError):
        crew.run_analysis()


@pytest.fixture
def image_model() -> ImageClassifier:
    images = np.zeros((16, 8, 8, 3), dtype=np.float32)
    images[:8, :, :, 0] = 1.0
    images[8:, :, :, 2] = 1.0
    labels = np.array(["tumor"] * 8 + ["normal"] * 8)
    return ImageClassifier(in_channels=3, epochs=2, batch_size=8, base_channels=2).fit(
        images, labels
    )


def test_run_analysis_image_path(image_model, pipeline) -> None:
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[:, :, 2] = 1.0
    crew = ClinicalCrew(
        patient=PatientInfo(name="P", id="p-img", age=60),
        input_type="image",
        image_model=image_model,
        image=image,
        rag_pipeline=pipeline,
        markers={"glucose": 140.0, "age": 60.0},
    )
    report = crew.run_analysis()
    assert isinstance(report, ClinicalReport)
    assert report.input_type == "image"
    assert report.prediction is not None
    assert report.prediction.model_name == "image-cnn"
    assert report.risk is not None
    assert report.evidence


def test_run_analysis_image_requires_image_array(image_model) -> None:
    crew = ClinicalCrew(patient=PatientInfo(id="p-img2"), image_model=image_model)
    with pytest.raises(OrchestrationError, match="image array"):
        crew.run_analysis()
