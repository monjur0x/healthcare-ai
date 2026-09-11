"""
Tests for model-derived explanations (P2.2).

Tabular models explain disease predictions with SHAP values from the
fitted estimator (exact ``LinearExplainer`` for logistic regression,
bounded ``KernelExplainer`` otherwise); the image CNN explains with
Grad-CAM. The magnitude-sort heuristic survives only as a labeled
fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CrewAI.orchestrator.crew import ClinicalCrew
from CrewAI.orchestrator.exceptions import ExplanationError
from CrewAI.orchestrator.explain import (
    attribute_tabular,
    grad_cam_heatmap,
    summarize_grad_cam,
)
from CrewAI.orchestrator.schemas import (
    PatientInfo,
    PredictionResult,
)
from CrewAI.orchestrator.services import build_explanation
from models import ImageClassifier, TabularClassifier

FEATURES = ["glucose", "bmi"]


@pytest.fixture
def frame() -> pd.DataFrame:
    """Strongly separable two-feature frame: glucose drives the label."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 2))
    x[:, 0] *= 2.0
    return pd.DataFrame(x, columns=FEATURES)


@pytest.fixture
def labels(frame) -> pd.Series:
    return pd.Series((frame["glucose"] > 0).astype(int), name="outcome")


@pytest.fixture
def logistic_model(frame, labels) -> TabularClassifier:
    return TabularClassifier(model_name="logistic").fit(frame, labels)


@pytest.fixture
def background(frame) -> pd.DataFrame:
    return frame.iloc[:20].copy()


@pytest.fixture
def disease_row() -> dict[str, float]:
    """Extreme disease-side row: glucose far above the decision boundary."""
    return {"glucose": 4.0, "bmi": 0.0}


def _prediction() -> PredictionResult:
    return PredictionResult(
        predicted_class="1",
        probabilities={"0": 0.1, "1": 0.9},
        confidence=0.9,
        model_name="logistic",
    )


def test_linear_shap_names_dominant_driver(
    logistic_model, disease_row, background
) -> None:
    attribution = attribute_tabular(logistic_model, disease_row, background=background)
    assert attribution.method == "shap_linear"
    assert attribution.predicted_class == "1"
    assert len(attribution.attributions) == 2
    # Ranks are a 1-based permutation by descending |SHAP|.
    assert sorted(a.abs_rank for a in attribution.attributions) == [1, 2]
    top = attribution.top(1)[0]
    assert top.feature == "glucose"
    assert top.direction == "toward"
    assert "glucose" in attribution.text(disease="diabetes")
    assert attribution.text().startswith("SHAP (shap_linear)")


def test_kernel_shap_covers_nonlinear_model(frame, labels, disease_row) -> None:
    model = TabularClassifier(model_name="gradient_boosting").fit(frame, labels)
    attribution = attribute_tabular(
        model,
        disease_row,
        background=frame.iloc[:15],
        nsamples=60,
    )
    assert attribution.method == "shap_kernel"
    assert len(attribution.attributions) == 2
    assert attribution.top(1)[0].feature == "glucose"
    assert attribution.text().startswith("SHAP (shap_kernel)")


def test_attribute_tabular_rejects_unfitted_model(disease_row, background) -> None:
    with pytest.raises(ExplanationError):
        attribute_tabular(
            TabularClassifier(model_name="logistic"),
            disease_row,
            background=background,
        )


def test_attribute_tabular_rejects_bad_background(logistic_model, disease_row) -> None:
    with pytest.raises(ExplanationError):
        attribute_tabular(
            logistic_model,
            disease_row,
            background=pd.DataFrame(columns=FEATURES),
        )
    with pytest.raises(ExplanationError):
        attribute_tabular(logistic_model, {"glucose": 1.0}, background=pd.DataFrame())


def test_build_explanation_uses_shap_with_model(
    logistic_model, disease_row, background
) -> None:
    text, contributing = build_explanation(
        _prediction(), disease_row, model=logistic_model, background=background
    )
    assert text.startswith("SHAP (shap_linear)")
    assert contributing and contributing[0] == "glucose"


def test_build_explanation_fallback_says_heuristic(disease_row) -> None:
    text, contributing = build_explanation(_prediction(), disease_row)
    assert "not model-derived" in text
    assert contributing == ["glucose", "bmi"]


def test_build_explanation_empty_without_prediction(disease_row) -> None:
    assert build_explanation(None, disease_row) == ("", [])


def test_crew_agent5_tags_shap_method(logistic_model, background) -> None:
    crew = ClinicalCrew(
        patient=PatientInfo(id="p-shap"),
        model=logistic_model,
        features={"glucose": 4.0, "bmi": 0.0},
        background=background,
    )
    crew.run_analysis()
    step5 = next(
        s
        for s in crew.crew_trace.steps
        if isinstance(s.output_data, dict) and "method" in s.output_data
    )
    assert step5.output_data["method"] == "shap_linear"


def test_crew_agent5_tags_heuristic_without_background(logistic_model) -> None:
    crew = ClinicalCrew(
        patient=PatientInfo(id="p-heur"),
        model=logistic_model,
        features={"glucose": 4.0, "bmi": 0.0},
    )
    crew.run_analysis()
    step5 = next(
        s
        for s in crew.crew_trace.steps
        if isinstance(s.output_data, dict) and "method" in s.output_data
    )
    assert step5.output_data["method"] == "magnitude_heuristic"


def _trained_cnn() -> tuple[ImageClassifier, np.ndarray]:
    rng = np.random.default_rng(5)
    images = rng.normal(size=(16, 12, 12, 3)).astype(np.float32)
    labels = np.repeat([0, 1], 8)
    images[np.arange(16), :6, :6, labels] += 2.0
    model = ImageClassifier(epochs=1, batch_size=8).fit(images, labels)
    return model, images


def test_grad_cam_heatmap_shape_and_range() -> None:
    model, images = _trained_cnn()
    heatmap = grad_cam_heatmap(model, images[0])
    assert heatmap.shape == (12, 12)
    assert heatmap.dtype == np.float32
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0
    summary = summarize_grad_cam(heatmap, "1")
    assert "Grad-CAM for class 1" in summary
    assert "quadrant" in summary


def test_grad_cam_rejects_unfitted_model() -> None:
    image = np.zeros((12, 12, 3), dtype=np.float32)
    with pytest.raises(ExplanationError):
        grad_cam_heatmap(ImageClassifier(), image)


def test_summarize_grad_cam_rejects_bad_heatmap() -> None:
    with pytest.raises(ExplanationError):
        summarize_grad_cam(np.zeros((0, 0)), "1")
