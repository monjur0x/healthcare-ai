"""
Tests for the deterministic prediction service.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CrewAI.orchestrator.exceptions import PredictionToolError
from CrewAI.orchestrator.schemas import PredictionResult
from CrewAI.orchestrator.services import run_image_prediction, run_prediction
from models import ImageClassifier, TabularClassifier


@pytest.fixture
def fitted_model() -> TabularClassifier:
    """A small deterministic logistic model with known feature columns."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 3))
    y = (x[:, 0] + 2.0 * x[:, 1] > 0).astype(int)
    model = TabularClassifier(model_name="logistic")
    model.fit(pd.DataFrame(x, columns=["glucose", "bmi", "age"]), y)
    return model


def test_run_prediction_returns_schema(fitted_model) -> None:
    result = run_prediction(fitted_model, {"glucose": 1.0, "bmi": 2.0, "age": 0.5})
    assert isinstance(result, PredictionResult)
    assert result.predicted_class in {"0", "1"}
    assert set(result.probabilities) == {"0", "1"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.model_name == "logistic"


def test_run_prediction_orders_by_feature_names(fitted_model) -> None:
    result = run_prediction(fitted_model, {"age": 0.5, "bmi": 2.0, "glucose": 1.0})
    expected = run_prediction(fitted_model, {"glucose": 1.0, "bmi": 2.0, "age": 0.5})
    assert result.probabilities == expected.probabilities


def test_run_prediction_missing_feature_raises(fitted_model) -> None:
    with pytest.raises(PredictionToolError, match="Missing feature"):
        run_prediction(fitted_model, {"glucose": 1.0, "bmi": 2.0})


def test_run_prediction_unfitted_model_raises() -> None:
    model = TabularClassifier(model_name="logistic")
    with pytest.raises(PredictionToolError):
        run_prediction(model, {"a": 1.0, "b": 2.0})


def _fitted_image_model() -> ImageClassifier:
    """A tiny CNN fitted on two synthetic classes."""
    images = np.zeros((16, 8, 8, 3), dtype=np.float32)
    images[:8, :, :, 0] = 1.0
    images[8:, :, :, 2] = 1.0
    labels = np.array(["glioma"] * 8 + ["normal"] * 8)
    return ImageClassifier(
        in_channels=3,
        epochs=2,
        batch_size=8,
        base_channels=2,
    ).fit(images, labels)


def test_run_image_prediction_returns_schema() -> None:
    image_model = _fitted_image_model()
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[:, :, 0] = 1.0
    result = run_image_prediction(image_model, image)
    assert isinstance(result, PredictionResult)
    assert result.predicted_class in {"glioma", "normal"}
    assert set(result.probabilities) == {"glioma", "normal"}
    assert 0.0 <= result.confidence <= 1.0
    assert result.model_name == "image-cnn"


def test_run_image_prediction_malformed_array_raises() -> None:
    image_model = _fitted_image_model()
    with pytest.raises(PredictionToolError, match="3 dimensions"):
        run_image_prediction(image_model, np.zeros((8, 8), dtype=np.float32))


def test_run_image_prediction_unfitted_model_raises() -> None:
    model = ImageClassifier(in_channels=3)
    image = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(PredictionToolError):
        run_image_prediction(model, image)
