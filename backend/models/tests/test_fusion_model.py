"""
Tests for the multimodal fusion prediction model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import FusionClassifier
from models.exceptions import (
    InvalidModelInputError,
    ModelLoadError,
    ModelNotFittedError,
)
from preprocessing.multimodal import FusionResult, MultimodalFusion


def _make_fusion(
    n_samples: int = 24,
) -> tuple[FusionResult, np.ndarray]:
    """Build a fused multimodal result with separable labels."""
    rng = np.random.default_rng(0)
    n = n_samples
    labels = np.repeat([0, 1], n // 2)
    dataframe = pd.DataFrame(
        {
            "age": np.concatenate(
                [rng.normal(40, 3, n // 2), rng.normal(60, 3, n // 2)]
            ),
            "bmi": np.concatenate(
                [rng.normal(22, 3, n // 2), rng.normal(30, 3, n // 2)]
            ),
            "patient_id": [f"P{index}" for index in range(n)],
        }
    )
    images = rng.normal(size=(n, 12, 12, 3)).astype(np.float32)
    for index, label in enumerate(labels):
        images[index, ..., label] += 3.0 * (label + 1)

    fusion = MultimodalFusion(image_reduction="summary")
    result = fusion.transform(dataframe, images)
    return result, labels


@pytest.fixture
def fusion_data() -> tuple[FusionResult, np.ndarray]:
    return _make_fusion()


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions == labels))


def test_fusion_predict_on_result(fusion_data) -> None:
    result, y = fusion_data
    model = FusionClassifier().fit(result, y)
    assert model.is_fitted
    preds = model.predict(result)
    assert preds.shape == (24,)
    assert _accuracy(preds, y) > 0.75


def test_fusion_predict_proba(fusion_data) -> None:
    result, y = fusion_data
    model = FusionClassifier().fit(result, y)
    proba = model.predict_proba(result)
    assert proba.shape == (24, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_fusion_requires_labels(fusion_data) -> None:
    result, _ = fusion_data
    with pytest.raises(InvalidModelInputError):
        FusionClassifier().fit(result, None)


def test_fusion_raw_matrix(fusion_data) -> None:
    result, y = fusion_data
    model = FusionClassifier().fit(result.fused, y)
    assert model.fused_dim == result.fused.shape[1]
    assert model.predict(result.fused).shape == (24,)


def test_fusion_feature_names(fusion_data) -> None:
    result, y = fusion_data
    model = FusionClassifier().fit(result, y)
    names = model.feature_names
    assert names is not None
    assert len(names) == model.fused_dim
    assert names[0] == "fused_0"


def test_fusion_requires_fit(fusion_data) -> None:
    _result, _labels = fusion_data
    with pytest.raises(ModelNotFittedError):
        FusionClassifier().predict(_result)


def test_fusion_invalid_ndim(fusion_data) -> None:
    result, y = fusion_data
    with pytest.raises(InvalidModelInputError):
        FusionClassifier().fit(result.fused.reshape(-1), y)


def test_fusion_save_load_roundtrip(fusion_data, tmp_path) -> None:
    result, y = fusion_data
    model = FusionClassifier().fit(result, y)
    original = model.predict(result)
    target = tmp_path / "fusion.joblib"

    model.save(target)
    loaded = FusionClassifier.load(target)

    assert loaded.is_fitted
    assert loaded.model_name == "mlp"
    np.testing.assert_array_equal(loaded.predict(result), original)


def test_fusion_load_missing_file(tmp_path) -> None:
    with pytest.raises(ModelLoadError):
        FusionClassifier.load(tmp_path / "missing.joblib")


def test_fusion_save_requires_fit(fusion_data, tmp_path) -> None:
    _result, _labels = fusion_data
    with pytest.raises(ModelNotFittedError):
        FusionClassifier().save(tmp_path / "fusion.joblib")
