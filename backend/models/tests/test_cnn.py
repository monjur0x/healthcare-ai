"""
Tests for the PyTorch CNN image classification model.
"""

from __future__ import annotations

import numpy as np
import pytest

from models import ImageClassifier
from models.exceptions import (
    InvalidModelInputError,
    ModelLoadError,
    ModelNotFittedError,
)


def _make_batch(
    n_samples: int = 24,
    size: int = 12,
    channels: int = 3,
    n_classes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a separable synthetic image batch (channels-last)."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    images = rng.normal(size=(n_samples, size, size, channels)).astype(np.float32)
    for index, label in enumerate(labels):
        images[index, ..., label % channels] += 2.0 * (label + 1)
    return images, labels


@pytest.fixture
def image_data() -> tuple[np.ndarray, np.ndarray]:
    return _make_batch()


def _fast_model(**kwargs) -> ImageClassifier:
    """ImageClassifier with tiny training budget for fast tests."""
    defaults = {"epochs": 2, "batch_size": 8}
    defaults.update(kwargs)
    return ImageClassifier(**defaults)


def _accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions == labels))


def test_image_predict(image_data) -> None:
    X, y = image_data
    model = _fast_model().fit(X, y)
    assert model.is_fitted
    preds = model.predict(X)
    assert preds.shape == (24,)
    assert set(np.unique(preds)) <= {0, 1}
    assert _accuracy(preds, y) > 0.7


def test_image_predict_proba(image_data) -> None:
    X, y = image_data
    model = _fast_model().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (24, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_image_channels_first_accepted(image_data) -> None:
    X, y = image_data
    channels_first = X.transpose(0, 3, 1, 2)
    model = _fast_model().fit(channels_first, y)
    assert model.predict(channels_first).shape == (24,)


def test_image_multiclass() -> None:
    X, y = _make_batch(n_samples=30, n_classes=3)
    model = _fast_model().fit(X, y)
    assert len(model.classes_) == 3
    assert model.predict(X).shape == (30,)
    assert _accuracy(model.predict(X), y) > 0.6


def test_image_requires_fit(image_data) -> None:
    X, _ = image_data
    with pytest.raises(ModelNotFittedError):
        _fast_model().predict(X)


def test_image_invalid_dimensions(image_data) -> None:
    X, _ = image_data
    with pytest.raises(InvalidModelInputError):
        _fast_model().fit(X[:, :, :, 0:2, np.newaxis].reshape(24, -1), np.zeros(24))


def test_image_wrong_channels(image_data) -> None:
    X, y = image_data
    with pytest.raises(InvalidModelInputError):
        _fast_model(in_channels=4).fit(X, y)


def test_image_save_load_roundtrip(image_data, tmp_path) -> None:
    X, y = image_data
    model = _fast_model().fit(X, y)
    original = model.predict(X)
    target = tmp_path / "model" / "cnn.pt"

    model.save(target)
    loaded = ImageClassifier.load(target)

    assert loaded.is_fitted
    np.testing.assert_array_equal(loaded.classes_, model.classes_)
    np.testing.assert_array_equal(loaded.predict(X), original)


def test_image_load_missing_file(tmp_path) -> None:
    with pytest.raises(ModelLoadError):
        ImageClassifier.load(tmp_path / "missing.pt")


def test_image_save_requires_fit(image_data, tmp_path) -> None:
    _X, _y = image_data
    with pytest.raises(ModelNotFittedError):
        _fast_model().save(tmp_path / "model.pt")


def test_image_deterministic_same_seed(image_data) -> None:
    X, y = image_data
    first = _fast_model().fit(X, y).predict(X)
    second = _fast_model().fit(X, y).predict(X)
    np.testing.assert_array_equal(first, second)


def test_image_invalid_constructor_args() -> None:
    with pytest.raises(ValueError):
        ImageClassifier(epochs=0)
    with pytest.raises(ValueError):
        ImageClassifier(learning_rate=-1.0)
    with pytest.raises(ValueError):
        ImageClassifier(in_channels=0)
