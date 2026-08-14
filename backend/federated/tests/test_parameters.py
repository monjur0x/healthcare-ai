"""
Tests for federated weight aggregation and model weight exchange.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification

from federated import average_weights
from models import FusionClassifier, ImageClassifier, TabularClassifier
from models.exceptions import InvalidModelInputError, UnsupportedModelError
from preprocessing.multimodal import MultimodalFusion


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


def test_average_weights_elementwise() -> None:
    first = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    second = [np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    averaged = average_weights([first, second])
    np.testing.assert_allclose(averaged[0], [2.0, 3.0])
    np.testing.assert_allclose(averaged[1], [4.0, 5.0])


def test_average_weights_empty() -> None:
    with pytest.raises(ValueError):
        average_weights([])


def test_average_weights_count_mismatch() -> None:
    first = [np.zeros(2), np.zeros(2)]
    second = [np.zeros(2)]
    with pytest.raises(ValueError):
        average_weights([first, second])


def test_average_weights_shape_mismatch() -> None:
    first = [np.zeros((2, 2))]
    second = [np.zeros((3, 3))]
    with pytest.raises(ValueError):
        average_weights([first, second])


def test_tabular_logistic_roundtrip(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="logistic").fit(X, y)
    parameters = model.get_parameters()
    assert len(parameters) == 2

    replica = TabularClassifier(model_name="logistic").fit(X, y)
    replica.set_parameters(parameters)
    np.testing.assert_array_equal(replica.predict(X), model.predict(X))


def test_tabular_mlp_roundtrip(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="mlp").fit(X, y)
    parameters = model.get_parameters()
    assert len(parameters) >= 2

    replica = TabularClassifier(model_name="mlp").fit(X, y)
    replica.set_parameters(parameters)
    np.testing.assert_array_equal(replica.predict(X), model.predict(X))


def test_tabular_gradient_boosting_unsupported(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="gradient_boosting").fit(X, y)
    with pytest.raises(UnsupportedModelError):
        model.get_parameters()


def test_tabular_partial_fit_continues(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="mlp").fit(X, y)
    model.partial_fit(X, y)
    assert model.is_fitted
    assert model.predict(X).shape == (120,)


def test_tabular_partial_fit_logistic_unsupported(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="logistic").fit(X, y)
    with pytest.raises(UnsupportedModelError):
        model.partial_fit(X, y)


def test_tabular_set_parameters_empty(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="logistic").fit(X, y)
    with pytest.raises(InvalidModelInputError):
        model.set_parameters([])


def test_tabular_set_parameters_shape_mismatch(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="logistic").fit(X, y)
    with pytest.raises(InvalidModelInputError):
        model.set_parameters([np.zeros((7, 3)), np.zeros(1)])


def test_fusion_roundtrip() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 12)
    dataframe = pd.DataFrame(
        {"age": np.concatenate([rng.normal(40, 3, 12), rng.normal(60, 3, 12)])}
    )
    images = rng.normal(size=(24, 12, 12, 3)).astype(np.float32)
    for index, label in enumerate(labels):
        images[index, ..., label] += 3.0 * (label + 1)
    result = MultimodalFusion(image_reduction="summary").transform(dataframe, images)

    model = FusionClassifier().fit(result, labels)
    parameters = model.get_parameters()
    replica = FusionClassifier().fit(result, labels)
    replica.set_parameters(parameters)
    np.testing.assert_array_equal(replica.predict(result), model.predict(result))

    model.partial_fit(result, labels)
    assert model.is_fitted


def test_cnn_roundtrip() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 12)
    images = rng.normal(size=(24, 12, 12, 3)).astype(np.float32)
    for index, label in enumerate(labels):
        images[index, ..., label] += 2.0 * (label + 1)

    model = ImageClassifier(epochs=2, batch_size=8).fit(images, labels)
    parameters = model.get_parameters()
    assert len(parameters) > 0

    replica = ImageClassifier(epochs=2, batch_size=8).fit(images, labels)
    replica.set_parameters(parameters)
    np.testing.assert_array_equal(replica.predict(images), model.predict(images))

    with pytest.raises(InvalidModelInputError):
        replica.set_parameters(parameters[:-1])
