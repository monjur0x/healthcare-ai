"""
Tests for the tabular prediction model and shared model interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification

from models import TabularClassifier
from models.exceptions import (
    InvalidModelInputError,
    ModelLoadError,
    ModelNotFittedError,
    UnsupportedModelError,
)


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """A small two-class synthetic dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


def test_tabular_predict(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="logistic").fit(X, y)
    assert model.is_fitted
    preds = model.predict(X)
    assert preds.shape == (200,)
    assert set(np.unique(preds)) <= {0, 1}


def test_tabular_predict_proba(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_tabular_requires_fit(classification_data) -> None:
    X, _ = classification_data
    with pytest.raises(ModelNotFittedError):
        TabularClassifier().predict(X)


def test_tabular_save_load_roundtrip(classification_data, tmp_path) -> None:
    X, y = classification_data
    model = TabularClassifier().fit(X, y)
    path = tmp_path / "tabular.joblib"
    model.save(path)
    loaded = TabularClassifier.load(path)
    assert loaded.is_fitted
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))
    assert np.array_equal(loaded.classes_, model.classes_)


def test_tabular_load_missing_file(tmp_path) -> None:
    with pytest.raises(ModelLoadError):
        TabularClassifier.load(tmp_path / "missing.joblib")


def test_tabular_fits_dataframe_feature_names(classification_data, tmp_path) -> None:
    X, y = classification_data
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = TabularClassifier().fit(frame, y)
    assert model.feature_names == [f"f{i}" for i in range(X.shape[1])]


def test_tabular_rejects_bad_inputs() -> None:
    model = TabularClassifier()
    with pytest.raises(InvalidModelInputError):
        model.fit(np.zeros((10, 1)), np.zeros((5,)))
    with pytest.raises(InvalidModelInputError):
        model.fit(np.zeros((5,)), np.zeros((5,)))


def test_tabular_unsupported_model() -> None:
    with pytest.raises(UnsupportedModelError):
        TabularClassifier(model_name="bogus")


def test_tabular_seed_reproducible(classification_data) -> None:
    X, y = classification_data
    first = TabularClassifier(model_name="mlp").fit(X, y)
    second = TabularClassifier(model_name="mlp").fit(X, y)
    assert np.array_equal(first.predict(X), second.predict(X))


def test_mlp_model_name(classification_data) -> None:
    X, y = classification_data
    model = TabularClassifier(model_name="mlp").fit(X, y)
    assert model.model_name == "mlp"
    assert model.predict(X).shape == (200,)
