"""
Tests for the Flower federated client.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.datasets import make_classification

from federated import FederatedClient, average_weights
from models import TabularClassifier


@pytest.fixture
def split_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X[:120], y[:120], X[120:], y[120:]


def _tabular_client(
    split_data,
) -> FederatedClient:
    X_train, y_train, X_val, y_val = split_data
    return FederatedClient(
        lambda: TabularClassifier(model_name="mlp"),
        X_train,
        y_train,
        X_val,
        y_val,
    )


def test_client_get_parameters(split_data) -> None:
    client = _tabular_client(split_data)
    parameters = client.get_parameters({})
    assert len(parameters) >= 2
    assert all(isinstance(parameter, np.ndarray) for parameter in parameters)


def test_client_fit_returns_updated_weights(split_data) -> None:
    client = _tabular_client(split_data)
    initial = client.get_parameters({})
    weights, n_samples, metrics = client.fit(initial, {})
    assert n_samples == 120
    assert metrics == {}
    assert len(weights) == len(initial)


def test_client_evaluate(split_data) -> None:
    client = _tabular_client(split_data)
    parameters = client.get_parameters({})
    loss, n_samples, metrics = client.evaluate(parameters, {})
    assert n_samples == 80
    assert metrics["accuracy"] > 0.7
    assert loss >= 0.0


def test_fedavg_round_improves(split_data) -> None:
    X_train, y_train, X_val, y_val = split_data
    first = FederatedClient(
        lambda: TabularClassifier(model_name="mlp"),
        X_train[:60],
        y_train[:60],
        X_val,
        y_val,
    )
    second = FederatedClient(
        lambda: TabularClassifier(model_name="mlp"),
        X_train[60:],
        y_train[60:],
        X_val,
        y_val,
    )

    global_weights = average_weights(
        [first.get_parameters({}), second.get_parameters({})]
    )
    updated_first = first.fit(global_weights, {})[0]
    updated_second = second.fit(global_weights, {})[0]

    _, _, first_metrics = first.evaluate(updated_first, {})
    _, _, second_metrics = second.evaluate(updated_second, {})
    assert first_metrics["accuracy"] > 0.7
    assert second_metrics["accuracy"] > 0.7


