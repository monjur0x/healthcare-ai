"""
Tests for the synchronous FedAvg server driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.datasets import make_classification

from federated import FedAvgServer, FederatedClient, make_global_evaluator
from models import TabularClassifier


@pytest.fixture
def clients() -> list[FederatedClient]:
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    shards = [X[:80], X[80:160], X[160:]]
    labels = [y[:80], y[80:160], y[160:]]
    return [
        FederatedClient(
            lambda: TabularClassifier(model_name="mlp"),
            shard,
            label,
            shard,
            label,
        )
        for shard, label in zip(shards, labels, strict=True)
    ]


def test_server_run_produces_history(clients) -> None:
    server = FedAvgServer(clients=clients, num_rounds=3).run()
    assert len(server.history) == 3
    assert server.global_parameters is not None
    assert all(
        result.round_index == index
        for index, result in enumerate(server.history, start=1)
    )
    assert all(result.accuracy > 0.7 for result in server.history)
    assert all(
        result.log_loss is not None and result.log_loss >= 0.0
        for result in server.history
    )


def test_server_global_evaluator() -> None:
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=0,
    )
    train_x, train_y = X[:140], y[:140]
    test_x, test_y = X[140:], y[140:]

    clients = [
        FederatedClient(
            lambda: TabularClassifier(model_name="mlp"),
            train_x[:70],
            train_y[:70],
            train_x,
            train_y,
        ),
        FederatedClient(
            lambda: TabularClassifier(model_name="mlp"),
            train_x[70:],
            train_y[70:],
            train_x,
            train_y,
        ),
    ]
    evaluator = make_global_evaluator(
        lambda: TabularClassifier(model_name="mlp"), test_x, test_y
    )

    server = FedAvgServer(clients=clients, num_rounds=2, evaluate_fn=evaluator).run()

    final = server.history[-1]
    assert final.accuracy > 0.7
    assert final.roc_auc is not None and final.roc_auc > 0.7


def test_server_empty_clients() -> None:
    with pytest.raises(ValueError):
        FedAvgServer(clients=[])


def test_server_invalid_rounds(clients) -> None:
    with pytest.raises(ValueError):
        FedAvgServer(clients=clients, num_rounds=0)


def test_server_global_parameters_structure(clients) -> None:
    server = FedAvgServer(clients=clients, num_rounds=1).run()
    parameters = server.global_parameters
    assert parameters is not None
    assert all(isinstance(parameter, np.ndarray) for parameter in parameters)
    assert len(parameters) >= 2
