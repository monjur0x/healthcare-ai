"""
Tests for the synchronous FedAvg server driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.datasets import make_classification

from federated import FedAvgServer, FederatedClient, make_global_evaluator
from models import TabularClassifier, TorchMLPClassifier


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


def test_server_metrics_report(clients) -> None:
    server = FedAvgServer(clients=clients, num_rounds=2).run()

    metrics = server.metrics
    assert metrics.n_rounds == 2
    assert metrics.n_clients == 3
    assert metrics.total_bytes_exchanged > 0
    assert metrics.bytes_exchanged_per_round == metrics.total_bytes_exchanged // 2
    assert len(metrics.round_times_s) == 2
    assert all(duration > 0.0 for duration in metrics.round_times_s)
    assert metrics.total_time_s > 0.0
    assert len(metrics.accuracy_deltas) == 1

    assert all(result.round_duration_s is not None for result in server.history)
    assert all(result.bytes_exchanged is not None for result in server.history)


def test_server_metrics_require_run(clients) -> None:
    server = FedAvgServer(clients=clients, num_rounds=2)
    with pytest.raises(RuntimeError):
        _ = server.metrics


def test_server_differential_privacy_reports_epsilon() -> None:
    X, y = make_classification(
        n_samples=150,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=7,
    )
    shards = [X[:50], X[50:100], X[100:]]
    labels = [y[:50], y[50:100], y[100:]]

    def make_model() -> TorchMLPClassifier:
        return TorchMLPClassifier(n_features=8, n_classes=2, epochs=3, seed=7)

    from federated.privacy import PrivacyConfig

    clients = [
        FederatedClient(
            make_model,
            shard,
            label,
            shard,
            label,
            privacy=PrivacyConfig(
                enabled=True,
                noise_multiplier=1.1,
                max_grad_norm=1.0,
                local_epochs=1,
            ),
        )
        for shard, label in zip(shards, labels, strict=True)
    ]
    server = FedAvgServer(clients=clients, num_rounds=2).run()
    metrics = server.metrics
    assert metrics.differential_privacy is True
    assert metrics.epsilon is not None and metrics.epsilon > 0.0
    assert server.max_epsilon == metrics.epsilon
    assert all(result.accuracy >= 0.4 for result in server.history)


def test_server_secure_aggregation_matches_plain_average(clients) -> None:
    plain = FedAvgServer(clients=clients, num_rounds=1).run()
    secure = FedAvgServer(clients=clients, num_rounds=1, secure_aggregation=True).run()
    assert secure.metrics.secure_aggregation is True
    assert plain.metrics.secure_aggregation is False
    np.testing.assert_allclose(
        secure.global_parameters[0],
        plain.global_parameters[0],
        atol=1e-4,
    )
