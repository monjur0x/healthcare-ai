"""
End-to-end federated tests for the CNN image model path.

The image classifier joins federated rounds through
:class:`ImageClassifier.partial_fit`; the synchronous
:class:`FedAvgServer` driver and per-client evaluation are reused
unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from federated import FedAvgServer, FederatedClient
from models import ImageClassifier


def _make_batch(
    n_samples: int = 48,
    size: int = 12,
    channels: int = 3,
    n_classes: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a separable synthetic image batch (channels-last)."""
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(n_classes), n_samples // n_classes)
    images = rng.normal(size=(n_samples, size, size, channels)).astype(np.float32)
    for index, label in enumerate(labels):
        images[index, ..., label % channels] += 2.0 * (label + 1)
    return images, labels


def _fast_image(**kwargs) -> ImageClassifier:
    """ImageClassifier with a tiny training budget for fast tests."""
    defaults = {"epochs": 2, "batch_size": 8}
    defaults.update(kwargs)
    return ImageClassifier(**defaults)


@pytest.fixture
def cnn_clients() -> list[FederatedClient]:
    X, y = _make_batch()
    return [
        FederatedClient(lambda: _fast_image(), X[:24], y[:24], X, y),
        FederatedClient(lambda: _fast_image(), X[24:], y[24:], X, y),
    ]


def test_cnn_federated_rounds_produce_history(cnn_clients) -> None:
    server = FedAvgServer(clients=cnn_clients, num_rounds=2).run()

    assert len(server.history) == 2
    assert server.global_parameters is not None
    assert all(
        result.round_index == index
        for index, result in enumerate(server.history, start=1)
    )
    assert all(result.accuracy > 0.8 for result in server.history)
    assert all(result.log_loss is not None for result in server.history)


def test_cnn_federated_weights_exchangeable(cnn_clients) -> None:
    server = FedAvgServer(clients=cnn_clients, num_rounds=1).run()

    parameters = server.global_parameters
    assert parameters is not None
    assert len(parameters) >= 10

    X, y = _make_batch()
    model = _fast_image().fit(X, y)
    model.set_parameters(parameters)
    assert float(np.mean(model.predict(X) == y)) > 0.8
