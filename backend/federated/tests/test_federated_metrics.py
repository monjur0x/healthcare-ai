"""
Tests for federated metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from federated import (
    FederatedMetrics,
    convergence_round,
    parameter_set_bytes,
    round_accuracy_deltas,
)


def test_parameter_set_bytes_sums_nbytes() -> None:
    parameters = [np.zeros((4, 2), dtype=np.float64), np.ones(3, dtype=np.float32)]
    assert parameter_set_bytes(parameters) == 4 * 2 * 8 + 3 * 4


def test_parameter_set_bytes_empty() -> None:
    assert parameter_set_bytes([]) == 0


def test_round_accuracy_deltas() -> None:
    assert round_accuracy_deltas([0.5, 0.6, 0.62]) == (0.1, 0.02)
    assert round_accuracy_deltas([0.8]) == ()


def test_convergence_round_found() -> None:
    accuracies = [0.4, 0.7, 0.72, 0.721]
    assert convergence_round(accuracies, threshold=0.01) == 4


def test_convergence_round_none() -> None:
    accuracies = [0.4, 0.5, 0.6, 0.7]
    assert convergence_round(accuracies, threshold=0.001) is None


def test_convergence_round_single_round() -> None:
    assert convergence_round([0.9]) is None


def test_federated_metrics_to_dict() -> None:
    metrics = FederatedMetrics(
        n_rounds=3,
        n_clients=2,
        total_bytes_exchanged=1200,
        bytes_exchanged_per_round=400,
        round_times_s=(0.1, 0.2, 0.3),
        total_time_s=0.6,
        accuracy_deltas=(0.2, 0.02),
        convergence_round=3,
    )
    payload = metrics.to_dict()

    assert payload["n_rounds"] == 3
    assert payload["bytes_exchanged_per_round"] == 400
    assert payload["round_times_s"] == [0.1, 0.2, 0.3]
    assert payload["convergence_round"] == 3
    assert isinstance(payload["accuracy_deltas"], list)


def test_federated_metrics_frozen() -> None:
    metrics = FederatedMetrics(
        n_rounds=1,
        n_clients=1,
        total_bytes_exchanged=0,
        bytes_exchanged_per_round=0,
        round_times_s=(),
        total_time_s=0.0,
        accuracy_deltas=(),
        convergence_round=None,
    )
    with pytest.raises(AttributeError):
        metrics.total_time_s = 1.0
