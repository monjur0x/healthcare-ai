"""
Federated training metrics.

Pure helpers that quantify the cost of a federated run: communication
cost (bytes exchanged per round), convergence (round-to-round accuracy
change), and wall-clock training time. Kept dependency-free so the
synchronous :class:`FedAvgServer` driver can surface them without
coupling to Flower or a specific training backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np


def parameter_set_bytes(parameters: Sequence[np.ndarray]) -> int:
    """
    Total bytes for one full weight exchange.

    Parameters
    ----------
    parameters : Sequence[np.ndarray]
        Ordered weight arrays.

    Returns
    -------
    int
        Sum of the in-memory byte sizes of every array.
    """

    return int(sum(int(np.asarray(parameter).nbytes) for parameter in parameters))


def round_accuracy_deltas(accuracies: Sequence[float]) -> tuple[float, ...]:
    """
    Absolute accuracy change between consecutive rounds.

    Parameters
    ----------
    accuracies : Sequence[float]
        Global accuracy per round, in round order.

    Returns
    -------
    tuple[float, ...]
        ``len(accuracies) - 1`` deltas.
    """

    return tuple(
        round(abs(float(after) - float(before)), 6)
        for before, after in pairwise(accuracies)
    )


def convergence_round(
    accuracies: Sequence[float], threshold: float = 1e-3
) -> int | None:
    """
    First round index whose accuracy changed less than ``threshold``.

    Parameters
    ----------
    accuracies : Sequence[float]
        Global accuracy per round, in round order.
    threshold : float
        Maximum allowed absolute accuracy change to count as converged.

    Returns
    -------
    int | None
        1-based round index, or ``None`` if never converged.
    """

    for offset, delta in enumerate(round_accuracy_deltas(accuracies)):
        if delta < threshold:
            return offset + 2
    return None


@dataclass(frozen=True)
class FederatedMetrics:
    """
    Aggregate cost, convergence, timing, and privacy statistics.

    Parameters
    ----------
    n_rounds : int
        Number of executed rounds.
    n_clients : int
        Number of participating clients.
    total_bytes_exchanged : int
        Total bytes moved over the whole run.
    bytes_exchanged_per_round : int
        Estimated bytes moved per round (client upload + broadcast).
    round_times_s : tuple[float, ...]
        Wall-clock duration of each round.
    total_time_s : float
        Sum of the per-round durations.
    accuracy_deltas : tuple[float, ...]
        Round-to-round absolute accuracy changes.
    convergence_round : int | None
        First converged round index (see :func:`convergence_round`).
    secure_aggregation : bool
        Whether secure aggregation masked the client updates.
    differential_privacy : bool
        Whether local training used DP-SGD.
    epsilon : float | None
        Worst-case per-client epsilon when differential privacy is on.
    """

    n_rounds: int
    n_clients: int
    total_bytes_exchanged: int
    bytes_exchanged_per_round: int
    round_times_s: tuple[float, ...]
    total_time_s: float
    accuracy_deltas: tuple[float, ...]
    convergence_round: int | None
    secure_aggregation: bool = False
    differential_privacy: bool = False
    epsilon: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the metrics to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Metrics keyed by name.
        """

        return {
            "n_rounds": self.n_rounds,
            "n_clients": self.n_clients,
            "total_bytes_exchanged": self.total_bytes_exchanged,
            "bytes_exchanged_per_round": self.bytes_exchanged_per_round,
            "round_times_s": list(self.round_times_s),
            "total_time_s": self.total_time_s,
            "accuracy_deltas": list(self.accuracy_deltas),
            "convergence_round": self.convergence_round,
            "secure_aggregation": self.secure_aggregation,
            "differential_privacy": self.differential_privacy,
            "epsilon": self.epsilon,
        }


__all__ = [
    "FederatedMetrics",
    "convergence_round",
    "parameter_set_bytes",
    "round_accuracy_deltas",
]
