"""
Federated learning tie-in.

Orchestrates weight exchange and aggregation (FedAvg) over the
:mod:`models` package. Federation only moves weights between the server
and clients; all training and inference stays inside the models.
"""

from .client import FederatedClient
from .metrics import (
    FederatedMetrics,
    convergence_round,
    parameter_set_bytes,
    round_accuracy_deltas,
)
from .parameters import average_weights
from .server import FedAvgServer, RoundResult, make_global_evaluator

__all__ = [
    "FedAvgServer",
    "FederatedClient",
    "FederatedMetrics",
    "RoundResult",
    "average_weights",
    "convergence_round",
    "make_global_evaluator",
    "parameter_set_bytes",
    "round_accuracy_deltas",
]
