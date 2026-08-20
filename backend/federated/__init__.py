"""
Federated learning tie-in.

Orchestrates weight exchange and aggregation (FedAvg) over the
:mod:`models` package. Federation only moves weights between the server
and clients; all training and inference stays inside the models.
"""

from .client import FederatedClient
from .distributed import DistributedFedAvg, ModelSpec, run_distributed_server
from .hospitals import (
    PRESETS,
    HospitalConfig,
    build_hospital_sites,
    load_hospital_dataset,
)
from .metrics import (
    FederatedMetrics,
    convergence_round,
    parameter_set_bytes,
    round_accuracy_deltas,
)
from .parameters import average_weights
from .registry import ModelRegistry
from .server import FedAvgServer, RoundResult, make_global_evaluator

__all__ = [
    "PRESETS",
    "DistributedFedAvg",
    "FedAvgServer",
    "FederatedClient",
    "FederatedMetrics",
    "HospitalConfig",
    "ModelRegistry",
    "ModelSpec",
    "RoundResult",
    "average_weights",
    "build_hospital_sites",
    "convergence_round",
    "load_hospital_dataset",
    "make_global_evaluator",
    "parameter_set_bytes",
    "round_accuracy_deltas",
    "run_distributed_server",
]
