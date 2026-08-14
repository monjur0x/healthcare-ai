"""
Federated learning tie-in.

Orchestrates weight exchange and aggregation (FedAvg) over the
:mod:`models` package. Federation only moves weights between the server
and clients; all training and inference stays inside the models.
"""

from .client import FederatedClient
from .parameters import average_weights

__all__ = ["FederatedClient", "average_weights"]
