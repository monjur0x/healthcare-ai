"""
Tabular (CSV/EHR) prediction models.

Models in this package consume the all-numeric output of the CSV
preprocessing pipeline and perform classification/regression only.
"""

from .tabular import TabularClassifier
from .torch_mlp import TorchMLPClassifier

__all__ = ["TabularClassifier", "TorchMLPClassifier"]
