"""
Tabular (CSV/EHR) prediction models.

Models in this package consume the all-numeric output of the CSV
preprocessing pipeline and perform classification/regression only.
"""

from .tabular import TabularClassifier

__all__ = ["TabularClassifier"]
