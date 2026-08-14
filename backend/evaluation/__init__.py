"""
Healthcare AI - Evaluation Package

Metrics and evaluation helpers used to score prediction models. Consumes
model outputs only; no training or preprocessing logic lives here.
"""

from .metrics import (
    ClassificationMetrics,
    classification_metrics,
    evaluate_classifier,
)

__all__ = ["ClassificationMetrics", "classification_metrics", "evaluate_classifier"]
