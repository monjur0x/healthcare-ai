"""
Multimodal prediction models.

Multimodal models consume ``FusionResult`` objects produced by
``preprocessing.multimodal`` and perform classification on the fused
feature matrix.
"""

from .fusion_model import FusionClassifier

__all__ = ["FusionClassifier"]
