"""
Image prediction models.

Image models consume the channels-last ``(N, H, W, C)`` batches produced
by ``preprocessing.image`` and perform classification only.
"""

from .cnn import ImageClassifier

__all__ = ["ImageClassifier"]
