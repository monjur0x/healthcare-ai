"""
Healthcare AI - Prediction Models Package

Prediction models consume preprocessing outputs and perform inference
only. Module responsibilities:

base
    Abstract model interface shared by all models.

csv
    Tabular classification models for structured healthcare data.

image
    PyTorch CNN image classification models.

This module never performs preprocessing; use ``preprocessing`` for
validation, cleaning, scaling, encoding, and image normalization.
"""

from .base import BaseModel
from .csv import TabularClassifier, TorchMLPClassifier
from .exceptions import (
    InvalidModelInputError,
    ModelError,
    ModelLoadError,
    ModelNotFittedError,
    UnsupportedModelError,
)
from .image import ImageClassifier

__all__ = [
    "BaseModel",
    "ImageClassifier",
    "InvalidModelInputError",
    "ModelError",
    "ModelLoadError",
    "ModelNotFittedError",
    "TabularClassifier",
    "TorchMLPClassifier",
    "UnsupportedModelError",
]
