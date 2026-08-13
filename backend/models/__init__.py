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

multimodal
    Models consuming the multimodal ``FusionResult``.

This module never performs preprocessing; use ``preprocessing`` for
validation, cleaning, scaling, encoding, and image normalization.
"""

from .base import BaseModel
from .csv import TabularClassifier
from .exceptions import (
    InvalidModelInputError,
    ModelError,
    ModelLoadError,
    ModelNotFittedError,
    UnsupportedModelError,
)
from .image import ImageClassifier
from .multimodal import FusionClassifier

__all__ = [
    "BaseModel",
    "FusionClassifier",
    "ImageClassifier",
    "InvalidModelInputError",
    "ModelError",
    "ModelLoadError",
    "ModelNotFittedError",
    "TabularClassifier",
    "UnsupportedModelError",
]
