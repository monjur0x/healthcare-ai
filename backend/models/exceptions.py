"""
Custom exceptions used by prediction models.
"""


class ModelError(Exception):
    """
    Base model exception.
    """


class ModelNotFittedError(ModelError):
    """
    Raised when an unfitted model is used for prediction.
    """


class ModelLoadError(ModelError):
    """
    Raised when a persisted model cannot be loaded.
    """


class InvalidModelInputError(ModelError):
    """
    Raised when model input arrays are malformed.
    """


class UnsupportedModelError(ModelError):
    """
    Raised when an unknown model type is requested.
    """
