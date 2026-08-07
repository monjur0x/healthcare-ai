"""
Custom exceptions used by preprocessing modules.
"""


class PreprocessingError(Exception):
    """
    Base preprocessing exception.
    """


class InvalidCSVError(PreprocessingError):
    """
    Raised when CSV validation fails.
    """


class InvalidImageError(PreprocessingError):
    """
    Raised when image validation fails.
    """


class MissingColumnError(PreprocessingError):
    """
    Raised when required CSV columns are missing.
    """


class UnsupportedFileTypeError(PreprocessingError):
    """
    Raised when uploaded file type is unsupported.
    """


class EmptyDatasetError(PreprocessingError):
    """
    Raised when dataset is empty.
    """


class CorruptedImageError(PreprocessingError):
    """
    Raised when an image cannot be decoded.
    """


class FeatureEngineeringError(PreprocessingError):
    """
    Raised when engineered features cannot be created.
    """


class ScalingError(PreprocessingError):
    """
    Raised when feature scaling fails.
    """


class EncodingError(PreprocessingError):
    """
    Raised when categorical encoding fails.
    """
