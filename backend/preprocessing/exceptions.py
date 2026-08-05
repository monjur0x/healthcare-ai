"""
Custom exceptions used by preprocessing modules.
"""


class PreprocessingError(Exception):
    """
    Base preprocessing exception.
    """

    pass


class InvalidCSVError(PreprocessingError):
    """
    Raised when CSV validation fails.
    """

    pass


class InvalidImageError(PreprocessingError):
    """
    Raised when image validation fails.
    """

    pass


class MissingColumnError(PreprocessingError):
    """
    Raised when required CSV columns are missing.
    """

    pass


class UnsupportedFileTypeError(PreprocessingError):
    """
    Raised when uploaded file type is unsupported.
    """

    pass


class EmptyDatasetError(PreprocessingError):
    """
    Raised when dataset is empty.
    """

    pass


class CorruptedImageError(PreprocessingError):
    """
    Raised when an image cannot be decoded.
    """

    pass


class FeatureEngineeringError(PreprocessingError):
    """
    Raised when engineered features cannot be created.
    """

    pass


class ScalingError(PreprocessingError):
    """
    Raised when feature scaling fails.
    """

    pass


class EncodingError(PreprocessingError):
    """
    Raised when categorical encoding fails.
    """

    pass