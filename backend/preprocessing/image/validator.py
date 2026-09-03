"""
Medical image validation.

Validates file type, decodability, dimensions, and channel count before
any downstream image preprocessing step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PIL import Image

from ..config import settings
from ..exceptions import (
    CorruptedImageError,
    EmptyDatasetError,
    InvalidImageError,
    UnsupportedFileTypeError,
)
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ImageValidationResult:
    """
    Immutable result of an image validation run.
    """

    is_valid: bool
    width: int = 0
    height: int = 0
    channels: int = 0
    mode: str = ""
    format: str | None = None
    path: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


class ImageValidator:
    """
    Validates image files and in-memory image arrays.

    The validator is the first stage of the image pipeline. It enforces
    supported file types, decodability, and sensible dimension/channel
    values, producing warnings instead of hard errors for minor
    mismatches against the configured image size.

    Methods
    -------
    is_supported(path)
        Check whether a path has a supported image extension.
    validate_file(path)
        Validate an image file on disk.
    validate_array(array)
        Validate an in-memory image array.
    """

    @staticmethod
    def is_supported(path: str | Path) -> bool:
        """
        Check whether a file path has a supported image extension.

        Parameters
        ----------
        path : str | Path
            Path to inspect.

        Returns
        -------
        bool
            True if the extension is in ``settings.SUPPORTED_IMAGE_TYPES``.
        """

        return Path(path).suffix.lower() in settings.SUPPORTED_IMAGE_TYPES

    def validate_file(self, path: str | Path) -> ImageValidationResult:
        """
        Validate an image file on disk.

        Parameters
        ----------
        path : str | Path
            Path to the image file.

        Returns
        -------
        ImageValidationResult
            Validation metadata for the image.

        Raises
        ------
        InvalidImageError
            If the file does not exist.
        UnsupportedFileTypeError
            If the extension is not a supported image type.
        CorruptedImageError
            If the file cannot be decoded as an image.
        """

        path = Path(path)

        if not path.exists():
            logger.error("Image path does not exist: %s", path)
            raise InvalidImageError(f"Image path does not exist: '{path}'.")

        if not self.is_supported(path):
            logger.error("Unsupported image type: %s", path)
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported types: {settings.SUPPORTED_IMAGE_TYPES}."
            )

        try:
            with Image.open(path) as image:
                image.load()
        except Exception as exc:
            logger.error("Failed to decode image %s: %s", path, exc)
            raise CorruptedImageError(
                f"Could not decode image '{path}': {exc}"
            ) from exc

        return self._build_result(
            width=image.width,
            height=image.height,
            channels=len(image.getbands()),
            mode=image.mode,
            image_format=image.format,
            path=str(path),
        )

    def validate_array(self, array: np.ndarray) -> ImageValidationResult:
        """
        Validate an in-memory image array.

        Parameters
        ----------
        array : np.ndarray
            Image array with shape (H, W) or (H, W, C).

        Returns
        -------
        ImageValidationResult
            Validation metadata for the array.

        Raises
        ------
        InvalidImageError
            If the array shape or dtype is not image-like.
        EmptyDatasetError
            If the array contains no pixels.
        """

        data = np.asarray(array)

        if not np.issubdtype(data.dtype, np.number):
            logger.error("Image array dtype %s is not numeric", data.dtype)
            raise InvalidImageError(f"Image array dtype '{data.dtype}' is not numeric.")

        if data.ndim not in (2, 3):
            logger.error("Image array has %d dimensions", data.ndim)
            raise InvalidImageError(
                f"Image array must be 2D (grayscale) or 3D (channels), "
                f"got {data.ndim} dimensions."
            )

        if data.ndim == 3 and data.shape[2] not in (1, 3, 4):
            logger.error("Image array has %d channels", data.shape[2])
            raise InvalidImageError(
                f"Image array must have 1, 3, or 4 channels, got {data.shape[2]}."
            )

        if data.size == 0:
            logger.error("Image array is empty")
            raise EmptyDatasetError("Image array contains no pixels.")

        height, width = data.shape[:2]
        channels = data.shape[2] if data.ndim == 3 else 1
        mode = "L" if channels == 1 else ("RGB" if channels == 3 else "RGBA")

        return self._build_result(
            width=width,
            height=height,
            channels=channels,
            mode=mode,
            image_format=None,
            path=None,
        )

    @staticmethod
    def _build_result(
        width: int,
        height: int,
        channels: int,
        mode: str,
        image_format: str | None,
        path: str | None,
    ) -> ImageValidationResult:
        """Build a validation result, appending any config-based warnings."""

        warnings: list[str] = []

        if channels != settings.IMAGE_CHANNELS:
            warnings.append(
                f"Expected {settings.IMAGE_CHANNELS} channels, got {channels}."
            )
        if width != settings.IMAGE_WIDTH or height != settings.IMAGE_HEIGHT:
            warnings.append(
                f"Expected {settings.IMAGE_WIDTH}x{settings.IMAGE_HEIGHT} "
                f"pixels, got {width}x{height}."
            )

        result = ImageValidationResult(
            is_valid=True,
            width=width,
            height=height,
            channels=channels,
            mode=mode,
            format=image_format,
            path=path,
            warnings=tuple(warnings),
        )
        logger.info(
            "Validated image: %dx%d, %d channel(s), format=%s",
            width,
            height,
            channels,
            image_format,
        )
        return result
