"""
Medical image loader.

Decodes PNG/JPG/DICOM sources into consistent NumPy arrays ready for
downstream preprocessing. Decoding is performed with Pillow; DICOM
support is optional and requires ``pydicom``.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np

from PIL import Image

from ..config import settings
from ..exceptions import (
    CorruptedImageError,
    InvalidImageError,
    UnsupportedFileTypeError,
)
from ..logger import get_logger

logger = get_logger(__name__)


class ImageLoader:
    """
    Load images from paths, bytes, or arrays.

    Loading normalizes every source to an RGB or grayscale ``uint8``
    array and optionally resizes it to the configured image size so that
    downstream stages always receive consistent shapes.

    Parameters
    ----------
    resize : bool | None
        Whether to resize images to ``settings.IMAGE_WIDTH`` /
        ``settings.IMAGE_HEIGHT``. Defaults to ``settings.IMAGE_RESIZE``.
    size : tuple[int, int] | None
        Target ``(width, height)``. Defaults to the configured size.
    channels : int | None
        Target channel count: 1 (grayscale) or 3 (RGB). Defaults to
        ``settings.IMAGE_CHANNELS``.
    """

    def __init__(
        self,
        resize: bool | None = None,
        size: tuple[int, int] | None = None,
        channels: int | None = None,
    ) -> None:
        self._resize = settings.IMAGE_RESIZE if resize is None else resize
        self._size = size or (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT)
        self._channels = channels or settings.IMAGE_CHANNELS

    def load(self, source: str | Path | bytes | np.ndarray) -> np.ndarray:
        """
        Load an image from a path, raw bytes, or an array.

        Parameters
        ----------
        source : str | Path | bytes | np.ndarray
            Image file path, raw image bytes, or a NumPy array.

        Returns
        -------
        np.ndarray
            ``uint8`` image array with shape (H, W) or (H, W, C).

        Raises
        ------
        InvalidImageError
            If the source type is unsupported or DICOM requires pydicom.
        CorruptedImageError
            If the bytes cannot be decoded.
        """

        if isinstance(source, np.ndarray):
            return self.load_array(source)

        if isinstance(source, bytes):
            return self.load_bytes(source)

        path = Path(source)
        if path.suffix.lower() == ".dcm":
            return self.load_dicom(path)
        return self.load_file(path)

    def load_file(self, path: str | Path) -> np.ndarray:
        """
        Load an image file from disk.

        Parameters
        ----------
        path : str | Path
            Path to the image file.

        Returns
        -------
        np.ndarray
            ``uint8`` image array.

        Raises
        ------
        InvalidImageError
            If the file does not exist.
        UnsupportedFileTypeError
            If the file extension is not supported.
        CorruptedImageError
            If the file cannot be decoded.
        """

        path = Path(path)

        if not path.exists():
            logger.error("Image path does not exist: %s", path)
            raise InvalidImageError(f"Image path does not exist: '{path}'.")

        if path.suffix.lower() not in settings.SUPPORTED_IMAGE_TYPES:
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

        return self._prepare(image)

    def load_bytes(self, data: bytes) -> np.ndarray:
        """
        Load an image from raw bytes (e.g. an upload payload).

        Parameters
        ----------
        data : bytes
            Raw image bytes.

        Returns
        -------
        np.ndarray
            ``uint8`` image array.

        Raises
        ------
        CorruptedImageError
            If the bytes cannot be decoded as an image.
        """

        try:
            with Image.open(BytesIO(data)) as image:
                image.load()
        except Exception as exc:
            logger.error("Failed to decode image bytes: %s", exc)
            raise CorruptedImageError(f"Could not decode image bytes: {exc}") from exc

        return self._prepare(image)

    def load_array(self, array: np.ndarray) -> np.ndarray:
        """
        Load from an in-memory array, converting to ``uint8``.

        Parameters
        ----------
        array : np.ndarray
            Input image array.

        Returns
        -------
        np.ndarray
            ``uint8`` image array.
        """

        data = np.asarray(array)

        if data.ndim == 3 and data.shape[2] == 4:
            data = self._drop_alpha(data)

        if data.ndim == 3 and data.shape[2] == 3 and self._channels == 1:
            data = self._rgb_to_gray(data)
        elif data.ndim == 2 and self._channels == 3:
            data = np.repeat(data[..., np.newaxis], 3, axis=2)

        return self._to_uint8(data)

    def load_dicom(self, path: str | Path) -> np.ndarray:
        """
        Load a DICOM file via ``pydicom``.

        Parameters
        ----------
        path : str | Path
            Path to the DICOM file.

        Returns
        -------
        np.ndarray
            ``uint8`` image array.

        Raises
        ------
        InvalidImageError
            If ``pydicom`` is not installed or the file cannot be read.
        """

        path = Path(path)
        try:
            import pydicom
        except ImportError as exc:
            logger.error("pydicom is required to load DICOM files")
            raise InvalidImageError(
                "pydicom is required to load DICOM files (.dcm). "
                "Install it to enable DICOM support."
            ) from exc

        try:
            dataset = pydicom.dcmread(path)
            data = np.asarray(dataset.pixel_array)
        except Exception as exc:
            logger.error("Failed to read DICOM %s: %s", path, exc)
            raise InvalidImageError(
                f"Could not read DICOM file '{path}': {exc}"
            ) from exc

        return self.load_array(data)

    def load_batch(
        self, sources: Sequence[str | Path | bytes | np.ndarray]
    ) -> np.ndarray:
        """
        Load multiple images into a single stacked batch.

        All images are resized/normalized to identical shapes, so the
        result is a dense ``uint8`` array.

        Parameters
        ----------
        sources : Sequence[str | Path | bytes | np.ndarray]
            Image sources to load.

        Returns
        -------
        np.ndarray
            Stacked ``uint8`` array with shape (N, H, W, C).

        Raises
        ------
        InvalidImageError
            If no sources are provided.
        """

        if not sources:
            logger.error("No image sources provided for batch load")
            raise InvalidImageError("No image sources provided for batch load.")

        images = [self.load(source) for source in sources]
        batch = np.stack(images, axis=0)
        logger.info("Loaded batch of %d images with shape %s", len(images), batch.shape)
        return batch

    def _prepare(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL image to the target channels and size."""
        image = image.convert("L") if self._channels == 1 else image.convert("RGB")

        if self._resize and image.size != self._size:
            image = image.resize(self._size, Image.Resampling.LANCZOS)

        return self._to_uint8(np.asarray(image))

    @staticmethod
    def _to_uint8(data: np.ndarray) -> np.ndarray:
        """Convert an array to ``uint8``, scaling float input to 0-255."""
        if np.issubdtype(data.dtype, np.floating):
            data = np.clip(data, 0.0, 1.0) if data.max() <= 1.0 else data
            data = (data * 255.0).astype(np.uint8)
            return data
        return np.clip(data, 0, 255).astype(np.uint8)

    @staticmethod
    def _drop_alpha(data: np.ndarray) -> np.ndarray:
        """Drop the alpha channel from an RGBA array."""
        return data[..., :3]

    @staticmethod
    def _rgb_to_gray(data: np.ndarray) -> np.ndarray:
        """Convert an RGB array to grayscale luminance."""
        coefficients = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
        return (data.astype(np.float64) @ coefficients).astype(np.uint8)
