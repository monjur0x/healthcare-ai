"""
Convenience functions for medical image preprocessing.

These functions wrap :class:`ImagePipeline` for callers that need a
ready-to-use NumPy array without managing pipeline objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..logger import get_logger
from .pipeline import ImagePipeline, ImageResult

logger = get_logger(__name__)

_ImageSource = str | Path | bytes | np.ndarray


def preprocess_image(
    source: _ImageSource,
    size: tuple[int, int] | None = None,
    channels: int | None = None,
    normalize: bool | None = None,
    normalize_mode: str | None = None,
    augment: bool | None = None,
) -> np.ndarray:
    """
    Preprocess a single image and return the pixel array.

    Parameters
    ----------
    source : str | Path | bytes | np.ndarray
        Image file path, raw image bytes, or array.
    size : tuple[int, int] | None
        Target ``(width, height)`` for resizing.
    channels : int | None
        Target channel count (1 or 3).
    normalize : bool | None
        Whether to normalize pixel values.
    normalize_mode : str | None
        Normalization mode ("minmax", "zero_mean", "standard").
    augment : bool | None
        Whether augmentation is active.

    Returns
    -------
    np.ndarray
        Preprocessed image array.
    """

    pipeline = ImagePipeline(
        size=size,
        channels=channels,
        normalize=normalize,
        normalize_mode=normalize_mode,
        augment=augment,
    )
    result = pipeline.run(source)
    logger.info("preprocess_image returned array with shape %s", result.shape)
    return result.image


def preprocess_batch(
    sources: Sequence[_ImageSource],
    size: tuple[int, int] | None = None,
    channels: int | None = None,
    normalize: bool | None = None,
    normalize_mode: str | None = None,
    augment: bool | None = None,
) -> np.ndarray:
    """
    Preprocess a batch of images into a stacked array.

    Parameters
    ----------
    sources : Sequence[str | Path | bytes | np.ndarray]
        Image sources to preprocess.
    size : tuple[int, int] | None
        Target ``(width, height)`` for resizing.
    channels : int | None
        Target channel count (1 or 3).
    normalize : bool | None
        Whether to normalize pixel values.
    normalize_mode : str | None
        Normalization mode ("minmax", "zero_mean", "standard").
    augment : bool | None
        Whether augmentation is active.

    Returns
    -------
    np.ndarray
        Stacked ``(N, H, W, C)`` array.
    """

    pipeline = ImagePipeline(
        size=size,
        channels=channels,
        normalize=normalize,
        normalize_mode=normalize_mode,
        augment=augment,
    )
    result = pipeline.transform_batch(sources)
    logger.info("preprocess_batch returned array with shape %s", result.shape)
    return result.image


def preprocess_directory(
    directory: str | Path,
    pattern: str = "*.png",
    size: tuple[int, int] | None = None,
    channels: int | None = None,
    normalize: bool | None = None,
    normalize_mode: str | None = None,
    augment: bool | None = None,
) -> tuple[np.ndarray, list[Path]]:
    """
    Preprocess every matching image in a directory.

    Parameters
    ----------
    directory : str | Path
        Directory containing the images.
    pattern : str
        Glob pattern for image files, e.g. ``"*.png"``.
    size : tuple[int, int] | None
        Target ``(width, height)`` for resizing.
    channels : int | None
        Target channel count (1 or 3).
    normalize : bool | None
        Whether to normalize pixel values.
    normalize_mode : str | None
        Normalization mode ("minmax", "zero_mean", "standard").
    augment : bool | None
        Whether augmentation is active.

    Returns
    -------
    tuple[np.ndarray, list[Path]]
        Stacked array and the list of paths in processing order.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no matching files.
    """

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: '{directory}'.")

    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{directory}'."
        )

    batch = preprocess_batch(
        paths,
        size=size,
        channels=channels,
        normalize=normalize,
        normalize_mode=normalize_mode,
        augment=augment,
    )
    logger.info(
        "preprocess_directory processed %d images from %s", len(paths), directory
    )
    return batch, paths


__all__ = [
    "ImageResult",
    "preprocess_batch",
    "preprocess_directory",
    "preprocess_image",
]
