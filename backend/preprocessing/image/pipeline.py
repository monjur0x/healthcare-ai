"""
High-level medical image preprocessing pipeline.

The pipeline is the reusable entry point for Flower, FastAPI, and
CrewAI. It composes loading, validation, augmentation, and normalization
into a single transform for a single image or a stacked batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import settings
from ..logger import get_logger
from .augmentation import AugmentationReport, ImageAugmenter
from .loader import ImageLoader
from .normalization import ImageNormalizer, NormalizationReport
from .validator import ImageValidationResult, ImageValidator

logger = get_logger(__name__)

_ImageSource = str | Path | bytes | np.ndarray


@dataclass(frozen=True)
class ImageResult:
    """
    Final pipeline output for one or more images.
    """

    image: np.ndarray
    reports: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the output array."""
        return self.image.shape

    @property
    def dtype(self) -> str:
        """Dtype of the output array."""
        return str(self.image.dtype)

    def to_dict(self) -> dict:
        """
        Serialize the result to a JSON-friendly dictionary.

        Returns
        -------
        dict
            Dictionary with shape, dtype, and per-stage reports.
        """

        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "reports": self.reports,
        }


class ImagePipeline:
    """
    Run the full preprocessing sequence on image sources.

    Pipeline order: load -> validate -> resize -> augment -> normalize.

    Parameters
    ----------
    size : tuple[int, int] | None
        Target ``(width, height)`` for resizing. Defaults to
        ``settings.IMAGE_WIDTH`` / ``settings.IMAGE_HEIGHT``.
    channels : int | None
        Target channel count (1 or 3). Defaults to
        ``settings.IMAGE_CHANNELS``.
    normalize : bool | None
        Whether to normalize pixel values. Defaults to
        ``settings.NORMALIZE_IMAGES``.
    normalize_mode : str | None
        Normalization mode; see :class:`ImageNormalizer`.
    augment : bool | None
        Whether augmentation is active. Defaults to
        ``settings.IMAGE_AUGMENTATION_ENABLED``.
    """

    def __init__(
        self,
        size: tuple[int, int] | None = None,
        channels: int | None = None,
        normalize: bool | None = None,
        normalize_mode: str | None = None,
        augment: bool | None = None,
    ) -> None:
        self._loader = ImageLoader(size=size, channels=channels)
        self._validator = ImageValidator()
        self._normalizer = ImageNormalizer(mode=normalize_mode)
        self._augmenter = ImageAugmenter(enabled=augment)
        self._normalize_enabled = (
            settings.NORMALIZE_IMAGES if normalize is None else normalize
        )

    def transform(self, source: _ImageSource) -> ImageResult:
        """
        Preprocess a single image source.

        Parameters
        ----------
        source : str | Path | bytes | np.ndarray
            Image file path, raw image bytes, or array.

        Returns
        -------
        ImageResult
            Preprocessed image plus per-stage reports.
        """

        image = self._loader.load(source)
        validation = self._validator.validate_array(image)
        image, augmentation = self._augmenter.transform(image)

        if self._normalize_enabled:
            image, normalization = self._normalizer.transform(image)
        else:
            normalization = None

        reports = {
            "validation": _asdict_optional(validation),
            "augmentation": _asdict_optional(augmentation),
            "normalization": _asdict_optional(normalization),
        }
        logger.info("Image pipeline completed with shape %s", image.shape)
        return ImageResult(image=image, reports=reports)

    def transform_batch(self, sources: Sequence[_ImageSource]) -> ImageResult:
        """
        Preprocess a batch of image sources into a stacked array.

        Parameters
        ----------
        sources : Sequence[str | Path | bytes | np.ndarray]
            Image sources to preprocess.

        Returns
        -------
        ImageResult
            Stacked ``(N, H, W, C)`` array plus per-stage reports.
        """

        if not sources:
            from ..exceptions import InvalidImageError

            raise InvalidImageError("No image sources provided for batch processing.")

        images = [self._loader.load(source) for source in sources]

        validations = []
        for image in images:
            validations.append(_asdict_optional(self._validator.validate_array(image)))

        augmented = []
        for image in images:
            image, _ = self._augmenter.transform(image)
            augmented.append(image)

        batch = np.stack(augmented, axis=0)

        if self._normalize_enabled:
            batch, normalization = self._normalizer.transform(batch)
        else:
            normalization = None

        reports = {
            "validation": validations,
            "augmentation": None,
            "normalization": _asdict_optional(normalization),
        }
        logger.info("Image batch pipeline completed with shape %s", batch.shape)
        return ImageResult(image=batch, reports=reports)

    def fit(self, sources: Sequence[_ImageSource]) -> ImagePipeline:
        """
        Fit trainable stages (normalizer statistics) on a batch.

        This is only meaningful for ``standard`` normalization, which
        benefits from dataset-level statistics.

        Parameters
        ----------
        sources : Sequence[str | Path | bytes | np.ndarray]
            Image sources to fit on.

        Returns
        -------
        ImagePipeline
            Self, fitted.
        """

        batch = self._loader.load_batch(sources)
        self._normalizer.fit(batch)
        self._fitted = True
        logger.info("ImagePipeline fitted on %d images", len(sources))
        return self

    def run(self, source: _ImageSource) -> ImageResult:
        """
        Convenience alias for :meth:`transform`.

        Parameters
        ----------
        source : str | Path | bytes | np.ndarray
            Image source to preprocess.

        Returns
        -------
        ImageResult
            Preprocessed image plus reports.
        """

        return self.transform(source)


def _asdict_optional(report) -> dict | None:
    """
    Convert a dataclass report to a dict, or return None.
    """

    return asdict(report) if report is not None else None


__all__ = [
    "AugmentationReport",
    "ImageResult",
    "ImageValidationResult",
    "NormalizationReport",
]
