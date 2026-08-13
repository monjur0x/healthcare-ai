"""
Deterministic medical image augmentation.

Augmentations are driven by a seeded RNG so that runs are reproducible,
which is a research requirement of this repository. Augmentation operates
on ``uint8`` pixel values and runs before normalization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from PIL import Image, ImageEnhance

from ..config import settings
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AugmentationReport:
    """
    Metadata describing an augmentation run.
    """

    enabled: bool
    applied: tuple[str, ...]
    seed: int


class ImageAugmenter:
    """
    Apply stochastic image augmentations with a fixed seed.

    Supported operations:
    - ``horizontal_flip``
    - ``vertical_flip``
    - ``random_rotation``
    - ``brightness_contrast``

    Each enabled operation is applied independently with probability
    ``apply_probability``.

    Parameters
    ----------
    enabled : bool | None
        Whether augmentation is active. Defaults to
        ``settings.IMAGE_AUGMENTATION_ENABLED``.
    operations : Sequence[str]
        Operations to consider. Defaults to all supported operations.
    apply_probability : float | None
        Probability per operation. Defaults to
        ``settings.IMAGE_AUGMENT_PROBABILITY``.
    rotation_range : int | None
        Max absolute rotation angle in degrees. Defaults to
        ``settings.IMAGE_ROTATION_RANGE``.
    seed : int | None
        Random seed for reproducibility. Defaults to
        ``settings.RANDOM_SEED``.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        operations: Sequence[str] = (
            "horizontal_flip",
            "vertical_flip",
            "random_rotation",
            "brightness_contrast",
        ),
        apply_probability: float | None = None,
        rotation_range: int | None = None,
        seed: int | None = None,
    ) -> None:
        self._enabled = (
            settings.IMAGE_AUGMENTATION_ENABLED if enabled is None else enabled
        )
        self._operations = tuple(operations)
        self._apply_probability = (
            settings.IMAGE_AUGMENT_PROBABILITY
            if apply_probability is None
            else apply_probability
        )
        self._rotation_range = (
            settings.IMAGE_ROTATION_RANGE if rotation_range is None else rotation_range
        )
        self._seed = settings.RANDOM_SEED if seed is None else seed
        self._rng = np.random.default_rng(self._seed)

    def transform(
        self, array: np.ndarray, enabled: bool | None = None
    ) -> tuple[np.ndarray, AugmentationReport]:
        """
        Apply configured augmentations to an image or batch.

        Parameters
        ----------
        array : np.ndarray
            ``uint8`` image with shape (H, W), (H, W, C), or (N, H, W, C).
        enabled : bool | None
            Override the ``enabled`` flag for this call.

        Returns
        -------
        tuple[np.ndarray, AugmentationReport]
            Augmented array and an augmentation report.
        """

        active = self._enabled if enabled is None else enabled

        if not active:
            return array, AugmentationReport(enabled=False, applied=(), seed=self._seed)

        data = np.asarray(array)
        applied: list[str] = []
        handlers = {
            "horizontal_flip": ImageAugmenter._flip_horizontal,
            "vertical_flip": ImageAugmenter._flip_vertical,
            "random_rotation": self._rotate,
            "brightness_contrast": self._enhance,
        }

        for name in self._operations:
            handler = handlers.get(name)
            if handler is None:
                logger.warning("Unknown augmentation operation skipped: %s", name)
                continue
            if self._rng.random() < self._apply_probability:
                data = handler(data)
                applied.append(name)

        report = AugmentationReport(
            enabled=True, applied=tuple(applied), seed=self._seed
        )
        logger.info("Applied augmentations: %s", applied)
        return data, report

    @staticmethod
    def _flip_horizontal(array: np.ndarray) -> np.ndarray:
        """Mirror the image along the horizontal axis."""
        return np.fliplr(array)

    @staticmethod
    def _flip_vertical(array: np.ndarray) -> np.ndarray:
        """Mirror the image along the vertical axis."""
        return np.flipud(array)

    def _rotate(self, array: np.ndarray) -> np.ndarray:
        """Rotate the image by a random angle within the configured range."""
        angle = float(self._rng.uniform(-self._rotation_range, self._rotation_range))
        image = Image.fromarray(array)
        rotated = image.rotate(angle, resample=Image.Resampling.BILINEAR)
        return np.asarray(rotated)

    def _enhance(self, array: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast adjustments."""
        image = Image.fromarray(array)
        brightness = float(self._rng.uniform(0.8, 1.2))
        contrast = float(self._rng.uniform(0.8, 1.2))
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        return np.asarray(image)
