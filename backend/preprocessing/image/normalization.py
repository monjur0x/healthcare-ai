"""
Medical image normalization.

Converts ``uint8`` pixel values into model-friendly float ranges.
Supported modes: ``minmax`` ([0, 1]), ``zero_mean`` ([-1, 1]), and
``standard`` (per-channel z-score).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..exceptions import ImageNormalizationError
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class NormalizationReport:
    """
    Metadata describing the normalization transform.
    """

    mode: str
    min_value: float | None = None
    max_value: float | None = None
    mean: dict[str, float] | None = None
    std: dict[str, float] | None = None
    output_dtype: str = "float32"


class ImageNormalizer:
    """
    Normalize image pixel values into the configured numeric range.

    Parameters
    ----------
    mode : str | None
        Normalization mode: "minmax", "zero_mean", or "standard".
        Defaults to ``settings.IMAGE_NORMALIZE_MODE``.
    mean : Sequence[float] | None
        Per-channel mean used by "standard" mode. Defaults to
        ``settings.IMAGE_MEAN``.
    std : Sequence[float] | None
        Per-channel std used by "standard" mode. Defaults to
        ``settings.IMAGE_STD``.
    """

    def __init__(
        self,
        mode: str | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        self._mode = (settings.IMAGE_NORMALIZE_MODE if mode is None else mode).lower()
        if self._mode not in {"minmax", "zero_mean", "standard"}:
            raise ImageNormalizationError(
                f"Unsupported normalization mode '{self._mode}'. "
                "Use 'minmax', 'zero_mean', or 'standard'."
            )

        self._mean = tuple(mean) if mean is not None else tuple(settings.IMAGE_MEAN)
        self._std = tuple(std) if std is not None else tuple(settings.IMAGE_STD)

    def fit(self, array: np.ndarray) -> ImageNormalizer:
        """
        Fit per-channel statistics for "standard" mode on an array or batch.

        Parameters
        ----------
        array : np.ndarray
            Single image or stacked batch with shape (H, W, C) or
            (N, H, W, C).

        Returns
        -------
        ImageNormalizer
            Self, fitted.

        Raises
        ------
        ImageNormalizationError
            If the array has no spatial data.
        """

        data = np.asarray(array, dtype=np.float32)
        channels = self._channel_count(data)

        if data.size == 0:
            raise ImageNormalizationError("Cannot fit on an empty image array.")

        grouped = data.reshape(-1, channels)
        self._fitted_mean = tuple(float(v) for v in grouped.mean(axis=0))
        self._fitted_std = tuple(float(v) or 1.0 for v in grouped.std(axis=0))

        self._fitted = True
        logger.info("Fitted standard normalization on %d channel(s)", channels)
        return self

    def transform(self, array: np.ndarray) -> tuple[np.ndarray, NormalizationReport]:
        """
        Normalize an image or a stacked batch.

        Parameters
        ----------
        array : np.ndarray
            ``uint8`` (0-255) image with shape (H, W), (H, W, C), or
            (N, H, W, C).

        Returns
        -------
        tuple[np.ndarray, NormalizationReport]
            Normalized ``float32`` array and a normalization report.

        Raises
        ------
        ImageNormalizationError
            If the array is empty.
        """

        data = np.asarray(array)
        if data.size == 0:
            raise ImageNormalizationError("Cannot normalize an empty image array.")

        if self._mode == "minmax":
            return self._minmax(data)
        if self._mode == "zero_mean":
            return self._zero_mean(data)
        return self._standard(data)

    def _minmax(self, array: np.ndarray) -> tuple[np.ndarray, NormalizationReport]:
        """Scale pixel values to the [0, 1] range."""
        data = array.astype(np.float32)
        low = float(data.min())
        high = float(data.max())

        scaled = (data - low) / (high - low) if high - low > 0 else np.zeros_like(data)

        report = NormalizationReport(
            mode=self._mode,
            min_value=low,
            max_value=high,
            output_dtype=str(scaled.dtype),
        )
        logger.info("Normalized images with 'minmax' to [0, 1]")
        return scaled, report

    def _zero_mean(self, array: np.ndarray) -> tuple[np.ndarray, NormalizationReport]:
        """Map 0-255 input to the [-1, 1] range."""
        data = array.astype(np.float32)
        scaled = (data / 255.0) * 2.0 - 1.0

        report = NormalizationReport(
            mode=self._mode,
            min_value=float(array.min()),
            max_value=float(array.max()),
            output_dtype=str(scaled.dtype),
        )
        logger.info("Normalized images with 'zero_mean' to [-1, 1]")
        return scaled, report

    def _standard(self, array: np.ndarray) -> tuple[np.ndarray, NormalizationReport]:
        """Apply per-channel z-score normalization."""
        if getattr(self, "_fitted", False):
            mean = self._fitted_mean
            std = self._fitted_std
        else:
            mean = tuple(float(m) for m in self._mean)
            std = tuple(float(s) for s in self._std)

        data = array.astype(np.float32)
        channels = self._channel_count(data)

        if len(mean) != channels or len(std) != channels:
            raise ImageNormalizationError(
                f"Expected {channels} mean/std values for 'standard' "
                f"normalization, got {len(mean)}/{len(std)}."
            )

        shape = (1,) * (data.ndim - 1) + (channels,)
        mean_arr = np.asarray(mean[:channels], dtype=np.float32).reshape(shape)
        std_arr = np.asarray(std[:channels], dtype=np.float32).reshape(shape)

        scaled = (data - mean_arr) / std_arr

        report = NormalizationReport(
            mode=self._mode,
            mean={str(i): float(mean[i]) for i in range(channels)},
            std={str(i): float(std[i]) for i in range(channels)},
            output_dtype=str(scaled.dtype),
        )
        logger.info("Normalized images with 'standard' (z-score)")
        return scaled, report

    @staticmethod
    def _channel_count(array: np.ndarray) -> int:
        """Return the channel count of a 2D, 3D, or batched 4D array."""
        if array.ndim == 2:
            return 1
        if array.ndim == 3:
            return array.shape[2]
        return array.shape[-1]
