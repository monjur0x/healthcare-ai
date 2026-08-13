"""
Multimodal fusion for preprocessed tabular and image data.

Consumes the outputs of the CSV and image pipelines (an all-numeric
dataframe and a normalized image batch) and combines them into a single
feature matrix aligned per sample index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings
from ..exceptions import FusionError
from ..logger import get_logger
from .metadata import ImageInfo, SampleMetadata, native

logger = get_logger(__name__)


@dataclass(frozen=True)
class FusionReport:
    """
    Metadata describing a fusion run.
    """

    mode: str
    image_reduction: str
    n_samples: int
    feature_dim: int
    image_dim: int
    fused_dim: int
    dropped_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class FusionResult:
    """
    Fused multimodal output.
    """

    features: np.ndarray
    images: np.ndarray
    fused: np.ndarray
    metadata: tuple[SampleMetadata, ...]
    report: FusionReport

    def to_dict(self) -> dict:
        """
        Serialize the result to a JSON-friendly dictionary.

        Returns
        -------
        dict
            Dictionary with shapes, report, and sample metadata.
        """

        return {
            "features_shape": list(self.features.shape),
            "images_shape": list(self.images.shape),
            "fused_shape": list(self.fused.shape),
            "report": {
                "mode": self.report.mode,
                "image_reduction": self.report.image_reduction,
                "n_samples": self.report.n_samples,
                "feature_dim": self.report.feature_dim,
                "image_dim": self.report.image_dim,
                "fused_dim": self.report.fused_dim,
                "dropped_columns": list(self.report.dropped_columns),
            },
            "metadata": [sample.to_dict() for sample in self.metadata],
        }


class MultimodalFusion:
    """
    Align and fuse preprocessed tabular features with image batch data.

    ``transform`` expects the outputs of the CSV and image pipelines:
    a dataframe (all-numeric after encoding/scaling) and a channels-last
    image batch ``(N, H, W, C)``. Samples are aligned positionally:
    dataframe row ``i`` corresponds to image ``i``.

    Parameters
    ----------
    mode : str | None
        Fusion mode. Currently supports ``"concatenate"``. Defaults to
        ``settings.FUSION_MODE``.
    image_reduction : str | None
        Image representation: ``"summary"`` (per-channel
        mean/std/min/max, compact) or ``"flatten"`` (full pixel
        vector). Defaults to ``settings.FUSION_IMAGE_REDUCTION``.
    patient_id_column : str | None
        Dataframe column holding patient identifiers used for metadata.
        If None, both the dataframe index and image order are used.
    """

    def __init__(
        self,
        mode: str | None = None,
        image_reduction: str | None = None,
        patient_id_column: str | None = None,
    ) -> None:
        self._mode = (settings.FUSION_MODE if mode is None else mode).lower()
        if self._mode != "concatenate":
            raise FusionError(
                f"Unsupported fusion mode '{self._mode}'. Use 'concatenate'."
            )

        self._image_reduction = (
            settings.FUSION_IMAGE_REDUCTION
            if image_reduction is None
            else image_reduction.lower()
        )
        if self._image_reduction not in {"summary", "flatten"}:
            raise FusionError(
                f"Unsupported image reduction '{self._image_reduction}'. "
                "Use 'summary' or 'flatten'."
            )

        self._patient_id_column = patient_id_column

    def transform(
        self,
        dataframe: pd.DataFrame,
        images: np.ndarray,
        image_sources: Sequence[str | Path] | None = None,
    ) -> FusionResult:
        """
        Fuse tabular features with an image batch.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Preprocessed, all-numeric dataframe from the CSV pipeline.
        images : np.ndarray
            Preprocessed image batch with shape (N, H, W, C),
            (H, W, C), or (H, W).
        image_sources : Sequence[str | Path] | None
            Optional per-image source paths for metadata.

        Returns
        -------
        FusionResult
            Fused feature matrix plus sample metadata and report.

        Raises
        ------
        FusionError
            If the sample counts differ or inputs are invalid.
        """

        df = dataframe.copy()
        n_samples = len(df)

        image_batch = self._to_batch(images)
        if len(image_batch) != n_samples:
            logger.error(
                "Sample mismatch: %d dataframe rows vs %d images",
                n_samples,
                len(image_batch),
            )
            raise FusionError(
                f"Dataframe has {n_samples} rows but images batch has "
                f"{len(image_batch)}. Sample counts must match."
            )

        features, dropped = self._extract_features(df)
        images_reduced = self._reduce_images(image_batch)

        if self._mode == "concatenate":
            fused = np.hstack([features, images_reduced])

        metadata = self._build_metadata(df, image_batch, image_sources, dropped)

        report = FusionReport(
            mode=self._mode,
            image_reduction=self._image_reduction,
            n_samples=n_samples,
            feature_dim=features.shape[1],
            image_dim=images_reduced.shape[1],
            fused_dim=fused.shape[1],
            dropped_columns=dropped,
        )
        logger.info(
            "Fused %d samples: %d features + %d image features = %d dims",
            n_samples,
            features.shape[1],
            images_reduced.shape[1],
            fused.shape[1],
        )
        return FusionResult(
            features=features,
            images=images_reduced,
            fused=fused,
            metadata=metadata,
            report=report,
        )

    def _extract_features(
        self, dataframe: pd.DataFrame
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Extract an all-numeric feature matrix, dropping un-coercible
        columns."""
        numeric = dataframe.apply(pd.to_numeric, errors="coerce")

        dropped: list[str] = []
        keep: list[str] = []
        for column in numeric.columns:
            if numeric[column].isna().all():
                dropped.append(str(column))
                continue
            keep.append(str(column))

        if dropped:
            logger.warning("Dropped non-numeric columns: %s", dropped)

        values = numeric[keep].fillna(0.0).to_numpy(dtype=np.float64)
        return values, tuple(dropped)

    def _reduce_images(self, batch: np.ndarray) -> np.ndarray:
        """Reduce an image batch to a feature matrix."""
        if self._image_reduction == "flatten":
            return batch.reshape(batch.shape[0], -1)

        spatial = batch.reshape(batch.shape[0], -1, batch.shape[-1])
        mean = spatial.mean(axis=1)
        std = spatial.std(axis=1)
        minimum = spatial.min(axis=1)
        maximum = spatial.max(axis=1)
        return np.concatenate([mean, std, minimum, maximum], axis=1)

    def _build_metadata(
        self,
        dataframe: pd.DataFrame,
        batch: np.ndarray,
        image_sources: Sequence[str | Path] | None,
        dropped: tuple[str, ...],
    ) -> tuple[SampleMetadata, ...]:
        """Build per-sample metadata from the fused inputs."""
        dropped_set = set(dropped)
        height, width, channels = batch.shape[1:]

        if self._patient_id_column and self._patient_id_column in dataframe.columns:
            patient_ids = dataframe[self._patient_id_column].astype(str).tolist()
        else:
            patient_ids = [str(i) for i in range(len(dataframe))]

        records: list[SampleMetadata] = []
        sources = image_sources or [None] * len(dataframe)

        for i, (patient_id, source) in enumerate(
            zip(patient_ids, sources, strict=True)
        ):
            features = {
                str(column): native(value)
                for column, value in dataframe.iloc[i].items()
                if str(column) not in dropped_set
            }
            if source is not None:
                images = (
                    ImageInfo(
                        source=str(source),
                        width=width,
                        height=height,
                        channels=channels,
                    ),
                )
            else:
                images = ()
            records.append(
                SampleMetadata(patient_id=patient_id, features=features, images=images)
            )

        return tuple(records)

    @staticmethod
    def _to_batch(images: np.ndarray) -> np.ndarray:
        """Normalize an image array into a (N, H, W, C) batch."""
        data = np.asarray(images)

        if data.ndim == 2:
            return data[np.newaxis, ..., np.newaxis]
        if data.ndim == 3 and data.shape[-1] in (1, 3, 4):
            return data[np.newaxis, ...]
        if data.ndim == 4:
            return data

        raise FusionError(
            f"Images must have shape (N, H, W, C), (H, W, C), or (H, W); "
            f"got {data.ndim} dimensions."
        )


__all__ = ["FusionReport", "FusionResult", "MultimodalFusion"]
