"""
Shared metadata schemas for multimodal preprocessing.

Describes the structure of a multimodal sample: a tabular EHR record
plus one or more associated images.
"""

from __future__ import annotations

import math

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ImageInfo:
    """
    Structural metadata for a single image in a multimodal sample.
    """

    source: str
    width: int
    height: int
    channels: int

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """

        return asdict(self)


@dataclass(frozen=True)
class SampleMetadata:
    """
    Metadata for one multimodal sample.

    Attributes
    ----------
    patient_id : str
        Identifier linking the tabular record to its image(s).
    features : dict[str, float | str | None]
        Tabular feature values for the sample.
    images : tuple[ImageInfo, ...]
        Metadata for each associated image.
    target : str | float | None
        Optional label/value for the sample.
    """

    patient_id: str
    features: dict[str, float | str | None]
    images: tuple[ImageInfo, ...] = ()
    target: str | float | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with nested image metadata.
        """

        return asdict(self)


def native(value: Any) -> Any:
    """
    Convert a numpy or pandas scalar to a native Python value.

    Parameters
    ----------
    value : Any
        Value to convert.

    Returns
    -------
    Any
        JSON-friendly native value, or None for missing values.
    """

    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            return None

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


__all__ = ["ImageInfo", "SampleMetadata", "native"]
