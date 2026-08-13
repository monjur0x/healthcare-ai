"""
Multimodal preprocessing.

Provides shared metadata schemas and fusion for combining preprocessed
tabular (EHR/CSV) and medical image data into unified feature matrices.
"""

from .fusion import FusionReport, FusionResult, MultimodalFusion
from .metadata import ImageInfo, SampleMetadata, native

__all__ = [
    "FusionReport",
    "FusionResult",
    "ImageInfo",
    "MultimodalFusion",
    "SampleMetadata",
    "native",
]
