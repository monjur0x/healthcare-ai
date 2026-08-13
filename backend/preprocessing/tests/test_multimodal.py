"""
Tests for the multimodal preprocessing module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.exceptions import FusionError
from preprocessing.multimodal import MultimodalFusion, SampleMetadata
from preprocessing.multimodal.metadata import native


@pytest.fixture
def features_frame() -> pd.DataFrame:
    """A small all-numeric preprocessed dataframe."""
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "age": [30, 45, 60],
            "bmi": [22.5, 27.0, 31.2],
            "target": [0, 1, 0],
        }
    )


@pytest.fixture
def image_batch() -> np.ndarray:
    """A small channels-last image batch."""
    rng = np.random.default_rng(0)
    return rng.random((3, 8, 8, 3), dtype=np.float32)


def test_fusion_summary_shapes(features_frame, image_batch) -> None:
    result = MultimodalFusion().transform(features_frame, image_batch)
    assert result.features.shape == (3, 3)
    assert result.images.shape == (3, 12)
    assert result.fused.shape == (3, 15)
    assert result.report.fused_dim == 15


def test_fusion_flatten_shapes(features_frame, image_batch) -> None:
    result = MultimodalFusion(image_reduction="flatten").transform(
        features_frame, image_batch
    )
    assert result.images.shape == (3, 8 * 8 * 3)
    assert result.fused.shape == (3, 3 + 8 * 8 * 3)


def test_fusion_sample_mismatch_raises(features_frame, image_batch) -> None:
    with pytest.raises(FusionError):
        MultimodalFusion().transform(features_frame, image_batch[:2])


def test_fusion_drops_non_numeric_columns() -> None:
    frame = pd.DataFrame({"age": [30.0, 40.0], "note": ["a", "b"]})
    result = MultimodalFusion().transform(frame, np.zeros((2, 4, 4, 3)))
    assert result.features.shape == (2, 1)
    assert "note" in result.report.dropped_columns


def test_fusion_single_image() -> None:
    frame = pd.DataFrame({"age": [30.0], "bmi": [22.5]})
    single = np.random.default_rng(1).random((8, 8, 3), dtype=np.float32)
    result = MultimodalFusion().transform(frame, single)
    assert result.fused.shape == (1, 2 + 12)
    assert result.report.n_samples == 1


def test_fusion_metadata_patient_ids(features_frame, image_batch) -> None:
    result = MultimodalFusion(patient_id_column="patient_id").transform(
        features_frame, image_batch
    )
    assert isinstance(result.metadata[0], SampleMetadata)
    assert [m.patient_id for m in result.metadata] == ["p1", "p2", "p3"]
    assert set(result.metadata[0].features) == {"age", "bmi", "target"}


def test_fusion_metadata_image_sources(features_frame, image_batch, tmp_path) -> None:
    sources = [tmp_path / f"img{i}.png" for i in range(3)]
    result = MultimodalFusion().transform(
        features_frame, image_batch, image_sources=sources
    )
    assert result.metadata[0].images[0].width == 8
    assert result.metadata[0].images[0].channels == 3


def test_fusion_rejects_unknown_mode() -> None:
    with pytest.raises(FusionError):
        MultimodalFusion(mode="bogus")


def test_fusion_rejects_unknown_reduction() -> None:
    with pytest.raises(FusionError):
        MultimodalFusion(image_reduction="bogus")


def test_fusion_rejects_bad_image_ndim(features_frame) -> None:
    with pytest.raises(FusionError):
        MultimodalFusion().transform(features_frame, np.zeros((5,), dtype=np.float32))


def test_fusion_result_to_dict(features_frame, image_batch) -> None:
    result = MultimodalFusion().transform(features_frame, image_batch)
    payload = result.to_dict()
    assert payload["fused_shape"] == [3, 15]
    assert payload["report"]["fused_dim"] == 15
    assert len(payload["metadata"]) == 3


def test_native_converts_numpy_scalars() -> None:
    assert native(np.float32(2.5)) == 2.5
    assert native(np.nan) is None
    assert native("text") == "text"
