"""
Tests for the image preprocessing module.
"""

from __future__ import annotations

import numpy as np
import pytest

from PIL import Image

from preprocessing.exceptions import (
    CorruptedImageError,
    InvalidImageError,
    UnsupportedFileTypeError,
)
from preprocessing.image import (
    ImageAugmenter,
    ImageLoader,
    ImageNormalizer,
    ImagePipeline,
    ImageValidator,
    preprocess_batch,
    preprocess_image,
)


@pytest.fixture
def sample_image(tmp_path) -> tuple[object, np.ndarray]:
    """Write a synthetic 64x48 RGB PNG and return (path, reference array)."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    path = tmp_path / "sample.png"
    Image.fromarray(array).save(path)
    return path, array


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_is_supported(sample_image) -> None:
    path, _ = sample_image
    assert ImageValidator.is_supported(path)
    assert not ImageValidator.is_supported(str(path).replace(".png", ".txt"))


def test_validator_accepts_file(sample_image) -> None:
    path, _ = sample_image
    result = ImageValidator().validate_file(path)
    assert result.is_valid
    assert result.width == 64
    assert result.height == 48
    assert result.channels == 3


def test_validator_rejects_unsupported_type(tmp_path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("not an image")
    with pytest.raises(UnsupportedFileTypeError):
        ImageValidator().validate_file(path)


def test_validator_rejects_corrupted_file(tmp_path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n this is not a valid png payload")
    with pytest.raises(CorruptedImageError):
        ImageValidator().validate_file(path)


def test_validator_rejects_missing_path() -> None:
    with pytest.raises(InvalidImageError):
        ImageValidator().validate_file("does/not/exist.png")


def test_validator_accepts_array() -> None:
    result = ImageValidator().validate_array(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.is_valid
    assert result.channels == 3


def test_validator_rejects_bad_ndim() -> None:
    with pytest.raises(InvalidImageError):
        ImageValidator().validate_array(np.zeros((5,), dtype=np.uint8))


def test_validator_rejects_bad_channels() -> None:
    with pytest.raises(InvalidImageError):
        ImageValidator().validate_array(np.zeros((5, 5, 2), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_loads_file(sample_image) -> None:
    path, array = sample_image
    loaded = ImageLoader(resize=False, channels=3).load_file(path)
    assert loaded.dtype == np.uint8
    assert loaded.shape == (48, 64, 3)
    assert loaded[0, 0, 0] == array[0, 0, 0]


def test_loader_loads_bytes(sample_image) -> None:
    path, array = sample_image
    data = path.read_bytes()
    loaded = ImageLoader(resize=False, channels=3).load_bytes(data)
    assert loaded.shape == (48, 64, 3)
    assert loaded[0, 0, 0] == array[0, 0, 0]


def test_loader_loads_array() -> None:
    source = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    loaded = ImageLoader().load_array(source)
    assert loaded.dtype == np.uint8
    assert loaded.shape == source.shape


def test_loader_grayscale_conversion() -> None:
    source = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    loaded = ImageLoader(channels=1).load_array(source)
    assert loaded.ndim == 2


def test_loader_batch(sample_image) -> None:
    path, _ = sample_image
    batch = ImageLoader(size=(16, 16), channels=3).load_batch([path, path])
    assert batch.shape == (2, 16, 16, 3)
    assert batch.dtype == np.uint8


def test_loader_rejects_missing_path() -> None:
    with pytest.raises(InvalidImageError):
        ImageLoader().load_file("does/not/exist.png")


def test_loader_rejects_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "data.xyz"
    path.write_text("nope")
    with pytest.raises(UnsupportedFileTypeError):
        ImageLoader().load_file(path)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


def test_normalizer_minmax() -> None:
    array = np.array([[[0], [255]], [[64], [128]]], dtype=np.uint8)
    scaled, report = ImageNormalizer(mode="minmax").transform(array)
    assert report.mode == "minmax"
    assert scaled.dtype == np.float32
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)


def test_normalizer_zero_mean() -> None:
    array = np.array([[[0], [255]]], dtype=np.uint8)
    scaled, _ = ImageNormalizer(mode="zero_mean").transform(array)
    assert scaled[0, 0, 0] == pytest.approx(-1.0)
    assert scaled[0, 1, 0] == pytest.approx(1.0)


def test_normalizer_standard_uses_defaults() -> None:
    array = np.full((8, 8, 3), 128, dtype=np.uint8)
    scaled, report = ImageNormalizer(mode="standard").transform(array)
    assert report.mean is not None
    assert scaled.dtype == np.float32


def test_normalizer_standard_fit() -> None:
    array = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    normalizer = ImageNormalizer(mode="standard").fit(array)
    scaled, report = normalizer.transform(array)
    assert report.mean is not None
    assert np.isfinite(scaled).all()


def test_normalizer_rejects_unknown_mode() -> None:
    from preprocessing.exceptions import ImageNormalizationError

    with pytest.raises(ImageNormalizationError):
        ImageNormalizer(mode="bogus")


def test_normalizer_rejects_empty() -> None:
    from preprocessing.exceptions import ImageNormalizationError

    with pytest.raises(ImageNormalizationError):
        ImageNormalizer().transform(np.zeros((0, 0), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------


def test_augmenter_disabled_returns_identity() -> None:
    array = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    result, report = ImageAugmenter(enabled=False).transform(array)
    assert report.enabled is False
    assert report.applied == ()
    assert np.array_equal(result, array)


def test_augmenter_is_deterministic() -> None:
    array = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    first = ImageAugmenter(enabled=True, seed=7, operations=("horizontal_flip",))
    second = ImageAugmenter(enabled=True, seed=7, operations=("horizontal_flip",))
    a, report_a = first.transform(array)
    b, report_b = second.transform(array)
    assert report_a.applied == report_b.applied
    assert np.array_equal(a, b)


def test_augmenter_applies_ops() -> None:
    array = np.random.randint(0, 256, size=(16, 16, 3), dtype=np.uint8)
    result, report = ImageAugmenter(
        enabled=True, seed=1, apply_probability=1.0
    ).transform(array)
    assert report.applied
    assert not np.array_equal(result, array)


def test_augmenter_flip_changes_pixels() -> None:
    array = np.zeros((16, 16), dtype=np.uint8)
    array[:, 0] = 255
    flipped = ImageAugmenter._flip_horizontal(array)
    assert flipped[:, -1].max() == 255
    assert flipped[:, 0].max() == 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_single_image(sample_image) -> None:
    path, _ = sample_image
    pipeline = ImagePipeline(size=(32, 32), channels=3)
    result = pipeline.run(path)
    assert result.shape == (32, 32, 3)
    assert result.dtype == "float32"
    assert result.image.min() >= 0.0
    assert result.image.max() <= 1.0
    assert "normalization" in result.reports
    assert "validation" in result.reports


def test_pipeline_batch(sample_image) -> None:
    path, _ = sample_image
    pipeline = ImagePipeline(size=(16, 16), channels=3)
    result = pipeline.transform_batch([path, path])
    assert result.shape == (2, 16, 16, 3)
    assert "validation" in result.reports
    assert len(result.reports["validation"]) == 2


def test_pipeline_skips_normalization(sample_image) -> None:
    path, _ = sample_image
    pipeline = ImagePipeline(size=(16, 16), normalize=False)
    result = pipeline.run(path)
    assert result.dtype == "uint8"
    assert result.reports["normalization"] is None


def test_pipeline_fit_standard(sample_image) -> None:
    path, _ = sample_image
    pipeline = ImagePipeline(size=(16, 16), normalize_mode="standard").fit([path, path])
    result = pipeline.run(path)
    assert result.reports["normalization"] is not None


def test_preprocess_image_function(sample_image) -> None:
    path, _ = sample_image
    array = preprocess_image(path, size=(16, 16))
    assert array.shape == (16, 16, 3)


def test_preprocess_batch_function(sample_image) -> None:
    path, _ = sample_image
    batch = preprocess_batch([path, path], size=(16, 16))
    assert batch.shape == (2, 16, 16, 3)
