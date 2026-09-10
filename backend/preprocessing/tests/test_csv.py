"""
Tests for the CSV preprocessing module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.csv import (
    CSVCleaner,
    CSVEncoder,
    CSVFeatureEngineer,
    CSVImputer,
    CSVPipeline,
    CSVScaler,
    CSVTransformer,
    CSVValidator,
)
from preprocessing.exceptions import (
    EmptyDatasetError,
    InvalidCSVError,
    MissingColumnError,
    UnsupportedFileTypeError,
)


@pytest.fixture
def df_frame() -> pd.DataFrame:
    """A small representative healthcare dataframe."""
    return pd.DataFrame(
        {
            "Age": [30, 40, 50, 60, None],
            "Weight Kg": [80.0, 85.5, None, 70.0, 90.0],
            "Height Cm": [170.0, 175.0, 168.0, None, 180.0],
            "Gender": ["M", "F", "M", "F", "M"],
            "Heart Rate": [72, 78, 65, 90, 110],
        }
    )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_accepts_valid_dataframe(df_frame: pd.DataFrame) -> None:
    result = CSVValidator(required_columns=("Age",)).validate_dataframe(df_frame)
    assert result.is_valid
    assert result.total_rows == 5


def test_validator_raises_missing_column(df_frame: pd.DataFrame) -> None:
    with pytest.raises(MissingColumnError):
        CSVValidator(required_columns=("nonexistent",)).validate_dataframe(df_frame)


def test_validator_raises_empty() -> None:
    with pytest.raises(EmptyDatasetError):
        CSVValidator().validate_dataframe(pd.DataFrame())


def test_validator_rejects_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("a,b\n1,2")
    validator = CSVValidator()
    assert not validator.is_supported(path)
    with pytest.raises(UnsupportedFileTypeError):
        validator.validate_file(path)


def test_validator_parses_csv_file(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4")
    result = CSVValidator().validate_file(path)
    assert result.total_rows == 2


def test_validator_rejects_unparsable_file(tmp_path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"\x80\x81\x82")
    with pytest.raises(InvalidCSVError):
        CSVValidator().validate_file(path)


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------


def test_cleaner_normalizes_columns_and_strips(df_frame: pd.DataFrame) -> None:
    cleaned = CSVCleaner().transform(df_frame)
    assert "Weight Kg" not in cleaned.columns
    assert "weight_kg" in cleaned.columns
    assert "Heart Rate" not in cleaned.columns
    assert "heart_rate" in cleaned.columns


def test_cleaner_removes_duplicates() -> None:
    data = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned = CSVCleaner().transform(data)
    assert len(cleaned) == 2


def test_cleaner_drops_fully_empty_rows() -> None:
    data = pd.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})
    cleaned = CSVCleaner().transform(data)
    assert len(cleaned) == 2


# ---------------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------------


def test_imputer_fills_numeric_missing(df_frame: pd.DataFrame) -> None:
    work, report = CSVImputer().transform(df_frame)
    assert report.missing_after == 0
    assert not work["Age"].isnull().any()
    assert not work["Weight Kg"].isnull().any()


def test_imputer_drops_excessive_missing_columns() -> None:
    data = pd.DataFrame({"keep": [1.0, 2.0, 3.0], "drop": [None, None, 1.0]})
    work, report = CSVImputer(max_missing_ratio=0.4).transform(data)
    assert "drop" not in work.columns
    assert "drop" in report.dropped_columns


def test_imputer_raises_when_all_columns_dropped() -> None:
    data = pd.DataFrame({"drop": [None, None, None]})
    with pytest.raises(EmptyDatasetError):
        CSVImputer(max_missing_ratio=0.2).transform(data)


def test_imputer_rejects_empty() -> None:
    with pytest.raises(EmptyDatasetError):
        CSVImputer().transform(pd.DataFrame())


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_encoder_label_encodes() -> None:
    data = pd.DataFrame({"category": ["b", "a", "b"]})
    encoder = CSVEncoder(mode="label").fit(data)
    work, report = encoder.transform(data)
    assert "category" in report.label_encoded
    assert work["category"].dtype.kind in "iu"


def test_encoder_onehot_encodes() -> None:
    data = pd.DataFrame({"category": ["b", "a", "b"]})
    encoder = CSVEncoder(mode="onehot").fit(data)
    work, _ = encoder.transform(data)
    assert "category_a" in work.columns
    assert "category_b" in work.columns
    assert "category" not in work.columns


def test_encoder_requires_fit() -> None:
    with pytest.raises(AttributeError):
        CSVEncoder().transform(pd.DataFrame({"a": ["x"]}))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def test_engineer_builds_bmi() -> None:
    data = pd.DataFrame({"weight_kg": [80.0], "height_cm": [160.0]})
    work, _ = CSVFeatureEngineer(feature_types=("bmi",)).transform(data)
    expected = 80.0 / (1.6**2)
    assert "bmi" in work.columns
    assert work["bmi"].iloc[0] == pytest.approx(expected, rel=1e-6)


def test_engineer_raises_on_missing_columns() -> None:
    data = pd.DataFrame({"weight_kg": [80.0]})
    from preprocessing.exceptions import FeatureEngineeringError

    with pytest.raises(FeatureEngineeringError):
        CSVFeatureEngineer(feature_types=("bmi",), strict=True).transform(data)


def test_engineer_age_group() -> None:
    data = pd.DataFrame({"age": [10, 30, 50, 70]})
    work, _ = CSVFeatureEngineer(feature_types=("age_group",)).transform(data)
    assert set(work["age_group"].astype(str)) == {
        "child",
        "adult",
        "middle_age",
        "senior",
    }


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------


def test_scaler_standard_centers_data() -> None:
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    scaler = CSVScaler(method="standard").fit(data)
    work, report = scaler.transform(data)
    assert np.isclose(work["x"].mean(), 0.0, atol=1e-8)
    assert report.scaled_columns == ("x",)


def test_scaler_minmax_bounds() -> None:
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = CSVScaler(method="minmax").fit(data)
    work, _ = scaler.transform(data)
    assert work["x"].min() == pytest.approx(0.0)
    assert work["x"].max() == pytest.approx(1.0)


def test_scaler_requires_fit() -> None:
    from preprocessing.exceptions import ScalingError

    with pytest.raises(ScalingError):
        CSVScaler().transform(pd.DataFrame({"x": [1.0]}))


def test_scaler_rejects_unknown_method() -> None:
    from preprocessing.exceptions import ScalingError

    with pytest.raises(ScalingError):
        CSVScaler(method="bogus").fit(pd.DataFrame({"x": [1.0]}))


# ---------------------------------------------------------------------------
# Transformer / Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_end_to_end() -> None:
    pipeline = CSVPipeline(
        required_columns=("age",),
        encode_columns=("gender",),
        scale_columns=("heart",),
    )
    data = pd.DataFrame(
        {
            "Age": [30, 40],
            "Heart": [72, 78],
            "Gender": ["M", "F"],
            "row_id": [1, 2],
        }
    )
    result = pipeline.run(data)
    assert len(result.dataframe) == 2
    assert "reports" in result.to_dict()
    assert result.reports["validation"] is not None


def test_pipeline_reads_bytes() -> None:
    csv_bytes = b"Age,Heart\n30,72\n40,78\n"
    pipe = CSVPipeline(required_columns=("age",))
    result = pipe.run(csv_bytes)
    assert len(result.dataframe) == 2


def test_transformer_fit_then_transform(df_frame: pd.DataFrame) -> None:
    transformer = CSVTransformer(required_columns=("age",))
    transformer.fit(df_frame)
    result = transformer.transform(df_frame)
    assert not result.dataframe.empty
    assert result.scaling is None or result.scaling.scaled_columns


def test_pipeline_input_columns_subset() -> None:
    data = pd.DataFrame({"age": [30, 40], "heart": [72, 78], "extra": [9, 9]})
    pipe = CSVPipeline(input_columns=("age", "heart"))
    result = pipe.run(data)
    assert set(result.dataframe.columns) == {"age", "heart"}
