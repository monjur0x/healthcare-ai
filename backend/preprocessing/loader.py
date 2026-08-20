"""
Shared tabular dataset loading.

Loads a raw CSV into an engineered feature frame and encoded labels,
reusing the :class:`preprocessing.csv.pipeline.CSVPipeline`. This is the
single canonical entry point used by both the API training path and the
federated hospital data layer, so preprocessing logic is never duplicated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from preprocessing.csv import CSVPipeline
from preprocessing.logger import get_logger

logger = get_logger(__name__)


def normalize_token(value: str) -> str:
    """
    Lowercase and normalize a column name to the pipeline convention.

    Parameters
    ----------
    value : str
        Raw column name.

    Returns
    -------
    str
        Lowercased snake_case name.
    """

    return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to lowercase snake_case.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw input frame.

    Returns
    -------
    pd.DataFrame
        Frame with normalized, deduplicated column names.
    """

    seen: set[str] = set()
    cleaned: list[str] = []
    for column in dataframe.columns:
        name = normalize_token(str(column))
        if name in seen:
            name = f"{name}_{len(seen)}"
        seen.add(name)
        cleaned.append(name)
    dataframe.columns = cleaned
    return dataframe


def load_classification_frame(
    dataset: str | Path,
    target: str,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """
    Load and preprocess a CSV into a feature frame and encoded labels.

    Parameters
    ----------
    dataset : str | Path
        Path to the source CSV.
    target : str
        Target column name.
    max_rows : int | None
        Optional cap on the number of rows used.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, dict[str, object]]
        Engineered feature frame, integer-encoded label series aligned
        by index, and the fitted scaler's serializable parameters.

    Raises
    ------
    ValueError
        If the target column is missing or the pipeline yields no data.
    """

    source = Path(dataset)
    raw = pd.read_csv(source)
    if max_rows is not None:
        raw = raw.head(max_rows)
    raw = normalize_columns(raw)
    target = normalize_token(target)

    if target not in raw.columns:
        raise ValueError(f"Target column '{target}' not found in {source}.")

    y_raw = raw[target]
    feature_frame = raw.drop(columns=[target])
    for column in ("id", "subject_id"):
        if column in feature_frame.columns:
            feature_frame = feature_frame.drop(columns=[column])

    valid = y_raw.notna()
    feature_frame = feature_frame.loc[valid]
    y_raw = y_raw.loc[valid]

    pipeline = CSVPipeline(input_columns=tuple(feature_frame.columns))
    result = pipeline.run(feature_frame)
    features = result.dataframe
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Pipeline produced no usable features.")

    labels = y_raw.loc[features.index]
    if pd.api.types.is_string_dtype(labels):
        labels = labels.str.strip()
        labels = pd.Series(
            LabelEncoder().fit_transform(labels), index=labels.index, name=target
        )
    else:
        labels = pd.to_numeric(labels).astype(int)
    logger.info(
        "Prepared %d samples, %d features, %d classes from %s",
        features.shape[0],
        features.shape[1],
        labels.nunique(),
        source,
    )
    return features, labels, pipeline.scaler_params()


__all__ = ["load_classification_frame", "normalize_columns", "normalize_token"]
