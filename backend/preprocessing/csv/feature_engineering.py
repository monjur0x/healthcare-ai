"""
Feature engineering for CSV data.

Creates derived healthcare-relevant features such as Body Mass Index,
ratios between numeric columns, and age groups.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..exceptions import FeatureEngineeringError
from ..logger import get_logger

logger = get_logger(__name__)

STANDARD_FEATURE_TYPES = ("bmi", "bmi_primitive", "ratio", "age_group", "interaction")


@dataclass(frozen=True)
class FeatureEngineeringReport:
    """
    Summary of engineered features.
    """

    added_columns: tuple[str, ...]
    feature_types: tuple[str, ...]


class CSVFeatureEngineer:
    """
    Build derived features from an existing dataframe.

    Supported feature types:

    - ``bmi``: requires ``weight_kg`` and ``height_cm`` columns.
    - ``ratio``: requires ``numerator`` and ``denominator`` columns.
    - ``age_group``: requires an ``age`` column; buckets into
      clinical age groups.
    - ``interaction``: requires ``left`` and ``right`` columns; computes
      the product.

    feature_types : tuple[str, ...]
        Features to create. Defaults to all supported types.
    strict : bool
        If True, raise ``FeatureEngineeringError`` when a requested
        feature's required columns are missing. If False (default),
        skip the unavailable feature with a logged warning.
    """

    def __init__(
        self,
        feature_types: tuple[str, ...] = STANDARD_FEATURE_TYPES,
        strict: bool = False,
    ) -> None:
        self._feature_types = feature_types
        self._strict = strict

    def _needs(
        self,
        feature_name: str,
        required: set[str],
        work: pd.DataFrame,
    ) -> bool:
        """Return True when the feature can run, otherwise warn or raise."""
        if required.issubset(work.columns):
            return True
        message = f"Feature '{feature_name}' requires columns {required}; skipping."
        if self._strict:
            raise FeatureEngineeringError(message)
        logger.warning(message)
        return False

    @staticmethod
    def _compute_bmi(dataframe: pd.DataFrame) -> pd.Series:
        weight = dataframe["weight_kg"].astype(float)
        height_cm = dataframe["height_cm"].astype(float)
        height_m = height_cm / 100.0
        return weight / (height_m**2)

    @staticmethod
    def _assign_age_group(age: pd.Series) -> pd.Series:
        """Bucket ages into clinical age groups."""
        bins = [0, 18, 40, 60, 120]
        labels = ["child", "adult", "middle_age", "senior"]
        return pd.cut(age, bins=bins, labels=labels, right=False)

    def transform(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
        """
        Add engineered columns to a copy of the dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        tuple[pd.DataFrame, FeatureEngineeringReport]
            Augmented dataframe and a report of added columns.

        Raises
        ------
        FeatureEngineeringError
            If a requested feature cannot be built because required
            columns are missing.
        """

        work = dataframe.copy()
        added: list[str] = []
        created_types: list[str] = []

        try:
            if "bmi" in self._feature_types and self._needs(
                "bmi", {"weight_kg", "height_cm"}, work
            ):
                work["bmi"] = self._compute_bmi(work)
                added.append("bmi")
                created_types.append("bmi")

            if "bmi_primitive" in self._feature_types and self._needs(
                "bmi_primitive", {"weight_kg"}, work
            ):
                work["bmi_primitive"] = work["weight_kg"].astype(float)
                added.append("bmi_primitive")
                created_types.append("bmi_primitive")

            if "ratio" in self._feature_types and self._needs(
                "ratio", {"numerator", "denominator"}, work
            ):
                work["ratio"] = work["numerator"].astype(float) / work[
                    "denominator"
                ].astype(float).replace(0, pd.NA)
                added.append("ratio")
                created_types.append("ratio")

            if "age_group" in self._feature_types and self._needs(
                "age_group", {"age"}, work
            ):
                work["age_group"] = self._assign_age_group(work["age"].astype(float))
                added.append("age_group")
                created_types.append("age_group")

            if "interaction" in self._feature_types and self._needs(
                "interaction", {"left", "right"}, work
            ):
                work["interaction"] = work["left"].astype(float) * work["right"].astype(
                    float
                )
                added.append("interaction")
                created_types.append("interaction")

        except FeatureEngineeringError:
            raise
        except Exception as exc:
            logger.error("Feature engineering failed: %s", exc)
            raise FeatureEngineeringError(f"Feature engineering failed: {exc}") from exc

        report = FeatureEngineeringReport(
            added_columns=tuple(added),
            feature_types=tuple(created_types),
        )
        logger.info("Engineered features: %s", added)
        return work, report
