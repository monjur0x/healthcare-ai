"""
Missing value handling for CSV data.

Drops columns that exceed the maximum allowed missing ratio and imputes
remaining missing values using the configured strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import settings
from ..exceptions import EmptyDatasetError
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ImputationReport:
    """
    Summary of imputation decisions.
    """

    dropped_columns: tuple[str, ...]
    imputed_columns: tuple[str, ...]
    imputation_method: str
    missing_before: int
    missing_after: int


class CSVImputer:
    """
    Imputes missing values in a structured dataframe.

    Numeric columns are imputed with the mean by default; categorical
    columns with the mode. Columns whose missing ratio exceeds
    ``settings.MAX_MISSING_RATIO`` are dropped.

    Parameters
    ----------
    max_missing_ratio : float | None
        Maximum fraction of missing values a column may have before it
        is dropped. Defaults to ``settings.MAX_MISSING_RATIO``.
    numeric_strategy : str
        Strategy for numeric columns: "mean", "median", or "most_frequent".
    categorical_strategy : str
        Strategy for categorical columns: "most_frequent" or "constant".
    """

    def __init__(
        self,
        max_missing_ratio: float | None = None,
        numeric_strategy: str = "mean",
        categorical_strategy: str = "most_frequent",
    ) -> None:
        self._max_missing_ratio = (
            settings.MAX_MISSING_RATIO
            if max_missing_ratio is None
            else max_missing_ratio
        )
        self._numeric_strategy = numeric_strategy
        self._categorical_strategy = categorical_strategy

    def fit(self, dataframe: pd.DataFrame) -> CSVImputer:
        """
        Compute imputation metadata from the dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        CSVImputer
            Self, configured for transform.
        """

        self._fitted = True
        self._numeric_columns = tuple(dataframe.select_dtypes(include="number").columns)
        self._categorical_columns = tuple(
            dataframe.select_dtypes(include=["object", "category"]).columns
        )
        return self

    def transform(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, ImputationReport]:
        """
        Drop over-missing columns and impute remaining missing values.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        tuple[pd.DataFrame, ImputationReport]
            Imputed dataframe and a report of imputation decisions.

        Raises
        ------
        EmptyDatasetError
            If imputation removes every column or the dataset becomes empty.
        """

        if dataframe.empty:
            raise EmptyDatasetError("Cannot impute an empty dataset.")

        categorical_columns = set(
            dataframe.select_dtypes(include=["object", "category"]).columns
        )
        missing_before = int(dataframe.isnull().sum().sum())
        work = dataframe.copy()

        drop_cols: list[str] = []
        if self._max_missing_ratio < 1.0:
            for col in work.columns:
                ratio = work[col].isnull().mean()
                if ratio > self._max_missing_ratio:
                    drop_cols.append(col)

        work = work.drop(columns=drop_cols)

        if work.empty:
            logger.error(
                "All columns dropped; max_missing_ratio=%s",
                self._max_missing_ratio,
            )
            raise EmptyDatasetError(
                "All columns were dropped due to excessive missing values."
            )

        imputed_cols: list[str] = []
        for col in work.columns:
            n_missing = int(work[col].isnull().sum())
            if n_missing == 0:
                continue
            if col in categorical_columns:
                method = self._categorical_strategy
                if method == "most_frequent":
                    value = (
                        work[col].mode().iloc[0] if not work[col].mode().empty else None
                    )
                    work[col] = work[col].fillna(value)
                else:
                    work[col] = work[col].fillna("missing")
            else:
                method = self._numeric_strategy
                if method == "median":
                    work[col] = work[col].fillna(work[col].median())
                elif method == "most_frequent":
                    work[col] = work[col].fillna(work[col].mode()[0])
                else:
                    work[col] = work[col].fillna(work[col].mean())
            imputed_cols.append(col)

        missing_after = int(work.isnull().sum().sum())
        report = ImputationReport(
            dropped_columns=tuple(drop_cols),
            imputed_columns=tuple(imputed_cols),
            imputation_method=self._numeric_strategy,
            missing_before=missing_before,
            missing_after=missing_after,
        )
        logger.info(
            "Imputation: dropped %s, imputed %s (missing %d -> %d)",
            drop_cols,
            imputed_cols,
            missing_before,
            missing_after,
        )
        return work, report
