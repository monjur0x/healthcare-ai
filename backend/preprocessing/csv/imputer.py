"""
Missing value handling for CSV data.

Drops columns that exceed the maximum allowed missing ratio and imputes
remaining missing values using the configured strategy.
"""

from __future__ import annotations

from collections.abc import Mapping
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
        self._fill_values: dict[str, object] = {}
        self._dropped_columns: tuple[str, ...] = ()
        self._fitted = False

    def fit(self, dataframe: pd.DataFrame) -> CSVImputer:
        """
        Compute imputation metadata from the dataframe.

        Records which columns exceed the missing ratio (dropped) and
        the fill value for every kept column, so later batches reuse
        training-time statistics instead of their own.

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
        self._dropped_columns = tuple(
            column
            for column in dataframe.columns
            if self._max_missing_ratio < 1.0
            and dataframe[column].isnull().mean() > self._max_missing_ratio
        )
        self._fill_values = {}
        for column in dataframe.columns:
            if column in self._dropped_columns:
                continue
            self._fill_values[column] = self._fill_value(
                dataframe[column],
                column in self._categorical_columns,
                column,
            )
        return self

    def _fill_value(self, series: pd.Series, categorical: bool, column: str) -> object:
        """Resolve the fill value for one column under the strategies."""
        if categorical:
            if self._categorical_strategy == "most_frequent":
                mode = series.mode()
                return mode.iloc[0] if not mode.empty else "missing"
            return "missing"
        if self._numeric_strategy == "median":
            value = series.median()
        elif self._numeric_strategy == "most_frequent":
            mode = series.mode()
            value = mode.iloc[0] if not mode.empty else float("nan")
        else:
            value = series.mean()
        if pd.isna(value):
            raise EmptyDatasetError(
                f"Column '{column}' has no usable values for imputation."
            )
        return value

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

        if not getattr(self, "_fitted", False):
            # First batch (usually training): learn decisions from it.
            self.fit(work)

        # Never drop a column the training data kept: a sparse inference
        # batch must reuse training-time fill values instead of changing
        # shape (dropped columns break downstream stages).
        drop_cols = [
            column for column in self._dropped_columns if column in work.columns
        ]
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
            stored = self._fill_values.get(col)
            if stored is not None:
                work[col] = work[col].fillna(stored)
            elif col in categorical_columns:
                method = self._categorical_strategy
                if method == "most_frequent":
                    mode = work[col].mode()
                    work[col] = work[col].fillna(
                        mode.iloc[0] if not mode.empty else "missing"
                    )
                else:
                    work[col] = work[col].fillna("missing")
            else:
                method = self._numeric_strategy
                if method == "median":
                    work[col] = work[col].fillna(work[col].median())
                elif method == "most_frequent":
                    mode = work[col].mode()
                    if mode.empty:
                        raise EmptyDatasetError(
                            f"Column '{col}' has no usable values for imputation."
                        )
                    work[col] = work[col].fillna(mode.iloc[0])
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

    def params(self) -> dict[str, object]:
        """
        Return the fitted imputation state as a serializable mapping.

        Returns
        -------
        dict[str, object]
            Strategies, dropped columns, and per-column fill values
            (empty when not fitted).
        """

        if not self._fitted:
            return {}
        return {
            "max_missing_ratio": self._max_missing_ratio,
            "numeric_strategy": self._numeric_strategy,
            "categorical_strategy": self._categorical_strategy,
            "dropped_columns": list(self._dropped_columns),
            "fill_values": dict(self._fill_values),
        }

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> CSVImputer:
        """
        Rebuild a fitted imputer from persisted parameters.

        Parameters
        ----------
        params : Mapping[str, object]
            Output of :meth:`params`.

        Returns
        -------
        CSVImputer
            A fitted imputer reproducing the original transform.
        """

        imputer = cls(
            max_missing_ratio=params.get("max_missing_ratio", 0.3),
            numeric_strategy=str(params.get("numeric_strategy") or "mean"),
            categorical_strategy=str(
                params.get("categorical_strategy") or "most_frequent"
            ),
        )
        imputer._dropped_columns = tuple(params.get("dropped_columns") or ())
        imputer._fill_values = dict(params.get("fill_values") or {})
        imputer._numeric_columns = ()
        imputer._categorical_columns = ()
        imputer._fitted = True
        return imputer
