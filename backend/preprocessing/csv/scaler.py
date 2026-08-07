"""
Feature scaling for numeric CSV columns.

Supports standard (z-score) and Min-Max scaling implemented with NumPy
so that no external ML dependency is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import settings
from ..exceptions import ScalingError
from ..logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScalingReport:
    """
    Metadata describing the scaling transform.
    """

    scaled_columns: tuple[str, ...]
    method: str
    min_parameters: dict[str, float]
    max_parameters: dict[str, float]


class CSVScaler:
    """
    Scale selected numeric columns.

    Parameters
    ----------
    columns : tuple[str, ...] | None
        Numeric columns to scale. If None, all numeric columns are
        selected automatically.
    method : str | None
        Scaling method: "standard" or "minmax". Defaults to
        ``settings.SCALER``.
    """

    def __init__(
        self,
        columns: tuple[str, ...] | None = None,
        method: str | None = None,
    ) -> None:
        self._columns = columns
        self._method = (settings.SCALER if method is None else method).lower()

    def fit(self, dataframe: pd.DataFrame) -> CSVScaler:
        """
        Fit scaling parameters on the selected numeric columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        CSVScaler
            Self, ready for transform.

        Raises
        ------
        ScalingError
            If no numeric columns exist or the method is unsupported.
        """

        if self._columns is None:
            self._columns = tuple(dataframe.select_dtypes(include="number").columns)

        if not self._columns:
            raise ScalingError("No numeric columns available to scale.")

        if self._method not in {"standard", "minmax"}:
            raise ScalingError(
                f"Unsupported scaling method '{self._method}'. "
                "Use 'standard' or 'minmax'."
            )

        self._min_params: dict[str, float] = {}
        self._max_params: dict[str, float] = {}
        self._mean_params: dict[str, float] = {}
        self._std_params: dict[str, float] = {}

        try:
            numeric = dataframe[list(self._columns)].apply(
                pd.to_numeric, errors="coerce"
            )
            for col in self._columns:
                valid = numeric[col].dropna()
                if valid.empty:
                    continue
                self._min_params[col] = float(valid.min())
                self._max_params[col] = float(valid.max())
                self._mean_params[col] = float(valid.mean())
                self._std_params[col] = float(valid.std(ddof=0)) or 1.0
        except Exception as exc:
            logger.error("Scaler fit failed: %s", exc)
            raise ScalingError(f"Scaler fit failed: {exc}") from exc

        self._fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, ScalingReport]:
        """
        Scale the configured columns using fitted parameters.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        tuple[pd.DataFrame, ScalingReport]
            Scaled dataframe and a scaling report.

        Raises
        ------
        ScalingError
            If ``fit`` was not called prior to ``transform``.
        """

        if not getattr(self, "_fitted", False):
            raise ScalingError("CSVScaler must be fitted before transform.")

        work = dataframe.copy()
        numeric = work[list(self._columns)].apply(pd.to_numeric, errors="coerce")

        for col in self._columns:
            if col not in self._mean_params:
                continue
            if self._method == "standard":
                scaled = (numeric[col] - self._mean_params[col]) / self._std_params[col]
            else:
                span = self._max_params[col] - self._min_params[col]
                if span > 0:
                    scaled = (numeric[col] - self._min_params[col]) / span
                else:
                    scaled = numeric[col] - self._min_params[col]
            work[col] = scaled

        report = ScalingReport(
            scaled_columns=self._columns,
            method=self._method,
            min_parameters=self._min_params,
            max_parameters=self._max_params,
        )
        logger.info(
            "Scaled %d columns with '%s'",
            len(self._columns),
            self._method,
        )
        return work, report
