"""
CSV cleaning utilities.

Removes duplicates, trims whitespace, standardizes column names, and
drops rows that are entirely empty.
"""

from __future__ import annotations

import pandas as pd

from ..config import settings
from ..logger import get_logger

logger = get_logger(__name__)


class CSVCleaner:
    """
    Cleans a structured dataframe.

    The cleaner normalizes column names to lowercase snake_case, strips
    string whitespace, removes fully empty rows, and optionally removes
    duplicate rows.

    Parameters
    ----------
    remove_duplicates : bool | None
        Whether to drop duplicate rows. Defaults to the value of
        ``settings.REMOVE_DUPLICATES``.
    """

    def __init__(self, remove_duplicates: bool | None = None) -> None:
        self._remove_duplicates = (
            settings.REMOVE_DUPLICATES
            if remove_duplicates is None
            else remove_duplicates
        )

    @staticmethod
    def _normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Convert column names to lowercase snake_case."""
        # Preserve order and deduplicate renamed columns.
        seen: set[str] = set()
        cleaned: list[str] = []
        for col in dataframe.columns:
            name = str(col).strip().lower().replace(" ", "_").replace("-", "_")
            if name in seen:
                name = f"{name}_{len(seen)}"
            seen.add(name)
            cleaned.append(name)
        dataframe.columns = cleaned
        return dataframe

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe and return a new dataframe.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe.
        """

        cleaned = dataframe.copy()
        cleaned = self._normalize_columns(cleaned)

        for col in cleaned.columns:
            if cleaned[col].dtype == "object":
                cleaned[col] = cleaned[col].astype(str).str.strip()
                cleaned[col] = cleaned[col].replace(
                    {"": None, "nan": None, "NaN": None, "None": None}
                )

        cleaned = cleaned.dropna(how="all")

        if self._remove_duplicates:
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            removed = before - len(cleaned)
            if removed:
                logger.info("Removed %d duplicate rows", removed)

        logger.info("Cleaned CSV: %d rows", len(cleaned))
        return cleaned
