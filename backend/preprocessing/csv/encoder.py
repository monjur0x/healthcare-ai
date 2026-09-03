"""
Categorical encoding for CSV columns.

Converts categorical columns into numeric representations using either
label encoding or one-hot encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class EncodingReport:
    """
    Metadata about the encoding transformations applied.
    """

    label_encoded: tuple[str, ...] = ()
    one_hot_encoded: tuple[str, ...] = ()
    added_columns: tuple[str, ...] = field(default_factory=tuple)
    dropped_original: tuple[str, ...] = field(default_factory=tuple)


class CSVEncoder:
    """
    Encode categorical columns as numeric.

    Parameters
    ----------
    columns : tuple[str, ...] | None
        Columns to encode. If None, all object/category columns are
        selected automatically.
    method : str
        Encoding method: "label" or "onehot".
    drop_first : bool
        Only used with one-hot encoding. If True, drops the first
        category to avoid collinearity.
    """

    def __init__(
        self,
        columns: tuple[str, ...] | None = None,
        mode: str = "label",
        drop_first: bool = False,
    ) -> None:
        self._columns = columns
        self._mode = mode
        self._drop_first = drop_first

    def fit(self, dataframe: pd.DataFrame) -> CSVEncoder:
        """
        Determine which columns to encode and fit mappings.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        CSVEncoder
            Self, ready for transform.
        """

        if self._columns is None:
            self._columns = tuple(
                dataframe.select_dtypes(include=["object", "category"]).columns
            )
        self._label_mapping: dict[str, dict[object, int]] = {}

        if self._mode == "label":
            for col in self._columns:
                categories = dataframe[col].dropna().unique()
                self._label_mapping[col] = {
                    category: idx
                    for idx, category in enumerate(sorted(map(str, categories)))
                }

        self._fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, EncodingReport]:
        """
        Encode the configured categorical columns.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        tuple[pd.DataFrame, EncodingReport]
            Encoded dataframe and an encoding report.

        Raises
        ------
        AttributeError
            If ``fit`` was not called before ``transform``.
        RuntimeError
            If an unsupported encoding mode is requested.
        """

        if not getattr(self, "_fitted", False):
            raise AttributeError("CSVEncoder must be fitted before transform.")

        work = dataframe.copy()
        label_encoded: list[str] = []
        one_hot_encoded: tuple[str, ...] = ()
        added_columns: tuple[str, ...] = ()
        dropped_original: tuple[str, ...] = ()

        if self._mode == "label":
            for col in self._columns:
                if col not in work.columns:
                    continue
                mapping = self._label_mapping.get(col, {})
                work[col] = work[col].astype(str).map(mapping)
                label_encoded.append(col)
        elif self._mode == "onehot":
            columns_list = list(self._columns)
            encoded = pd.get_dummies(
                work[columns_list],
                columns=columns_list,
                drop_first=self._drop_first,
                dtype=int,
            )
            work = work.drop(columns=columns_list)
            work = pd.concat([work, encoded], axis=1)
            one_hot_encoded = self._columns
            added_columns = tuple(encoded.columns)
            dropped_original = self._columns
        else:
            raise RuntimeError(
                f"Unsupported encoding mode '{self._mode}'. Use 'label' or 'onehot'."
            )

        report = EncodingReport(
            label_encoded=tuple(label_encoded),
            one_hot_encoded=one_hot_encoded,
            added_columns=added_columns,
            dropped_original=dropped_original,
        )
        logger.info(
            "Encoding complete using mode '%s' for %d columns",
            self._mode,
            len(label_encoded) + len(one_hot_encoded),
        )
        return work, report
