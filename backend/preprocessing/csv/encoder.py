"""
Categorical encoding for CSV columns.

Converts categorical columns into numeric representations using either
label encoding or one-hot encoding.
"""

from __future__ import annotations

from collections.abc import Mapping
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
        self._label_mapping: dict[str, dict[str, int]] = {}
        self._one_hot_columns: tuple[str, ...] = ()
        self._fitted = False

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
        self._label_mapping = {}

        if self._mode == "label":
            for col in self._columns:
                categories = dataframe[col].dropna().unique()
                self._label_mapping[col] = {
                    category: idx
                    for idx, category in enumerate(sorted(map(str, categories)))
                }
            self._one_hot_columns = ()
        elif self._mode == "onehot":
            present = [col for col in self._columns if col in dataframe.columns]
            if present:
                dummies = pd.get_dummies(
                    dataframe[present],
                    columns=present,
                    drop_first=self._drop_first,
                    dtype=int,
                )
                self._one_hot_columns = tuple(dummies.columns)
            else:
                self._one_hot_columns = ()

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
        ValueError
            If label mode meets a non-null category unseen during ``fit``.
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
                raw = work[col]
                values = raw.astype(str)
                unseen = sorted(set(values[raw.notna()].unique()) - set(mapping))
                if unseen:
                    raise ValueError(
                        f"Column '{col}' contains categories unseen during "
                        f"fitting: {unseen}."
                    )
                work[col] = values.map(mapping)
                label_encoded.append(col)
        elif self._mode == "onehot":
            columns_list = list(self._columns)
            encoded = pd.get_dummies(
                work[columns_list],
                columns=columns_list,
                drop_first=self._drop_first,
                dtype=int,
            )
            if self._one_hot_columns:
                extra = [
                    col for col in encoded.columns if col not in self._one_hot_columns
                ]
                if extra:
                    logger.warning("Dropping unseen one-hot columns: %s", extra)
                encoded = encoded.reindex(
                    columns=list(self._one_hot_columns), fill_value=0
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

    def params(self) -> dict[str, object]:
        """
        Return the fitted encoding state as a serializable mapping.

        Returns
        -------
        dict[str, object]
            Encoding mode, column list, label mappings, and one-hot
            column list (empty when not fitted).
        """

        if not self._fitted:
            return {}
        return {
            "columns": list(self._columns or ()),
            "mode": self._mode,
            "drop_first": self._drop_first,
            "label_mapping": {
                column: dict(mapping) for column, mapping in self._label_mapping.items()
            },
            "one_hot_columns": list(self._one_hot_columns),
        }

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> CSVEncoder:
        """
        Rebuild a fitted encoder from persisted parameters.

        Parameters
        ----------
        params : Mapping[str, object]
            Output of :meth:`params`.

        Returns
        -------
        CSVEncoder
            A fitted encoder reproducing the original transform.
        """

        encoder = cls(
            columns=tuple(params.get("columns") or ()),
            mode=str(params.get("mode") or "label"),
            drop_first=bool(params.get("drop_first", False)),
        )
        raw_mapping = params.get("label_mapping") or {}
        encoder._label_mapping = {
            str(column): {str(value): int(code) for value, code in mapping.items()}
            for column, mapping in raw_mapping.items()
        }
        encoder._one_hot_columns = tuple(params.get("one_hot_columns") or ())
        encoder._fitted = True
        return encoder
