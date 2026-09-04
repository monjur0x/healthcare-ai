"""
High-level CSV preprocessing pipeline.

The pipeline is the reusable entry point for Flower, FastAPI, and
CrewAI. It reads CSV data from a path or bytes and produces a fully
preprocessed dataframe plus stage reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd

from ..exceptions import InvalidCSVError
from ..logger import get_logger
from .transformer import CSVTransformer

logger = get_logger(__name__)


@dataclass(frozen=True)
class CSVResult:
    """
    Final pipeline output.
    """

    dataframe: pd.DataFrame
    reports: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Serialize the result to a JSON-friendly dictionary.

        Returns
        -------
        dict
            Dictionary with dataframe rows and metadata.
        """

        return {
            "rows": self.dataframe.to_dict(orient="records"),
            "shape": {
                "rows": self.dataframe.shape[0],
                "columns": self.dataframe.shape[1],
            },
            "reports": self.reports,
        }


class CSVPipeline:
    """
    Run the full preprocessing sequence on CSV input.

    Parameters
    ----------
    required_columns : tuple[str, ...]
        Columns required for validation.
    input_columns : tuple[str, ...] | None
        Columns to retain. If None, all columns are kept.
    encode_columns : tuple[str, ...] | None
        Columns to encode as categorical.
    encode_mode : str
        Encode mode: "label" or "onehot".
    scale_columns : tuple[str, ...] | None
        Numeric columns to scale.
    scale_method : str | None
        Scaling method; defaults to ``settings.SCALER``.
    scaler_params : dict[str, object] | None
        Persisted ``CSVScaler.params()`` to reuse at inference instead of
        re-fitting on this batch (None re-fits as before).
    encoder_params : dict[str, object] | None
        Persisted ``CSVEncoder.params()`` to reuse at inference instead of
        re-fitting on this batch (None fits as before).
    imputer_params : dict[str, object] | None
        Persisted ``CSVImputer.params()`` to reuse at inference instead of
        re-fitting on this batch (None fits as before).
    enable_feature_engineering : bool | None
        Whether to create derived features. Defaults to
        ``settings.ENABLE_FEATURE_ENGINEERING``.
    """

    def __init__(
        self,
        required_columns: tuple[str, ...] = (),
        input_columns: tuple[str, ...] | None = None,
        encode_columns: tuple[str, ...] | None = None,
        encode_mode: str = "label",
        scale_columns: tuple[str, ...] | None = None,
        scale_method: str | None = None,
        scaler_params: dict[str, object] | None = None,
        encoder_params: dict[str, object] | None = None,
        imputer_params: dict[str, object] | None = None,
        enable_feature_engineering: bool | None = None,
    ) -> None:
        self._transformer = CSVTransformer(
            required_columns=required_columns,
            input_columns=input_columns,
            encode_columns=encode_columns,
            encode_mode=encode_mode,
            scale_columns=scale_columns,
            scale_method=scale_method,
            scaler_params=scaler_params,
            encoder_params=encoder_params,
            imputer_params=imputer_params,
            enable_feature_engineering=enable_feature_engineering,
        )

    @staticmethod
    def _read_input(source: str | bytes | pd.DataFrame) -> pd.DataFrame:
        """Read CSV from a path, bytes, or an existing dataframe."""
        if isinstance(source, pd.DataFrame):
            return source.copy()

        if isinstance(source, bytes):
            try:
                text = source.decode("utf-8")
            except UnicodeDecodeError as exc:
                logger.error("CSV bytes could not be decoded as UTF-8: %s", exc)
                raise InvalidCSVError(
                    f"CSV bytes could not be decoded as UTF-8: {exc}"
                ) from exc
            return pd.read_csv(StringIO(text))

        path = Path(source)
        try:
            return pd.read_csv(path)
        except Exception as exc:
            logger.error("Failed to read CSV from %s: %s", source, exc)
            raise InvalidCSVError(f"Failed to read CSV from '{source}': {exc}") from exc

    def fit(self, source: str | bytes | pd.DataFrame) -> CSVPipeline:
        """
        Fit trainable stages on the provided data.

        Parameters
        ----------
        source : str | bytes | pd.DataFrame
            CSV path, raw CSV bytes, or dataframe.

        Returns
        -------
        CSVPipeline
            Self, fitted.
        """

        dataframe = self._read_input(source)
        self._transformer.fit(dataframe)
        self._fitted = True
        return self

    def transform(self, source: str | bytes | pd.DataFrame) -> CSVResult:
        """
        Run the pipeline end to end.

        Parameters
        ----------
        source : str | bytes | pd.DataFrame
            CSV path, raw CSV bytes, or dataframe.

        Returns
        -------
        CSVResult
            Preprocessed dataframe and per-stage reports.
        """

        dataframe = self._read_input(source)
        result = self._transformer.transform(dataframe)
        reports = {
            "validation": _asdict_optional(result.validation),
            "imputation": _asdict_optional(result.imputation),
            "encoding": _asdict_optional(result.encoding),
            "features": _asdict_optional(result.features),
            "scaling": _asdict_optional(result.scaling),
        }
        logger.info(
            "Pipeline completed: %d rows, %d columns",
            len(result.dataframe),
            result.dataframe.shape[1],
        )
        return CSVResult(dataframe=result.dataframe, reports=reports)

    def run(self, source: str | bytes | pd.DataFrame) -> CSVResult:
        """
        Fit-if-needed and transform in a single call.

        Parameters
        ----------
        source : str | bytes | pd.DataFrame
            CSV path, raw CSV bytes, or dataframe.

        Returns
        -------
        CSVResult
            Preprocessed dataframe and reports.
        """

        if not getattr(self, "_fitted", False):
            self.fit(source)
        return self.transform(source)

    def scaler_params(self) -> dict[str, object]:
        """
        Return the fitted scaler's serializable parameters.

        Returns
        -------
        dict[str, object]
            ``CSVScaler.params()`` output; empty dict when the scaler has
            not been fitted yet.
        """

        return self._transformer.scaler_params()

    def encoder_params(self) -> dict[str, object]:
        """
        Return the fitted encoder's serializable parameters.

        Returns
        -------
        dict[str, object]
            ``CSVEncoder.params()`` output; empty dict when the encoder
            has not been fitted yet.
        """

        return self._transformer.encoder_params()

    def imputer_params(self) -> dict[str, object]:
        """
        Return the fitted imputer's serializable parameters.

        Returns
        -------
        dict[str, object]
            ``CSVImputer.params()`` output; empty dict when the imputer
            has not been fitted yet.
        """

        return self._transformer.imputer_params()


def _asdict_optional(report) -> dict | None:
    """
    Convert a dataclass report to a dict, or return None.
    """

    return asdict(report) if report is not None else None
