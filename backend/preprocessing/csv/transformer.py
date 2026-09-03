"""
Pipeline stage that orchestrates the CSV feature preparation steps.

The transformer composes cleaning, validation, imputation, feature
engineering, scaling, and encoding into a single reusable transformer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import settings
from ..logger import get_logger
from .cleaner import CSVCleaner
from .encoder import CSVEncoder, EncodingReport
from .feature_engineering import CSVFeatureEngineer, FeatureEngineeringReport
from .imputer import CSVImputer, ImputationReport
from .scaler import CSVScaler, ScalingReport
from .validator import CSVValidator, ValidationResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class CSVTransformResult:
    """
    Result of running the full CSV transformation.
    """

    dataframe: pd.DataFrame
    validation: ValidationResult | None
    imputation: ImputationReport | None
    encoding: EncodingReport | None
    features: FeatureEngineeringReport | None
    scaling: ScalingReport | None


class CSVTransformer:
    """
    Compose all CSV preprocessing stages into one transform.

    Column names are normalized to lowercase snake_case before any
    validation or feature logic runs, so callers should specify required
    and target columns using normalized names.

    Parameters
    ----------
    required_columns : tuple[str, ...]
        Normalized columns required for validation.
    input_columns : tuple[str, ...] | None
        Normalized columns to retain. If None, all columns are kept.
    encode_columns : tuple[str, ...] | None
        Normalized columns to encode as categorical. If None, all
        object/category columns are encoded.
    encode_mode : str
        Encoding mode: "label" or "onehot".
    scale_columns : tuple[str, ...] | None
        Normalized numeric columns to scale. If None, all numeric
        columns are scaled.
    scale_method : str | None
        Scaling method; defaults to ``settings.SCALER``.
    encoder_params : dict[str, object] | None
        Persisted ``CSVEncoder.params()`` to reuse at inference instead of
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
        enable_feature_engineering: bool | None = None,
    ) -> None:
        self._validator = CSVValidator(required_columns=required_columns)
        self._cleaner = CSVCleaner()
        self._imputer = CSVImputer()
        if encoder_params is not None:
            self._encoder = CSVEncoder.from_params(encoder_params)
        else:
            self._encoder = CSVEncoder(columns=encode_columns, mode=encode_mode)
        self._engineer = CSVFeatureEngineer()
        self._scaler_params = scaler_params
        if scaler_params is not None:
            self._scaler = CSVScaler.from_params(scaler_params)
        else:
            self._scaler = CSVScaler(columns=scale_columns, method=scale_method)
        self._encoder_params = encoder_params
        self._encode_columns = encode_columns
        self._encode_mode = encode_mode
        self._input_columns = input_columns
        self._enable_feature_engineering = (
            settings.ENABLE_FEATURE_ENGINEERING
            if enable_feature_engineering is None
            else enable_feature_engineering
        )

    def fit(self, dataframe: pd.DataFrame) -> CSVTransformer:
        """
        Run the pipeline once to fit all trainable stages.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe to fit on.

        Returns
        -------
        CSVTransformer
            Self, fit and ready for transform.
        """

        if self._encoder_params is None:
            self._encoder = CSVEncoder(
                columns=self._encode_columns, mode=self._encode_mode
            )
        self.transform(dataframe)
        self._fitted = True
        return self

    def transform(self, dataframe: pd.DataFrame) -> CSVTransformResult:
        """
        Run the full CSV transformation pipeline.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.

        Returns
        -------
        CSVTransformResult
            Final dataframe plus per-stage reports.
        """

        # 1. Clean and normalize column names first so all later stages
        #    operate on normalized names.
        work = self._cleaner.transform(dataframe)

        # 2. Validate the cleaned dataframe.
        validation = self._validator.validate_dataframe(work)

        # 3. Handle missing values.
        work, imputation = self._imputer.transform(work)

        # 4. Build derived features.
        if self._enable_feature_engineering:
            work, features = self._engineer.transform(work)
        else:
            features = None

        # 5. Scale numeric columns before encoding so one-hot columns are
        #    not accidentally scaled. When persisted scaler params were
        #    provided at construction time, reuse them (inference-time
        #    consistency) instead of re-fitting on this batch.
        if self._scaler_params is None:
            self._scaler.fit(work)
        work, scaling = self._scaler.transform(work)

        # 6. Encode categorical columns last. A re-fit on every batch
        #    would remap categories (a single-row batch always maps its
        #    sole value to 0), so reuse persisted params — or the
        #    first fit — for inference-time consistency.
        if self._encoder_params is None and not getattr(
            self._encoder, "_fitted", False
        ):
            self._encoder.fit(work)
        work, encoding = self._encoder.transform(work)

        if self._input_columns is not None:
            keep = [col for col in self._input_columns if col in work.columns]
            work = work[keep]

        result = CSVTransformResult(
            dataframe=work,
            validation=validation,
            imputation=imputation,
            encoding=encoding,
            features=features,
            scaling=scaling,
        )
        logger.info(
            "CSV transformation completed: %d rows, %d columns",
            len(work),
            work.shape[1],
        )
        return result

    def scaler_params(self) -> dict[str, object]:
        """
        Return the fitted scaler's serializable parameters.

        Returns
        -------
        dict[str, object]
            ``CSVScaler.params()`` output; empty dict when the scaler has
            not been fitted yet.
        """

        if getattr(self._scaler, "_fitted", False):
            return self._scaler.params()
        return {}

    def encoder_params(self) -> dict[str, object]:
        """
        Return the fitted encoder's serializable parameters.

        Returns
        -------
        dict[str, object]
            ``CSVEncoder.params()`` output; empty dict when the encoder
            has not been fitted yet.
        """

        if getattr(self._encoder, "_fitted", False):
            return self._encoder.params()
        return {}
