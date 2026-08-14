"""
Multimodal prediction model.

Consumes the ``FusionResult`` produced by ``preprocessing.multimodal``
and classifies the fused feature matrix. Composes ``TabularClassifier``;
no fusion or feature engineering is performed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from models.base import BaseModel
from models.csv import TabularClassifier
from models.exceptions import InvalidModelInputError
from preprocessing.logger import get_logger
from preprocessing.multimodal import FusionResult

logger = get_logger(__name__)


class FusionClassifier(BaseModel):
    """
    Classifier trained on fused multimodal features.

    Accepts a ``FusionResult`` directly, or any 2D feature matrix (e.g.
    the ``fused`` attribute of a result). The underlying estimator is an
    MLP by default, matching the high-dimensional fused representation.

    Parameters
    ----------
    model_name : str
        Estimator family passed to :class:`TabularClassifier`; defaults
        to ``"mlp"``.
    estimator : Any | None
        Optional pre-built sklearn estimator; overrides ``model_name``.
    random_state : int | None
        Random seed; defaults to ``settings.RANDOM_SEED``.
    """

    def __init__(
        self,
        model_name: str = "mlp",
        estimator: Any | None = None,
        random_state: int | None = None,
    ) -> None:
        self._tabular = TabularClassifier(
            model_name=model_name, estimator=estimator, random_state=random_state
        )
        self._fused_dim: int | None = None

    @property
    def is_fitted(self) -> bool:
        """True if the model is ready for prediction."""
        return self._tabular.is_fitted

    @property
    def model_name(self) -> str:
        """Name of the underlying classifier family."""
        return self._tabular.model_name

    @property
    def fused_dim(self) -> int | None:
        """Number of fused features seen at fit time, if any."""
        return self._fused_dim

    @property
    def classes_(self) -> np.ndarray:
        """Unique class labels learned during fit."""
        return self._tabular.classes_

    @property
    def feature_names(self) -> list[str] | None:
        """Generated fused feature names, if fitted."""
        if self._fused_dim is None:
            return None
        return [f"fused_{index}" for index in range(self._fused_dim)]

    def fit(
        self, X: FusionResult | np.ndarray, y: np.ndarray | None = None
    ) -> FusionClassifier:
        """
        Fit the classifier on fused features and labels.

        Parameters
        ----------
        X : FusionResult | np.ndarray
            Fused output from the multimodal pipeline, or a raw feature
            matrix.
        y : np.ndarray | None
            Target labels. Required.

        Returns
        -------
        FusionClassifier
            Self, fitted.

        Raises
        ------
        InvalidModelInputError
            If the inputs are malformed or labels are missing.
        """

        matrix = self._as_matrix(X)
        if y is None:
            raise InvalidModelInputError(
                "A target label array is required for fitting."
            )
        self._tabular.fit(matrix, np.asarray(y))
        logger.info("Fitted fusion model on %d fused features", self._fused_dim)
        return self

    def predict(self, X: FusionResult | np.ndarray) -> np.ndarray:
        """
        Predict labels from fused features.

        Parameters
        ----------
        X : FusionResult | np.ndarray
            Fused output or feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """

        return self._tabular.predict(self._as_matrix(X))

    def predict_proba(self, X: FusionResult | np.ndarray) -> np.ndarray:
        """
        Predict class probabilities from fused features.

        Parameters
        ----------
        X : FusionResult | np.ndarray
            Fused output or feature matrix.

        Returns
        -------
        np.ndarray
            Probability estimates per class.
        """

        return self._tabular.predict_proba(self._as_matrix(X))

    def save(self, path: str | Path) -> None:
        """
        Persist the underlying classifier to disk with joblib.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """

        self._tabular.save(path)

    @classmethod
    def load(cls, path: str | Path) -> FusionClassifier:
        """
        Reconstruct a FusionClassifier from disk.

        Parameters
        ----------
        path : str | Path
            Source file path.

        Returns
        -------
        FusionClassifier
            Loaded model instance.
        """

        instance = cls()
        instance._tabular = TabularClassifier.load(path)
        instance._fused_dim = None
        return instance

    def _as_matrix(self, X: FusionResult | np.ndarray) -> np.ndarray:
        """Extract a 2D feature matrix from a result or raw array."""
        if isinstance(X, FusionResult):
            self._fused_dim = X.fused.shape[1]
            return X.fused
        matrix = np.asarray(X)
        if matrix.ndim != 2:
            raise InvalidModelInputError(
                f"Expected a 2D feature matrix, got {matrix.ndim} dimensions."
            )
        self._fused_dim = matrix.shape[1]
        return matrix

    def get_parameters(self) -> list[np.ndarray]:
        """
        Return the underlying estimator's weights as NumPy arrays.

        Returns
        -------
        list[np.ndarray]
            Ordered weight arrays (delegated to ``TabularClassifier``).
        """

        return self._tabular.get_parameters()

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """
        Load weights from a list of NumPy arrays.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Ordered weight arrays matching ``get_parameters``.
        """

        self._tabular.set_parameters(parameters)

    def partial_fit(
        self, X: FusionResult | np.ndarray, y: np.ndarray
    ) -> FusionClassifier:
        """
        Continue local training from the current weights (one pass).

        Parameters
        ----------
        X : FusionResult | np.ndarray
            Local fused output or feature matrix.
        y : np.ndarray
            Local target labels.

        Returns
        -------
        FusionClassifier
            Self, updated.
        """

        self._tabular.partial_fit(self._as_matrix(X), np.asarray(y))
        return self


__all__ = ["FusionClassifier"]
