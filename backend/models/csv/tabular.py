"""
Tabular classification model for structured healthcare data.

Wraps an sklearn classifier so that the shared ``BaseModel`` interface
is satisfied. Consumes the all-numeric output of the CSV preprocessing
pipeline; no feature work is performed inside this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from models.base import BaseModel
from models.exceptions import (
    InvalidModelInputError,
    ModelLoadError,
    ModelNotFittedError,
    UnsupportedModelError,
)
from preprocessing.config import settings
from preprocessing.logger import get_logger

logger = get_logger(__name__)


def _estimator_factory(model_name: str, seed: int) -> Any:
    """Build a fresh sklearn classifier for the given model name."""
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed
        )
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if model_name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed
        )
    raise UnsupportedModelError(
        f"Unsupported model '{model_name}'. Use 'gradient_boosting', "
        "'logistic', or 'mlp'."
    )


class TabularClassifier(BaseModel):
    """
    Wrapper around an sklearn classifier for tabular healthcare data.

    Parameters
    ----------
    model_name : str
        Classifier family: "gradient_boosting" (default), "logistic",
        or "mlp".
    estimator : Any | None
        Optional pre-built sklearn estimator. Overrides ``model_name``.
    random_state : int | None
        Random seed; defaults to ``settings.RANDOM_SEED``.
    """

    def __init__(
        self,
        model_name: str = "gradient_boosting",
        estimator: Any | None = None,
        random_state: int | None = None,
    ) -> None:
        self._model_name = model_name.lower()
        seed = settings.RANDOM_SEED if random_state is None else random_state

        if estimator is not None:
            self._classifier = estimator
        else:
            self._classifier = _estimator_factory(self._model_name, seed)

        self._classes: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """True if the model is ready for prediction."""
        return self._fitted

    @property
    def model_name(self) -> str:
        """Name of the underlying classifier family."""
        return self._model_name

    @property
    def classes_(self) -> np.ndarray:
        """Unique class labels learned during fit."""
        self._require_fitted()
        return self._classes

    @property
    def feature_names(self) -> list[str] | None:
        """Column names captured during fit, if any."""
        return self._feature_names

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> TabularClassifier:
        """
        Fit the classifier on preprocessed features and labels.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Feature matrix or dataframe.
        y : np.ndarray
            Target labels.

        Returns
        -------
        TabularClassifier
            Self, fitted.

        Raises
        ------
        InvalidModelInputError
            If the inputs are malformed.
        """

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.to_numpy(dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)
            self._feature_names = None

        y = np.asarray(y)
        self._validate(X, y)

        self._classifier.fit(X, y)
        self._classes = np.asarray(self._classifier.classes_)
        self._fitted = True
        logger.info(
            "Fitted %s on %d samples, %d features",
            self._model_name,
            X.shape[0],
            X.shape[1],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """

        self._require_fitted()
        X = np.asarray(X)
        return self._classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability estimates per class.
        """

        self._require_fitted()
        X = np.asarray(X)
        return self._classifier.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted model to disk with joblib.

        Parameters
        ----------
        path : str | Path
            Destination file path.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted.
        """

        self._require_fitted()
        payload = {
            "classifier": self._classifier,
            "model_name": self._model_name,
            "classes": self._classes,
            "feature_names": self._feature_names,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, destination)
        logger.info("Saved %s model to %s", self._model_name, destination)

    @classmethod
    def load(cls, path: str | Path) -> TabularClassifier:
        """
        Reconstruct a TabularClassifier from disk.

        Parameters
        ----------
        path : str | Path
            Source file path.

        Returns
        -------
        TabularClassifier
            Loaded model instance.

        Raises
        ------
        ModelLoadError
            If the payload cannot be loaded.
        """

        source = Path(path)
        try:
            payload = joblib.load(source)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", source, exc)
            raise ModelLoadError(
                f"Failed to load model from '{source}': {exc}"
            ) from exc

        instance = cls(model_name=payload.get("model_name", "gradient_boosting"))
        instance._classifier = payload["classifier"]
        instance._classes = payload.get("classes")
        instance._feature_names = payload.get("feature_names")
        instance._fitted = True
        logger.info("Loaded %s model from %s", instance._model_name, source)
        return instance

    def _require_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._fitted:
            raise ModelNotFittedError(
                f"{self.__class__.__name__} must be fitted before use."
            )

    @staticmethod
    def _validate(X: np.ndarray, y: np.ndarray) -> None:
        """Validate feature/label input shapes."""
        if X.ndim != 2:
            raise InvalidModelInputError(
                f"Expected a 2D feature matrix, got {X.ndim} dimensions."
            )
        if X.shape[0] == 0:
            raise InvalidModelInputError("Feature matrix is empty.")
        if len(y) != X.shape[0]:
            raise InvalidModelInputError(
                f"X has {X.shape[0]} rows but y has {len(y)} labels."
            )
