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

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from models.base import BaseModel
from models.config import settings
from models.exceptions import (
    InvalidModelInputError,
    ModelLoadError,
    ModelNotFittedError,
    UnsupportedModelError,
)
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


def _parameter_shapes(model_name: str, parameters: list[np.ndarray]) -> tuple[int, int]:
    """Infer (n_features, n_classes) from a parameter list."""
    if model_name == "mlp":
        n_features = parameters[0].shape[0]
        n_classes = parameters[-1].shape[0]
        return n_features, n_classes

    coefs = parameters[0]
    n_features = coefs.shape[1]
    n_classes = coefs.shape[0] if coefs.shape[0] > 1 else 2
    return n_features, n_classes


def _assign_sklearn_weights(classifier: Any, parameters: list[np.ndarray]) -> None:
    """Inject weights into a fitted sklearn estimator."""
    if isinstance(classifier, MLPClassifier):
        classifier.coefs_ = [
            np.asarray(parameters[index]) for index in range(0, len(parameters), 2)
        ]
        classifier.intercepts_ = [
            np.asarray(parameters[index]) for index in range(1, len(parameters), 2)
        ]
    else:
        classifier.coef_ = np.asarray(parameters[0])
        classifier.intercept_ = np.asarray(parameters[1])


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
        self._scaler_params: dict[str, object] | None = None
        self._encoder_params: dict[str, object] | None = None
        self._imputer_params: dict[str, object] | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """True if the model is ready for prediction."""
        return self._fitted

    @property
    def scaler_params(self) -> dict[str, object] | None:
        """Persisted scaling parameters captured during training (if any)."""
        return self._scaler_params

    def set_scaler_params(self, params: dict[str, object] | None) -> None:
        """
        Attach the preprocessing scaler parameters for inference.

        Parameters
        ----------
        params : dict[str, object] | None
            ``CSVScaler.params()`` output from training, or None.
        """

        self._scaler_params = params

    @property
    def encoder_params(self) -> dict[str, object] | None:
        """Persisted encoding parameters captured during training (if any)."""
        return self._encoder_params

    def set_encoder_params(self, params: dict[str, object] | None) -> None:
        """
        Attach the preprocessing encoder parameters for inference.

        Parameters
        ----------
        params : dict[str, object] | None
            ``CSVEncoder.params()`` output from training, or None.
        """

        self._encoder_params = params

    @property
    def imputer_params(self) -> dict[str, object] | None:
        """Persisted imputation parameters captured during training (if any)."""
        return self._imputer_params

    def set_imputer_params(self, params: dict[str, object] | None) -> None:
        """
        Attach the preprocessing imputer parameters for inference.

        Parameters
        ----------
        params : dict[str, object] | None
            ``CSVImputer.params()`` output from training, or None.
        """

        self._imputer_params = params

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
            "scaler_params": self._scaler_params,
            "encoder_params": self._encoder_params,
            "imputer_params": self._imputer_params,
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
        instance._scaler_params = payload.get("scaler_params")
        instance._encoder_params = payload.get("encoder_params")
        instance._imputer_params = payload.get("imputer_params")
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

    def get_parameters(self) -> list[np.ndarray]:
        """
        Return trainable weights as a list of NumPy arrays.

        Only continuous-weight estimators (logistic / MLP) support
        federated exchange; tree ensembles have no coefficients.

        Returns
        -------
        list[np.ndarray]
            ``coefs_`` followed by ``intercepts_``.

        Raises
        ------
        UnsupportedModelError
            If the estimator has no continuous weights.
        """

        self._require_fitted()
        self._require_exchangeable()
        if isinstance(self._classifier, MLPClassifier):
            return [
                np.asarray(weight, dtype=np.float64)
                for pair in zip(
                    self._classifier.coefs_, self._classifier.intercepts_, strict=True
                )
                for weight in pair
            ]
        return [
            np.asarray(self._classifier.coef_, dtype=np.float64),
            np.asarray(self._classifier.intercept_, dtype=np.float64),
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """
        Load trainable weights from a list of NumPy arrays.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Ordered weight arrays matching ``get_parameters``.

        Raises
        ------
        UnsupportedModelError
            If the estimator has no continuous weights.
        InvalidModelInputError
            If the parameter list is empty or misaligned.
        """

        self._require_exchangeable()
        if not parameters:
            raise InvalidModelInputError("No parameters provided.")
        if len(parameters) % 2 != 0:
            raise InvalidModelInputError(
                "Parameters must alternate coefficient/intercept arrays."
            )

        n_features, n_classes = _parameter_shapes(self._model_name, parameters)

        if not self._fitted:
            n_dummy = max(n_classes, 2)
            estimator = clone(self._classifier)
            estimator.fit(
                np.zeros((n_dummy, n_features)),
                np.arange(n_dummy),
            )
            self._classifier = estimator
        else:
            if n_features != self._classifier.n_features_in_:
                raise InvalidModelInputError(
                    f"Parameter features ({n_features}) do not match fitted "
                    f"features ({self._classifier.n_features_in_})."
                )
            if isinstance(self._classifier, MLPClassifier):
                expected = 2 * len(self._classifier.coefs_)
            else:
                expected = 2
            if len(parameters) != expected:
                raise InvalidModelInputError(
                    f"Expected {expected} parameter arrays, got {len(parameters)}."
                )

        _assign_sklearn_weights(self._classifier, parameters)
        self._classes = np.asarray(self._classifier.classes_)
        self._fitted = True
        logger.info("Loaded %d federated weight arrays", len(parameters))

    def partial_fit(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray
    ) -> TabularClassifier:
        """
        Continue local training from the current weights (one pass).

        Used by federated clients to fine-tune the aggregated global
        weights on local data.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Local feature matrix or dataframe.
        y : np.ndarray
            Local target labels.

        Returns
        -------
        TabularClassifier
            Self, updated.

        Raises
        ------
        UnsupportedModelError
            If the estimator does not support incremental training.
        """

        self._require_fitted()

        if not isinstance(self._classifier, MLPClassifier):
            raise UnsupportedModelError(
                "Only the 'mlp' estimator supports incremental training; "
                "use model_name='mlp' for federated local steps."
            )

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.to_numpy(dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)

        y = np.asarray(y)
        self._validate(X, y)

        self._classifier.partial_fit(X, y, classes=self._classes)
        self._classes = np.asarray(self._classifier.classes_)
        self._fitted = True
        logger.info("Ran one partial-fit round on %d samples", X.shape[0])
        return self

    def _require_exchangeable(self) -> None:
        """Raise when the estimator cannot be exchanged over federation."""
        if self._model_name == "gradient_boosting":
            raise UnsupportedModelError(
                "Gradient boosting has no continuous weights; federated "
                "exchange requires 'logistic' or 'mlp'."
            )
