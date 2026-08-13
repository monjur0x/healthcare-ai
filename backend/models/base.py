"""
Shared model interface for healthcare prediction models.

Every model in this package exposes a common sklearn-like interface
(fit / predict / predict_proba / save / load) so that Flower, FastAPI,
and CrewAI can consume models uniformly. Models operate on NumPy arrays
or pandas DataFrames of already-preprocessed data; no preprocessing is
performed inside models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """
    Abstract interface implemented by all prediction models.

    Methods
    -------
    fit(X, y)
        Fit the model on preprocessed features and labels.
    predict(X)
        Return predicted labels.
    predict_proba(X)
        Return class probability estimates (classification models).
    save(path)
        Persist the fitted model.
    load(path)
        Reconstruct a model from disk.
    """

    @abstractmethod
    def fit(self, X: np.ndarray | object, y: np.ndarray) -> BaseModel:
        """
        Fit the model on preprocessed input data.

        Parameters
        ----------
        X : np.ndarray | object
            Feature matrix or dataframe.
        y : np.ndarray
            Target labels.

        Returns
        -------
        BaseModel
            Self, fitted.
        """

    @abstractmethod
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

        Raises
        ------
        NotImplementedError
            If the model does not support probability estimates.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba."
        )

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Persist the fitted model to disk.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseModel:
        """
        Reconstruct a model from disk.

        Parameters
        ----------
        path : str | Path
            Source file path.

        Returns
        -------
        BaseModel
            Loaded model instance.
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """
        Whether the model has been fitted.

        Returns
        -------
        bool
            True if the model is ready for prediction.
        """
