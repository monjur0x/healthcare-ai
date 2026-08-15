"""
Torch MLP tabular model.

``TorchMLPClassifier`` implements the shared :class:`BaseModel` interface
over a small multi-layer perceptron. It exists so the federated pipeline
can train local models under differential privacy (Opacus DP-SGD),
which requires a ``torch.nn.Module`` — the sklearn-based
``TabularClassifier`` cannot be used with Opacus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from models.base import BaseModel
from models.exceptions import (
    InvalidModelInputError,
    ModelLoadError,
    ModelNotFittedError,
)
from preprocessing.logger import get_logger

logger = get_logger(__name__)


class _TorchMLP(torch.nn.Module):
    """ReLU MLP: ``n_features -> hidden -> n_classes``."""

    def __init__(
        self, n_features: int, n_classes: int, hidden_sizes: tuple[int, ...]
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        input_dim = n_features
        for hidden in hidden_sizes:
            layers.append(torch.nn.Linear(input_dim, hidden))
            layers.append(torch.nn.ReLU())
            input_dim = hidden
        layers.append(torch.nn.Linear(input_dim, n_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.net(x)


class TorchMLPClassifier(BaseModel):
    """
    Torch MLP classifier implementing the shared model interface.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_classes : int
        Number of output classes.
    hidden_sizes : tuple[int, ...]
        Hidden-layer widths (default ``(16,)``).
    seed : int
        Reproducibility seed.
    learning_rate : float
        SGD learning rate.
    epochs : int
        Local training epochs per ``partial_fit`` call.
    batch_size : int
        Training batch size.
    device : str
        Torch device (default ``"cpu"``).
    """

    model_name = "torch_mlp"

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        hidden_sizes: tuple[int, ...] = (16,),
        seed: int = 42,
        learning_rate: float = 1e-2,
        epochs: int = 1,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if n_features < 1:
            raise InvalidModelInputError("n_features must be positive.")
        if n_classes < 1:
            raise InvalidModelInputError("n_classes must be positive.")

        self._n_features = int(n_features)
        self._n_classes = int(n_classes)
        self._hidden_sizes = tuple(hidden_sizes)
        self._seed = int(seed)
        self._learning_rate = float(learning_rate)
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._device = device

        self._classes: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._fitted = False

        self._model = _TorchMLP(self._n_features, self._n_classes, self._hidden_sizes)
        self._model.to(self._device)

    @property
    def module(self) -> torch.nn.Module:
        """
        The underlying torch module (used by Opacus DP-SGD).

        Returns
        -------
        torch.nn.Module
            The MLP module.
        """

        return self._model

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained at least once."""
        return self._fitted

    @property
    def classes_(self) -> np.ndarray:
        """Unique target classes seen during training."""
        if self._classes is None:
            return np.array([0, 1], dtype=np.int64)[: self._n_classes]
        return self._classes

    @property
    def feature_names(self) -> list[str] | None:
        """Column names captured during ``fit`` (when given a DataFrame)."""
        return self._feature_names

    def fit(self, X: np.ndarray | Any, y: np.ndarray) -> TorchMLPClassifier:
        """
        Train the MLP on preprocessed tabular data (plain SGD).

        Parameters
        ----------
        X : np.ndarray | Any
            Feature matrix or dataframe.
        y : np.ndarray
            Target labels.

        Returns
        -------
        TorchMLPClassifier
            Self, fitted.

        Raises
        ------
        InvalidModelInputError
            If the inputs are malformed.
        """

        array, self._feature_names = _as_array(X)
        y = np.asarray(y)
        self._validate(array, y)

        self._classes = np.unique(y)
        self._fit_sgd(array, y, epochs=self._epochs)
        self._fitted = True
        logger.info("Fitted TorchMLPClassifier on %d samples", array.shape[0])
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> TorchMLPClassifier:
        """
        Continue local training from the current weights (plain SGD).

        Used by federated clients without differential privacy.

        Parameters
        ----------
        X : np.ndarray
            Local feature matrix.
        y : np.ndarray
            Local target labels.

        Returns
        -------
        TorchMLPClassifier
            Self, updated.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted yet.
        """

        self._require_fitted()
        array, _ = _as_array(X)
        y = np.asarray(y)
        self._validate(array, y)
        self._fit_sgd(array, y, epochs=self._epochs)
        logger.info("Ran one partial-fit round on %d samples", array.shape[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input samples."""
        self._require_fitted()
        array, _ = _as_array(X)
        logits = self._infer(array)
        return self.classes_[torch.argmax(logits, dim=1).cpu().numpy()]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (softmax of logits)."""
        self._require_fitted()
        array, _ = _as_array(X)
        with torch.no_grad():
            logits = self._infer(array)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted model with joblib.

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
            "kind": "torch_mlp",
            "model_name": self.model_name,
            "n_features": self._n_features,
            "n_classes": self._n_classes,
            "hidden_sizes": self._hidden_sizes,
            "seed": self._seed,
            "learning_rate": self._learning_rate,
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "device": self._device,
            "state_dict": self._model.state_dict(),
            "classes": self._classes,
            "feature_names": self._feature_names,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, destination)
        logger.info("Saved TorchMLPClassifier model to %s", destination)

    @classmethod
    def load(cls, path: str | Path) -> TorchMLPClassifier:
        """
        Reconstruct a TorchMLPClassifier from disk.

        Parameters
        ----------
        path : str | Path
            Source file path.

        Returns
        -------
        TorchMLPClassifier
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

        if payload.get("kind") != "torch_mlp":
            raise ModelLoadError(
                f"'{source}' is not a TorchMLPClassifier artifact "
                f"(kind={payload.get('kind')!r})."
            )

        instance = cls(
            n_features=payload["n_features"],
            n_classes=payload["n_classes"],
            hidden_sizes=payload.get("hidden_sizes", (16,)),
            seed=payload.get("seed", 42),
            learning_rate=payload.get("learning_rate", 1e-2),
            epochs=payload.get("epochs", 1),
            batch_size=payload.get("batch_size", 32),
            device=payload.get("device", "cpu"),
        )
        instance._model.load_state_dict(payload["state_dict"])
        instance._classes = payload.get("classes")
        instance._feature_names = payload.get("feature_names")
        instance._fitted = True
        logger.info("Loaded TorchMLPClassifier model from %s", source)
        return instance

    def get_parameters(self) -> list[np.ndarray]:
        """Return the MLP state dict as NumPy arrays."""
        self._require_fitted()
        return [
            value.detach().cpu().numpy() for value in self._model.state_dict().values()
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """
        Load MLP weights from a list of NumPy arrays.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Ordered weight arrays matching ``get_parameters``.

        Raises
        ------
        InvalidModelInputError
            If the parameters are empty or misaligned.
        """

        if not parameters:
            raise InvalidModelInputError("No parameters provided.")
        state = self._model.state_dict()
        if len(parameters) != len(state):
            raise InvalidModelInputError(
                f"Expected {len(state)} parameter arrays, got {len(parameters)}."
            )

        updates: dict[str, torch.Tensor] = {}
        for (name, tensor), value in zip(state.items(), parameters, strict=True):
            value_array = np.asarray(value)
            if value_array.shape != tuple(tensor.shape):
                raise InvalidModelInputError(
                    f"Shape mismatch for '{name}': {value_array.shape} "
                    f"vs {tuple(tensor.shape)}."
                )
            updates[name] = torch.from_numpy(value_array)
        self._model.load_state_dict(updates)
        self._fitted = True
        logger.info("Loaded %d federated weight arrays", len(parameters))

    def _fit_sgd(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        """Run plain SGD training for ``epochs`` passes over the data."""
        label_map = {label: index for index, label in enumerate(self._classes)}
        targets = np.array([label_map[value] for value in y], dtype=np.int64)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
        )
        torch.manual_seed(self._seed)
        generator = torch.Generator().manual_seed(self._seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, generator=generator
        )

        optimizer = torch.optim.SGD(self._model.parameters(), lr=self._learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        self._model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()

    def _infer(self, X: np.ndarray) -> torch.Tensor:
        """Run inference, returning logits on the current device."""
        with torch.no_grad():
            self._model.eval()
            tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
            return self._model(tensor)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ModelNotFittedError(
                "TorchMLPClassifier is not fitted; call fit() first."
            )

    @staticmethod
    def _validate(X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise InvalidModelInputError("X must be 2D with one row per target label.")
        if X.shape[1] < 1:
            raise InvalidModelInputError("X must have at least one feature.")


def _as_array(X: np.ndarray | Any) -> tuple[np.ndarray, list[str] | None]:
    """Convert a DataFrame/array input to a float64 array."""
    if hasattr(X, "to_numpy"):
        names = [str(column) for column in X.columns]
        return np.asarray(X.to_numpy(), dtype=np.float64), names
    return np.asarray(X, dtype=np.float64), None


__all__ = ["TorchMLPClassifier"]
