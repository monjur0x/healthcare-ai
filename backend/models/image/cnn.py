"""
PyTorch CNN classifier for preprocessed image batches.

Consumes the channels-last ``(N, H, W, C)`` float32 batches produced by
the image preprocessing pipeline. No normalization, resizing, or
augmentation is performed here; the model only trains and infers on
already-preprocessed data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from models.base import BaseModel
from models.config import settings
from models.exceptions import (
    InvalidModelInputError,
    ModelLoadError,
    ModelNotFittedError,
)
from preprocessing.logger import get_logger

logger = get_logger(__name__)


def _resolve_device(requested: str | None) -> torch.device:
    """Resolve the requested device, defaulting to CUDA when available."""
    value = settings.IMAGE_DEVICE if requested is None else requested
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _as_channels_first(batch: Any, in_channels: int) -> np.ndarray:
    """Convert a batch to channels-first float32, validating layout."""
    data = np.asarray(batch, dtype=np.float32)
    if data.ndim != 4:
        raise InvalidModelInputError(
            f"Expected a 4D image batch, got {data.ndim} dimensions."
        )
    if data.shape[-1] == in_channels:
        return np.moveaxis(data, -1, 1)
    if data.shape[1] == in_channels:
        return data
    raise InvalidModelInputError(
        f"Image batch has {data.shape[-1]} trailing or {data.shape[1]} "
        f"leading channels; expected {in_channels}."
    )


class _SmallCNN(nn.Module):
    """
    Compact convolutional network used by :class:`ImageClassifier`.

    Uses adaptive average pooling so any spatial input size is accepted.
    """

    def __init__(self, in_channels: int, num_classes: int, base_channels: int) -> None:
        super().__init__()
        self._features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self._classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        images : torch.Tensor
            Channels-first batch of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Raw logits of shape (N, num_classes).
        """

        return self._classifier(self._features(images))


class _TensorDataset(Dataset):
    """In-memory tensor dataset pairing images with integer targets."""

    def __init__(self, images: np.ndarray, targets: np.ndarray) -> None:
        self._images = torch.from_numpy(images)
        self._targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._images[index], self._targets[index]


class ImageClassifier(BaseModel):
    """
    Trainable CNN image classifier.

    Parameters
    ----------
    in_channels : int
        Number of image channels (default 3).
    num_classes : int | None
        Number of output classes. Inferred from ``y`` on first fit when
        not provided.
    epochs : int | None
        Training epochs; defaults to ``settings.IMAGE_TRAIN_EPOCHS``.
    batch_size : int | None
        Training batch size; defaults to ``settings.IMAGE_TRAIN_BATCH_SIZE``.
    learning_rate : float | None
        Optimizer learning rate; defaults to
        ``settings.IMAGE_TRAIN_LEARNING_RATE``.
    device : str | None
        PyTorch device; ``"auto"`` selects CUDA when available.
    torch_seed : int | None
        Seed for deterministic training; defaults to ``settings.RANDOM_SEED``.
    base_channels : int
        Width of the first convolutional layer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        device: str | None = None,
        torch_seed: int | None = None,
        base_channels: int = 16,
    ) -> None:
        if in_channels < 1:
            raise ValueError("in_channels must be a positive integer.")
        if base_channels < 1:
            raise ValueError("base_channels must be a positive integer.")

        self._in_channels = in_channels
        self._num_classes = num_classes
        self._epochs = settings.IMAGE_TRAIN_EPOCHS if epochs is None else int(epochs)
        self._batch_size = (
            settings.IMAGE_TRAIN_BATCH_SIZE if batch_size is None else int(batch_size)
        )
        self._learning_rate = (
            settings.IMAGE_TRAIN_LEARNING_RATE
            if learning_rate is None
            else float(learning_rate)
        )
        self._seed = settings.RANDOM_SEED if torch_seed is None else int(torch_seed)
        self._base_channels = base_channels
        self._device = _resolve_device(device)

        if self._epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if self._batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if self._learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self._model: nn.Module | None = None
        self._classes: np.ndarray | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """True if the model is ready for prediction."""
        return self._fitted

    @property
    def classes_(self) -> np.ndarray:
        """Unique class labels learned during fit."""
        self._require_fitted()
        return self._classes

    @property
    def in_channels(self) -> int:
        """Number of input channels expected by the model."""
        return self._in_channels

    def fit(self, X: Any, y: np.ndarray) -> ImageClassifier:
        """
        Train the CNN on a preprocessed image batch.

        Parameters
        ----------
        X : Any
            Image batch with shape (N, H, W, C) or (N, C, H, W).
        y : np.ndarray
            Target labels.

        Returns
        -------
        ImageClassifier
            Self, fitted.

        Raises
        ------
        InvalidModelInputError
            If the inputs are malformed.
        """

        batch = _as_channels_first(X, self._in_channels)
        y = np.asarray(y)
        self._validate(batch, y)

        classes = np.unique(y)
        self._classes = classes
        num_classes = (
            self._num_classes if self._num_classes is not None else len(classes)
        )
        self._model = _SmallCNN(self._in_channels, num_classes, self._base_channels).to(
            self._device
        )

        label_map = {label: index for index, label in enumerate(classes)}
        targets = np.array([label_map[value] for value in y], dtype=np.int64)
        dataset = _TensorDataset(batch, targets)

        torch.manual_seed(self._seed)
        generator = torch.Generator().manual_seed(self._seed)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            for images, labels in loader:
                images = images.to(self._device)
                labels = labels.to(self._device)
                optimizer.zero_grad()
                output = self._model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            logger.info(
                "Epoch %d/%d: loss=%.4f",
                epoch + 1,
                self._epochs,
                running_loss / len(dataset),
            )

        self._fitted = True
        logger.info(
            "Fitted CNN on %d images across %d classes",
            batch.shape[0],
            len(classes),
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict labels for an image batch.

        Parameters
        ----------
        X : Any
            Image batch with shape (N, H, W, C) or (N, C, H, W).

        Returns
        -------
        np.ndarray
            Predicted labels.
        """

        self._require_fitted()
        probabilities = self.predict_proba(X)
        return np.asarray(self._classes)[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities for an image batch.

        Parameters
        ----------
        X : Any
            Image batch with shape (N, H, W, C) or (N, C, H, W).

        Returns
        -------
        np.ndarray
            Probability estimates per class.
        """

        self._require_fitted()
        batch = _as_channels_first(X, self._in_channels)
        tensor = torch.from_numpy(batch).to(self._device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def save(self, path: str | Path) -> None:
        """
        Persist the fitted model and training metadata to disk.

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
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._model.state_dict(),
            "in_channels": self._in_channels,
            "num_classes": (
                self._num_classes
                if self._num_classes is not None
                else len(self._classes)
            ),
            "classes": self._classes.tolist(),
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "base_channels": self._base_channels,
        }
        torch.save(payload, destination)
        logger.info("Saved CNN model to %s", destination)

    @classmethod
    def load(cls, path: str | Path) -> ImageClassifier:
        """
        Reconstruct an ImageClassifier from disk.

        Parameters
        ----------
        path : str | Path
            Source file path.

        Returns
        -------
        ImageClassifier
            Loaded model instance.

        Raises
        ------
        ModelLoadError
            If the payload cannot be loaded.
        """

        source = Path(path)
        try:
            payload = torch.load(source, weights_only=True, map_location="cpu")
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", source, exc)
            raise ModelLoadError(
                f"Failed to load model from '{source}': {exc}"
            ) from exc

        instance = cls(
            in_channels=payload["in_channels"],
            num_classes=payload["num_classes"],
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            learning_rate=payload["learning_rate"],
            base_channels=payload["base_channels"],
        )
        model = _SmallCNN(
            payload["in_channels"],
            payload["num_classes"],
            payload["base_channels"],
        )
        model.load_state_dict(payload["state_dict"])
        instance._model = model.to(instance._device)
        instance._classes = np.asarray(payload["classes"])
        instance._fitted = True
        logger.info("Loaded CNN model from %s", source)
        return instance

    def _require_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._fitted:
            raise ModelNotFittedError(
                f"{self.__class__.__name__} must be fitted before use."
            )

    def get_parameters(self) -> list[np.ndarray]:
        """
        Return the CNN state dict as NumPy arrays.

        Returns
        -------
        list[np.ndarray]
            Ordered weight arrays matching the state dict.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted.
        """

        self._require_fitted()
        return [
            value.detach().cpu().numpy() for value in self._model.state_dict().values()
        ]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """
        Load CNN weights from a list of NumPy arrays.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Ordered weight arrays matching ``get_parameters``.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted.
        InvalidModelInputError
            If the parameters are empty or misaligned.
        """

        self._require_fitted()
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
        logger.info("Loaded %d federated weight arrays", len(parameters))

    def partial_fit(self, X: Any, y: np.ndarray) -> ImageClassifier:
        """
        Continue local training from the current weights (one pass).

        Used by federated clients to fine-tune the aggregated global
        weights on local image data. Runs one epoch of gradient steps
        that reuse the existing CNN weights; labels must already be
        known from ``fit``.

        Parameters
        ----------
        X : Any
            Image batch with shape (N, H, W, C) or (N, C, H, W).
        y : np.ndarray
            Target labels (a subset of the classes seen at fit time).

        Returns
        -------
        ImageClassifier
            Self, updated.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted.
        InvalidModelInputError
            If the batch is malformed or contains unseen labels.
        """

        self._require_fitted()
        batch = _as_channels_first(X, self._in_channels)
        y = np.asarray(y)
        if batch.shape[0] == 0:
            raise InvalidModelInputError("Image batch is empty.")
        if len(y) != batch.shape[0]:
            raise InvalidModelInputError(
                f"X has {batch.shape[0]} rows but y has {len(y)} labels."
            )

        label_map = {label: index for index, label in enumerate(self._classes)}
        if not {value for value in y}.issubset(label_map):
            raise InvalidModelInputError(
                "partial_fit received labels not present at fit time."
            )
        targets = np.array([label_map[value] for value in y], dtype=np.int64)

        torch.manual_seed(self._seed)
        generator = torch.Generator().manual_seed(self._seed)
        loader = DataLoader(
            _TensorDataset(batch, targets),
            batch_size=self._batch_size,
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for images, labels in loader:
            images = images.to(self._device)
            labels = labels.to(self._device)
            optimizer.zero_grad()
            output = self._model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        self._fitted = True
        logger.info("Ran one partial-fit round on %d images", batch.shape[0])
        return self

    @staticmethod
    def _validate(batch: np.ndarray, y: np.ndarray) -> None:
        """Validate image/label input shapes."""
        if batch.shape[0] == 0:
            raise InvalidModelInputError("Image batch is empty.")
        if len(y) != batch.shape[0]:
            raise InvalidModelInputError(
                f"X has {batch.shape[0]} rows but y has {len(y)} labels."
            )
        if np.unique(y).size < 2:
            raise InvalidModelInputError(
                "Training set must contain at least two classes."
            )


__all__ = ["ImageClassifier"]
