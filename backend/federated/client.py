"""
Flower federated client for the healthcare prediction models.

Wraps a :class:`BaseModel` (tabular, fusion, or image) so its weights can
be trained locally and exchanged with a Flower server. The client owns
its local data; every round it rebuilds a model, applies the aggregated
global weights, runs one local partial-fit pass, and returns the updated
weights. All training and evaluation is delegated to the model and the
:mod:`evaluation` package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from flwr.client import NumPyClient

from evaluation import evaluate_classifier
from federated.privacy import PrivacyConfig, train_with_differential_privacy
from models.base import BaseModel

ModelFactory = Callable[[], BaseModel]

#: Prefix Opacus uses for wrapped-module keys in ``state_dict`` output.
_OPACUS_STATE_PREFIX = "_module."


def _apply_trained_weights(module: Any, trained_module: Any) -> None:
    """
    Copy DP-trained weights back into the model's own module.

    ``train_with_differential_privacy`` trains the Opacus-wrapped copy
    of the module, so the returned weights must be synced back before
    the client reads parameters from its model. Opacus prefixes wrapped
    keys with ``_module.``; that prefix is stripped when the direct
    load fails. A persistent mismatch raises instead of silently
    sharing stale weights.

    Parameters
    ----------
    module : Any
        The model's underlying torch module to update in place.
    trained_module : Any
        The trained (possibly Opacus-wrapped) module to copy from.
    """

    if trained_module is module:
        return
    try:
        module.load_state_dict(trained_module.state_dict())
        return
    except (RuntimeError, ValueError):
        pass
    state = trained_module.state_dict()
    stripped = {
        key[len(_OPACUS_STATE_PREFIX) :]
        if key.startswith(_OPACUS_STATE_PREFIX)
        else key: value
        for key, value in state.items()
    }
    module.load_state_dict(stripped)


class FederatedClient(NumPyClient):
    """
    Flower client federating a single model over local data.

    Parameters
    ----------
    model_factory : ModelFactory
        Callable returning a fresh model instance. If the instance is
        not fitted, it is warm-started on the local training data so
        weight structure and classes are available for exchange.
    X_train : Any
        Local training features (matrix, image batch, or fused result).
    y_train : np.ndarray
        Local training labels.
    X_val : Any
        Local validation features.
    y_val : np.ndarray
        Local validation labels.
    privacy : PrivacyConfig | None
        When enabled and the model exposes a torch ``module``, local
        training uses Opacus DP-SGD and the per-round ``fit`` metrics
        report the realized epsilon.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any,
        y_val: np.ndarray,
        privacy: PrivacyConfig | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._X_train = X_train
        self._y_train = np.asarray(y_train)
        self._X_val = X_val
        self._y_val = np.asarray(y_val)
        self._privacy = privacy or PrivacyConfig(enabled=False)

    def _build(self) -> BaseModel:
        """Construct and warm-start a fresh model instance."""
        model = self._model_factory()
        if not model.is_fitted:
            model.fit(self._X_train, self._y_train)
        return model

    def _train_locally(
        self, model: BaseModel
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        """
        Run one local training step, honoring DP-SGD when configured.

        Parameters
        ----------
        model : BaseModel
            Model carrying the aggregated global weights.

        Returns
        -------
        tuple[list[np.ndarray], dict[str, Any]]
            Updated weights and per-fit metrics (including ``epsilon``
            when differential privacy is active).
        """

        if not self._privacy.enabled:
            model.partial_fit(self._X_train, self._y_train)
            return model.get_parameters(), {}

        module = getattr(model, "module", None)
        if module is None:
            raise RuntimeError(
                "Differential privacy requires a torch-backed model "
                "(e.g. TorchMLPClassifier) exposing a 'module'."
            )
        trained_module, epsilon = train_with_differential_privacy(
            module, self._X_train, self._y_train, self._privacy
        )
        _apply_trained_weights(module, trained_module)
        return model.get_parameters(), {"epsilon": float(epsilon)}

    def get_parameters(self, config: dict[str, Any] | None = None) -> list[np.ndarray]:
        """
        Return the current model weights as NumPy arrays.

        Parameters
        ----------
        config : dict[str, Any] | None
            Flower client config (unused here).

        Returns
        -------
        list[np.ndarray]
            Ordered weight arrays.
        """

        return self._build().get_parameters()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """
        Apply global weights, run one local training pass, and return
        the updated weights.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Aggregated global weights.
        config : dict[str, Any] | None
            Flower client config (unused here).

        Returns
        -------
        tuple[list[np.ndarray], int, dict[str, Any]]
            Updated weights, number of local samples, and metrics
            (``epsilon`` when differential privacy is active).
        """

        model = self._build()
        model.set_parameters(parameters)
        updated, metrics = self._train_locally(model)
        return updated, len(self._X_train), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any] | None = None,
    ) -> tuple[float, int, dict[str, Any]]:
        """
        Evaluate the global weights on local validation data.

        Parameters
        ----------
        parameters : list[np.ndarray]
            Global weights to evaluate.
        config : dict[str, Any] | None
            Flower client config (unused here).

        Returns
        -------
        tuple[float, int, dict[str, Any]]
            Loss, number of validation samples, and accuracy.
        """

        model = self._build()
        model.set_parameters(parameters)
        metrics = evaluate_classifier(model, self._X_val, self._y_val)

        loss = (
            metrics.log_loss_value
            if metrics.log_loss_value is not None
            else float(1.0 - metrics.accuracy)
        )
        return loss, len(self._X_val), {"accuracy": metrics.accuracy}


__all__ = ["FederatedClient", "ModelFactory"]
