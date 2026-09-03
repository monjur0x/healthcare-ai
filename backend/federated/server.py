"""
Synchronous FedAvg server driver.

Implements the same FedAvg semantics as Flower's ``FedAvg`` strategy
(aggregate client weights, distribute, evaluate) without the Ray-based
``flwr.simulation.run_simulation`` process spawn, so experiments and
tests stay hermetic and deterministic. Weight aggregation is pluggable
and defaults to :func:`federated.parameters.average_weights`.
"""

from __future__ import annotations

import time

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from federated.client import FederatedClient
from federated.metrics import (
    FederatedMetrics,
    convergence_round,
    parameter_set_bytes,
    round_accuracy_deltas,
)
from federated.parameters import average_weights, scale_updates
from federated.privacy import SecureAggregator
from preprocessing.logger import get_logger

logger = get_logger(__name__)

EvaluateFn = Callable[[list[np.ndarray]], dict[str, float]]
AggregateFn = Callable[[Sequence[list[np.ndarray]]], list[np.ndarray]]


@dataclass(frozen=True)
class RoundResult:
    """
    Global evaluation outcome for one FedAvg round.

    Attributes
    ----------
    round_index : int
        1-based round number.
    n_clients : int
        Number of participating clients.
    accuracy : float
        Global accuracy after this round.
    log_loss : float | None
        Global log loss, when available.
    roc_auc : float | None
        Global ROC-AUC, when available.
    round_duration_s : float | None
        Wall-clock duration of the round.
    bytes_exchanged : int | None
        Estimated bytes moved during the round.
    """

    round_index: int
    n_clients: int
    accuracy: float
    log_loss: float | None = None
    roc_auc: float | None = None
    round_duration_s: float | None = None
    bytes_exchanged: int | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        """
        Serialize the round result to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, float | int | None]
            Round metadata and metrics keyed by name.
        """

        return {
            "round_index": self.round_index,
            "n_clients": self.n_clients,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "roc_auc": self.roc_auc,
            "round_duration_s": self.round_duration_s,
            "bytes_exchanged": self.bytes_exchanged,
        }


class FedAvgServer:
    """
    Orchestrate federated rounds over a set of clients.

    Parameters
    ----------
    clients : Sequence[FederatedClient]
        Local clients participating in federation.
    aggregate_fn : AggregateFn
        Weight aggregation function; defaults to ``average_weights``.
    num_rounds : int
        Number of FedAvg rounds to run.
    evaluate_fn : EvaluateFn | None
        Optional global evaluation of the aggregated weights. When
        omitted, per-client evaluations are combined (loss- and
        count-weighted) into a global estimate.
    secure_aggregation : bool
        When enabled, client updates are masked with the pairwise
        one-time-pad :class:`federated.privacy.SecureAggregator` before
        aggregation so the server never observes any single update.
        Updates are pre-scaled by sample share, so both paths produce
        the same count-weighted FedAvg mean.
    """

    def __init__(
        self,
        clients: Sequence[FederatedClient],
        aggregate_fn: AggregateFn = average_weights,
        num_rounds: int = 3,
        evaluate_fn: EvaluateFn | None = None,
        secure_aggregation: bool = False,
    ) -> None:
        if not clients:
            raise ValueError("FedAvgServer requires at least one client.")
        if num_rounds < 1:
            raise ValueError("num_rounds must be a positive integer.")

        self._clients = list(clients)
        self._aggregate_fn = aggregate_fn
        self._num_rounds = num_rounds
        self._evaluate_fn = evaluate_fn
        self._secure_aggregation = secure_aggregation
        self._aggregator = (
            SecureAggregator(len(self._clients)) if secure_aggregation else None
        )
        # Per-round worst-case client epsilon (basic-composition input for
        # cumulative_epsilon_upper_bound). See run().
        self._epsilons: list[float] = []
        self._payload_inspection: dict[str, Any] | None = None
        self._global_parameters: list[np.ndarray] | None = None
        self._history: list[RoundResult] = []
        self._round_durations: list[float] = []
        self._round_bytes: list[int] = []

    @property
    def global_parameters(self) -> list[np.ndarray] | None:
        """Most recently aggregated global weights."""
        return self._global_parameters

    @property
    def max_epsilon(self) -> float | None:
        """Worst-case per-client epsilon across all rounds."""
        return max(self._epsilons) if self._epsilons else None

    @property
    def per_round_epsilons(self) -> list[float]:
        """Per-round epsilon values (worst-case client per round)."""
        return list(self._epsilons)

    @property
    def cumulative_epsilon_upper_bound(self) -> float | None:
        """
        Upper bound on cumulative epsilon across all rounds.

        Uses basic composition (sum of per-round epsilons). This is a
        conservative upper bound, NOT tight RDP composition. See
        ``compute_cumulative_epsilon_upper_bound`` in privacy.py.
        """
        from federated.privacy import compute_cumulative_epsilon_upper_bound

        if not self._epsilons:
            return None
        return compute_cumulative_epsilon_upper_bound(self._epsilons)

    @property
    def payload_inspection(self) -> dict[str, Any] | None:
        """Evidence dict from payload inspection (leakage measurement)."""
        return self._payload_inspection

    @property
    def history(self) -> tuple[RoundResult, ...]:
        """Per-round global evaluation results."""
        return tuple(self._history)

    @property
    def metrics(self) -> FederatedMetrics:
        """
        Cost, convergence, and timing statistics for the run.

        Returns
        -------
        FederatedMetrics
            Aggregate cost, convergence, and timing statistics for the
            completed run.

        Raises
        ------
        RuntimeError
            If the server has not been run yet.
        """

        if not self._history:
            raise RuntimeError("FedAvgServer must be run before metrics are available.")
        accuracies = [result.accuracy for result in self._history]
        return FederatedMetrics(
            n_rounds=len(self._history),
            n_clients=len(self._clients),
            total_bytes_exchanged=sum(self._round_bytes),
            bytes_exchanged_per_round=(
                self._round_bytes[0] if self._round_bytes else 0
            ),
            round_times_s=tuple(self._round_durations),
            total_time_s=sum(self._round_durations),
            accuracy_deltas=round_accuracy_deltas(accuracies),
            convergence_round=convergence_round(accuracies),
            secure_aggregation=self._secure_aggregation,
            differential_privacy=any(
                bool(getattr(client, "_privacy", None) and client._privacy.enabled)
                for client in self._clients
            ),
            epsilon=self.max_epsilon,
        )

    def run(self) -> FedAvgServer:
        """
        Execute the federated training rounds.

        Returns
        -------
        FedAvgServer
            Self, with updated global weights and round history.
        """

        logger.info("Initializing global weights from %d clients", len(self._clients))
        initial = self._aggregate_fn(
            [client.get_parameters({}) for client in self._clients]
        )
        self._global_parameters = initial

        for round_index in range(1, self._num_rounds + 1):
            round_start = time.perf_counter()
            logger.info("Starting federated round %d", round_index)
            updated, counts, fit_metrics = zip(
                *[client.fit(self._global_parameters, {}) for client in self._clients],
                strict=True,
            )
            # Worst case across this round's clients: the basic-composition
            # input. Extending with every client's value would overcount
            # the budget by the number of clients.
            round_epsilons = [
                float(item["epsilon"]) for item in fit_metrics if "epsilon" in item
            ]
            if round_epsilons:
                self._epsilons.append(max(round_epsilons))
            sample_counts = [float(count) for count in counts]
            if self._aggregator is not None:
                # Pre-scale by sample share: masks added to pre-scaled
                # updates still cancel exactly, so the masked result is
                # the count-weighted mean (no secure/non-secure flip).
                scaled = scale_updates(list(updated), sample_counts)
                self._global_parameters = self._aggregator.aggregate(
                    scaled,
                    [1.0] * len(scaled),
                    average=False,
                    round_number=round_index,
                )
            elif self._aggregate_fn is average_weights:
                self._global_parameters = average_weights(list(updated), sample_counts)
            else:
                self._global_parameters = self._aggregate_fn(updated)
            round_duration_s = time.perf_counter() - round_start
            bytes_exchanged = (
                2 * len(self._clients) * parameter_set_bytes(self._global_parameters)
            )
            self._round_durations.append(round_duration_s)
            self._round_bytes.append(bytes_exchanged)
            self._history.append(
                self._evaluate_round(round_index, round_duration_s, bytes_exchanged)
            )

        # ── Measure data leakage from the last round's actual payloads ──
        from federated.privacy import inspect_federation_payloads

        try:
            feature_names: list[str] | None = None
            first_client = self._clients[0] if self._clients else None
            if first_client and hasattr(first_client, "_X_train"):
                cols = getattr(first_client._X_train, "columns", None)
                if cols is not None:
                    feature_names = [str(c) for c in cols]

            # Use the last round's raw (pre-aggregation) updates
            if updated:
                self._payload_inspection = inspect_federation_payloads(
                    updated, feature_names
                )
                logger.info(
                    "Payload inspection: leakage_rate=%.4f, %d payloads, %d bytes",
                    self._payload_inspection["leakage_rate"],
                    self._payload_inspection["num_payloads_inspected"],
                    self._payload_inspection["total_payload_bytes"],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Payload inspection failed (non-fatal): %s", exc)

        return self

    def _evaluate_round(
        self,
        round_index: int,
        round_duration_s: float,
        bytes_exchanged: int,
    ) -> RoundResult:
        """Evaluate the aggregated weights for a round."""
        if self._evaluate_fn is not None:
            metrics = self._evaluate_fn(self._global_parameters)
            return RoundResult(
                round_index=round_index,
                n_clients=len(self._clients),
                accuracy=metrics.get("accuracy", 0.0),
                log_loss=metrics.get("log_loss"),
                roc_auc=metrics.get("roc_auc"),
                round_duration_s=round_duration_s,
                bytes_exchanged=bytes_exchanged,
            )

        losses, counts, metric_dicts = zip(
            *[client.evaluate(self._global_parameters, {}) for client in self._clients],
            strict=True,
        )
        total = sum(counts)
        accuracy = (
            sum(
                count * metrics["accuracy"]
                for count, metrics in zip(counts, metric_dicts, strict=True)
            )
            / total
        )
        log_loss = (
            sum(count * loss for count, loss in zip(counts, losses, strict=True))
            / total
        )
        logger.info(
            "Round %d: global accuracy=%.4f log_loss=%.4f",
            round_index,
            accuracy,
            log_loss,
        )
        return RoundResult(
            round_index=round_index,
            n_clients=len(self._clients),
            accuracy=float(accuracy),
            log_loss=float(log_loss),
            round_duration_s=round_duration_s,
            bytes_exchanged=bytes_exchanged,
        )


def make_global_evaluator(
    model_factory: Callable[[], object],
    X: object,
    y_true: np.ndarray,
) -> EvaluateFn:
    """
    Build a global evaluator that scores the aggregated weights on a
    central hold-out dataset.

    Parameters
    ----------
    model_factory : Callable[[], object]
        Callable returning a fresh model with ``set_parameters``,
        ``predict``, and ``predict_proba``.
    X : object
        Central validation features.
    y_true : np.ndarray
        Central validation labels.

    Returns
    -------
    EvaluateFn
        Function mapping global weights to an accuracy / log-loss /
        ROC-AUC metrics dictionary.
    """

    from evaluation import evaluate_classifier

    def evaluate(parameters: list[np.ndarray]) -> dict[str, float]:
        model = model_factory()
        model.set_parameters(parameters)
        metrics = evaluate_classifier(model, X, y_true)
        return {
            "accuracy": metrics.accuracy,
            "log_loss": (
                metrics.log_loss_value if metrics.log_loss_value is not None else 0.0
            ),
            "roc_auc": metrics.roc_auc if metrics.roc_auc is not None else 0.0,
        }

    return evaluate


__all__ = ["FedAvgServer", "RoundResult", "make_global_evaluator"]
