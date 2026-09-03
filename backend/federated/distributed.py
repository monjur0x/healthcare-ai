"""
Distributed federated learning over Flower gRPC.

Implements the deployment path that runs hospitals as *separate
processes* connected to a real Flower server over gRPC (as opposed to the
hermetic in-process :class:`federated.server.FedAvgServer` used by the
API's quick-training path):

* :class:`DistributedFedAvg` — a FedAvg strategy that keeps the pairwise
  one-time-pad :class:`federated.privacy.SecureAggregator` semantics when
  secure aggregation is enabled, records per-round metrics into the
  :class:`federated.registry.ModelRegistry`, and retains the final global
  weights so they can be persisted.
* :func:`run_distributed_server` — starts the Flower gRPC server, waits
  for the configured number of hospitals to connect, and on completion
  saves the global model artifact and registers it.
* :func:`run_hospital_client` — loads one hospital's *own* local data,
  preprocesses it locally, and connects to the server as a NumPy client.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flwr.client import start_numpy_client
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg

from evaluation import evaluate_classifier
from federated.client import FederatedClient
from federated.hospitals import HospitalConfig, load_hospital_dataset
from federated.parameters import scale_updates
from federated.privacy import PrivacyConfig, SecureAggregator
from federated.registry import ModelRegistry
from models import BaseModel, TabularClassifier, TorchMLPClassifier
from preprocessing.logger import get_logger

logger = get_logger(__name__)


def _load_certificates(
    tls_enabled: bool,
    ca_cert: str | None,
    server_cert: str | None,
    server_key: str | None,
) -> tuple[bytes, bytes, bytes] | None:
    """
    Load TLS certificates for Flower gRPC server.

    Returns a tuple of (ca_cert, server_cert, server_key) as bytes,
    or None if TLS is disabled.
    """
    if not tls_enabled:
        return None
    if not ca_cert or not server_cert or not server_key:
        raise ValueError(
            "TLS enabled but certificate paths missing: "
            "ca_cert, server_cert, and server_key are required."
        )
    ca = Path(ca_cert).read_bytes()
    cert = Path(server_cert).read_bytes()
    key = Path(server_key).read_bytes()
    return ca, cert, key


def _load_client_certificates(
    tls_enabled: bool,
    ca_cert: str | None,
    client_cert: str | None,
    client_key: str | None,
) -> bytes | tuple[bytes, bytes, bytes] | None:
    """
    Load TLS certificates for Flower gRPC client.

    Returns:
    - None if TLS is disabled
    - CA cert bytes only (server verification only)
    - Tuple of (ca_cert, client_cert, client_key) for mutual TLS
    """
    if not tls_enabled:
        return None
    if not ca_cert:
        raise ValueError("TLS enabled but ca_cert path is required.")
    ca = Path(ca_cert).read_bytes()
    if client_cert and client_key:
        cert = Path(client_cert).read_bytes()
        key = Path(client_key).read_bytes()
        return ca, cert, key
    return ca


@dataclass(frozen=True)
class ModelSpec:
    """
    Shared model architecture agreed between the server and clients.

    Attributes
    ----------
    n_features : int
        Number of input features after local preprocessing.
    n_classes : int
        Number of output classes.
    feature_names : tuple[str, ...]
        Canonical ordered feature names derived from the full source
        dataset; every participant aligns its local features to this
        schema so the imputer cannot cause per-slice shape drift.
    differential_privacy : bool
        Whether clients apply Opacus DP-SGD (selects the torch backend).
    seed : int
        Reproducibility seed for model construction.
    """

    n_features: int
    n_classes: int
    feature_names: tuple[str, ...] = ()
    differential_privacy: bool = False
    seed: int = 42

    def make_model(self) -> BaseModel:
        """
        Build a model instance matching the agreed architecture.

        Returns
        -------
        BaseModel
            A ``TorchMLPClassifier`` when differential privacy is active,
            otherwise a ``TabularClassifier`` (``"mlp"`` family).
        """

        if self.differential_privacy:
            return TorchMLPClassifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
                seed=self.seed,
            )
        return TabularClassifier(model_name="mlp", random_state=self.seed)

    def align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder and/or zero-fill ``features`` to the canonical schema.

        Each participant preprocesses its own slice, so the imputer may
        have dropped different columns than the full dataset. Aligning by
        the shared ``feature_names`` guarantees an identical matrix shape
        everywhere while keeping column semantics consistent.

        Parameters
        ----------
        features : pd.DataFrame
            Locally preprocessed feature frame.

        Returns
        -------
        pd.DataFrame
            Frame with exactly the canonical feature columns in order.
        """

        if not self.feature_names:
            return features
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            logger.warning(
                "Aligning local features: filling %d missing columns with 0",
                len(missing),
            )
        return features.reindex(columns=self.feature_names).fillna(0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model spec for CLI passing."""
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "feature_names": list(self.feature_names),
            "differential_privacy": self.differential_privacy,
            "seed": self.seed,
        }


class DistributedFedAvg(FedAvg):
    """
    FedAvg strategy for the distributed deployment.

    Differs from plain :class:`flwr.server.strategy.FedAvg` in three ways:

    1. When ``secure_aggregation`` is enabled, ``aggregate_fit`` masks the
       client updates with the pairwise one-time-pad
       :class:`SecureAggregator` so the server only ever forms the exact
       mean without observing any single update. Updates are pre-scaled
       by sample share, so both paths produce the same count-weighted
       FedAvg mean (matching the in-process server).
    2. Per-round global metrics are written to the registry.
    3. The final aggregated weights are retained as ``global_parameters``
       so the server can persist the global model after the run.

    Parameters
    ----------
    num_clients : int
        Expected number of hospital clients.
    registry : ModelRegistry
        Registry receiving per-round metrics.
    run_id : str
        Registry run id this strategy reports into.
    secure_aggregation : bool
        Mask client updates with the pairwise OTP aggregator.
    seed : int
        Seed for the secure aggregator.
    min_available : int | None
        Override for the number of clients that must be available before
        training starts (defaults to ``num_clients``).
    """

    def __init__(
        self,
        num_clients: int,
        registry: ModelRegistry,
        run_id: str,
        secure_aggregation: bool = False,
        seed: int = 42,
        min_available: int | None = None,
    ) -> None:
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=min_available or num_clients,
        )
        self._num_clients = num_clients
        self._registry = registry
        self._run_id = run_id
        self._secure_aggregation = secure_aggregation
        self._aggregator = (
            SecureAggregator(num_clients, seed) if secure_aggregation else None
        )
        self._seed = seed
        self.global_parameters: list[np.ndarray] | None = None
        self._epsilons: list[float] = []
        self._last_accuracy: float | None = None

    @property
    def max_epsilon(self) -> float | None:
        """Worst-case per-client DP epsilon reported across all rounds."""
        return max(self._epsilons) if self._epsilons else None

    @property
    def last_accuracy(self) -> float | None:
        """Most recent count-weighted client accuracy, if any."""
        return self._last_accuracy

    def aggregate_fit(
        self,
        server_round: int,
        results: Sequence[Any],
        failures: Sequence[Any],
    ) -> tuple[Any, dict[str, Any]]:
        """
        Aggregate client updates, optionally masked, and persist metrics.

        Parameters
        ----------
        server_round : int
            Current round number.
        results : Sequence[Any]
            Per-client fit results.
        failures : Sequence[Any]
            Per-client failures.

        Returns
        -------
        tuple[Any, dict[str, Any]]
            Aggregated parameters and fit metrics.
        """

        if not results:
            return None, {}

        updates = [parameters_to_ndarrays(result.parameters) for _, result in results]
        weights = [float(result.num_examples) for _, result in results]

        for _, result in results:
            epsilon = result.metrics.get("epsilon")
            if epsilon is not None:
                self._epsilons.append(float(epsilon))

        if self._aggregator is not None:
            # Pre-scale by sample share (masks still cancel exactly), so
            # the secure path matches the count-weighted non-secure path.
            scaled = scale_updates(updates, weights)
            aggregated = self._aggregator.aggregate(
                scaled, [1.0] * len(scaled), average=False
            )
        else:
            aggregated = self._aggregate_weighted(updates, weights)

        self.global_parameters = aggregated
        return ndarrays_to_parameters(aggregated), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: Sequence[Any],
        failures: Sequence[Any],
    ) -> tuple[float | None, dict[str, Any]]:
        """
        Combine client evaluations into a count-weighted global accuracy.

        Parameters
        ----------
        server_round : int
            Current round number.
        results : Sequence[Any]
            Per-client evaluation results.
        failures : Sequence[Any]
            Per-client evaluation failures.

        Returns
        -------
        tuple[float | None, dict[str, Any]]
            Count-weighted loss and aggregated metrics.
        """

        if not results:
            return None, {}

        total = sum(float(result.num_examples) for _, result in results)
        loss = (
            sum(
                float(result.loss) * float(result.num_examples) for _, result in results
            )
            / total
        )
        accuracy = (
            sum(
                float(result.metrics.get("accuracy", 0.0)) * float(result.num_examples)
                for _, result in results
            )
            / total
        )
        self._registry.record_round(
            run_id=self._run_id,
            round_index=server_round,
            accuracy=accuracy,
            log_loss=float(loss),
            n_clients=len(results),
            bytes_exchanged=int(2 * len(results) * self._parameter_bytes()),
            duration_s=0.0,
        )
        self._last_accuracy = accuracy
        logger.info(
            "Round %d: global accuracy=%.4f log_loss=%.4f over %d clients",
            server_round,
            accuracy,
            loss,
            len(results),
        )
        return float(loss), {"accuracy": accuracy}

    def _aggregate_weighted(
        self, updates: list[list[np.ndarray]], weights: list[float]
    ) -> list[np.ndarray]:
        """Count-weighted element-wise mean of client updates."""
        total = sum(weights)
        reference = updates[0]
        return [
            np.sum(
                np.stack(
                    [
                        np.asarray(update[position], dtype=np.float64)
                        * (weight / total)
                        for update, weight in zip(updates, weights, strict=True)
                    ]
                ),
                axis=0,
            )
            for position in range(len(reference))
        ]

    def _parameter_bytes(self) -> int:
        """Estimated bytes of one weight exchange."""
        if self.global_parameters is None:
            return 0
        return int(
            sum(
                int(np.asarray(parameter).nbytes)
                for parameter in self.global_parameters
            )
        )


def run_distributed_server(
    address: str,
    num_rounds: int,
    num_clients: int,
    model_spec: ModelSpec,
    registry: ModelRegistry,
    preset: str,
    secure_aggregation: bool = False,
    holdout: tuple[Any, np.ndarray] | None = None,
    artifacts_dir: str | Path = "artifacts",
    min_available: int | None = None,
    tls_enabled: bool = False,
    tls_ca_cert: str | None = None,
    tls_server_cert: str | None = None,
    tls_server_key: str | None = None,
) -> tuple[str, str, int | None]:
    """
    Start the Flower gRPC server and persist the global model.

    Blocks until all ``num_rounds`` rounds complete. On completion the
    aggregated global weights are saved as a ``TorchMLPClassifier`` /
    ``TabularClassifier`` artifact under ``artifacts_dir/<preset>/`` and
    registered in the model registry.

    Parameters
    ----------
    address : str
        Server address (e.g. ``"0.0.0.0:8080"``).
    num_rounds : int
        Number of federated rounds.
    num_clients : int
        Number of hospital clients that must connect.
    model_spec : ModelSpec
        Shared model architecture.
    registry : ModelRegistry
        Registry receiving the run.
    preset : str
        Dataset preset being federated.
    secure_aggregation : bool
        Mask client updates with the pairwise OTP aggregator.
    holdout : tuple[Any, np.ndarray] | None
        Optional central hold-out ``(features, labels)`` evaluated after
        training on the final global model.
    artifacts_dir : str | Path
        Root directory for model artifacts.
    min_available : int | None
        Number of clients required before training starts.

    Returns
    -------
    tuple[str, str, int | None]
        ``(run_id, model_path, version)`` for the completed run.

    Raises
    ------
    RuntimeError
        If the server fails to produce a global model.
    """

    run_id = registry.start_run(
        preset=preset,
        n_hospitals=num_clients,
        n_rounds=num_rounds,
        secure_aggregation=secure_aggregation,
        differential_privacy=model_spec.differential_privacy,
    )
    logger.info(
        "Starting distributed server on %s for preset '%s' (%d clients, %d rounds)",
        address,
        preset,
        num_clients,
        num_rounds,
    )

    strategy = DistributedFedAvg(
        num_clients=num_clients,
        registry=registry,
        run_id=run_id,
        secure_aggregation=secure_aggregation,
        seed=model_spec.seed,
        min_available=min_available,
    )

    certs = _load_certificates(
        tls_enabled, tls_ca_cert, tls_server_cert, tls_server_key
    )
    start_server(
        server_address=address,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        certificates=certs,
    )

    if strategy.global_parameters is None:
        raise RuntimeError("Server finished without a global model.")

    global_model = model_spec.make_model()
    global_model.set_parameters(strategy.global_parameters)
    if getattr(global_model, "_feature_names", "missing") != "missing":
        global_model._feature_names = (
            list(model_spec.feature_names) if model_spec.feature_names else None
        )

    accuracy: float | None = None
    roc_auc: float | None = None
    if holdout is not None:
        holdout_x, holdout_y = holdout
        if int(holdout_x.shape[1]) == model_spec.n_features:
            try:
                metrics = evaluate_classifier(global_model, holdout_x, holdout_y)
                accuracy = metrics.accuracy
                roc_auc = metrics.roc_auc if metrics.roc_auc is not None else None
            except (ValueError, TypeError, RuntimeError) as error:
                logger.warning(
                    "Hold-out evaluation skipped (shape mismatch): %s", error
                )
                accuracy = strategy.last_accuracy
        else:
            logger.warning(
                "Hold-out has %d features but model expects %d; using "
                "client-aggregated accuracy",
                int(holdout_x.shape[1]),
                model_spec.n_features,
            )
            accuracy = strategy.last_accuracy
    elif strategy.last_accuracy is not None:
        accuracy = strategy.last_accuracy

    out_dir = Path(artifacts_dir) / preset
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "global_model.joblib"
    global_model.save(model_path)

    version = registry.register_model(
        run_id=run_id,
        preset=preset,
        model_path=model_path,
        accuracy=accuracy,
        roc_auc=roc_auc,
        epsilon=strategy.max_epsilon,
    )
    registry.complete_run(run_id)
    logger.info(
        "Distributed run %s complete: accuracy=%s artifact=%s version=%d",
        run_id,
        accuracy,
        model_path,
        version,
    )
    return run_id, str(model_path), version


def run_hospital_client(
    address: str,
    hospital: HospitalConfig,
    model_spec: ModelSpec,
    max_rows: int | None = None,
    privacy: PrivacyConfig | None = None,
    tls_enabled: bool = False,
    tls_ca_cert: str | None = None,
    tls_client_cert: str | None = None,
    tls_client_key: str | None = None,
    heterogeneous: bool = False,
) -> None:
    """
    Connect one hospital to the server as a Flower NumPy client.

    The hospital loads and preprocesses its *own* local dataset (its raw
    rows never leave the process), builds a :class:`FederatedClient`, and
    participates until the server finishes.

    Parameters
    ----------
    address : str
        Server address (e.g. ``"127.0.0.1:8080"``).
    hospital : HospitalConfig
        The hospital site to connect.
    model_spec : ModelSpec
        Shared model architecture.
    max_rows : int | None
        Optional cap on local rows.
    privacy : PrivacyConfig | None
        Local differential-privacy configuration.
    tls_enabled : bool
        Use TLS for the gRPC connection.
    tls_ca_cert : str | None
        CA certificate PEM path (TLS mode).
    tls_client_cert : str | None
        Client certificate PEM path (mutual TLS).
    tls_client_key : str | None
        Client key PEM path (mutual TLS).
    heterogeneous : bool
        When true the local CSV holds this hospital's own specialty
        dataset; it is mapped onto the shared canonical schema
        (:func:`federated.canonical.load_canonical_frame`) instead of the
        single-preset loader.
    """

    if heterogeneous:
        from federated.canonical import HOSPITAL_PRESETS, load_canonical_frame

        preset = HOSPITAL_PRESETS.get(hospital.hospital_id)
        features, labels = load_canonical_frame(str(hospital.dataset_path), preset)
    else:
        features, labels, _ = load_hospital_dataset(hospital, max_rows)
    features = model_spec.align_features(features)
    if features.shape[0] < 2:
        raise ValueError(f"{hospital.hospital_id} has too few local samples.")

    local_split = int(features.shape[0] * 0.8)
    train_x, val_x = features.iloc[:local_split], features.iloc[local_split:]
    train_y, val_y = labels.iloc[:local_split], labels.iloc[local_split:]
    if val_x.shape[0] == 0:
        val_x, val_y = train_x, train_y

    client = FederatedClient(
        model_factory=model_spec.make_model,
        X_train=train_x,
        y_train=train_y.to_numpy(),
        X_val=val_x,
        y_val=val_y.to_numpy(),
        privacy=privacy,
    )
    logger.info(
        "Hospital %s connecting to %s with %d local samples",
        hospital.hospital_id,
        address,
        features.shape[0],
    )
    start_numpy_client(
        server_address=address,
        client=client,
        root_certificates=_load_client_certificates(
            tls_enabled, tls_ca_cert, tls_client_cert, tls_client_key
        ),
    )
    logger.info("Hospital %s finished federation", hospital.hospital_id)


__all__ = [
    "DistributedFedAvg",
    "ModelSpec",
    "run_distributed_server",
    "run_hospital_client",
]
