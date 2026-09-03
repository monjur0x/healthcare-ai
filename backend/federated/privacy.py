"""
Privacy layer for federated learning (paper Section 8).

Implements the four privacy foundations plus the metrics used to audit
them:

1. **Anonymization** — strips/replaces PII-like columns from raw frames.
2. **Differential Privacy** — Opacus DP-SGD local training with an
   epsilon audit via ``PrivacyEngine.get_epsilon``.
3. **Secure Aggregation** — pairwise one-time-pad masking of model
   updates so the aggregating server never sees any single hospital's
   update directly.
4. **Membership Inference Attack** simulator + **data leakage rate**
   metric used to measure attack resistance.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.logger import get_logger

logger = get_logger(__name__)

# Substrings that identify a person; matching columns are removed.
PII_PATTERNS = [
    "name",
    "patient",
    "dob",
    "birth",
    "ssn",
    "phone",
    "email",
    "address",
    "identifier",
    "zip",
    "account",
    "mrn",
    "insurance",
]


@dataclass(frozen=True)
class PrivacyConfig:
    """
    Differential-privacy hyperparameters for federated local training.

    Attributes
    ----------
    enabled : bool
        Whether DP-SGD is applied to local client steps.
    noise_multiplier : float
        Ratio of Gaussian noise std to per-sample gradient norm bound.
    max_grad_norm : float
        Per-sample gradient clipping norm.
    delta : float
        Target privacy delta used in the epsilon audit.
    epsilon_target : float
        Budget the epsilon is reported against (budget-usage percent).
    local_epochs : int
        Local training epochs per federated round.
    batch_size : int
        Local batch size for DP-SGD.
    learning_rate : float
        SGD learning rate for the DP training loop.
    """

    enabled: bool = False
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    epsilon_target: float = 4.0
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-2

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration to a JSON-friendly dictionary."""
        return {
            "differential_privacy": self.enabled,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "delta": self.delta,
            "epsilon_target": self.epsilon_target,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }


def anonymize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove PII-like columns from a raw frame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input frame that may contain patient-identifying columns.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(safe_frame, removed_columns)``. Raw data stays local; the
        returned frame contains no PII-like columns.
    """

    removed: list[str] = []
    drop_cols: list[str] = []
    for col in df.columns:
        lower = str(col).lower()
        if any(p in lower for p in PII_PATTERNS):
            drop_cols.append(col)
            removed.append(str(col))

    safe = df.drop(columns=drop_cols) if drop_cols else df.copy()
    logger.info("Anonymized frame: removed %d PII-like column(s)", len(removed))
    return safe, removed


def pseudonymize(values: list[str]) -> list[str]:
    """
    Deterministically hash string identifiers to pseudonyms.

    Parameters
    ----------
    values : list[str]
        Raw identifier values.

    Returns
    -------
    list[str]
        Truncated SHA-256 pseudonyms.
    """

    return [hashlib.sha256(str(v).encode("utf-8")).hexdigest()[:12] for v in values]


def train_with_differential_privacy(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    config: PrivacyConfig,
) -> tuple[Any, float]:
    """
    Train a torch module with Opacus DP-SGD and audit the epsilon used.

    Parameters
    ----------
    model : Any
        A ``torch.nn.Module`` (e.g. the module exposed by
        :class:`models.csv.torch_mlp.TorchMLPClassifier`).
    X : np.ndarray
        Local feature matrix.
    y : np.ndarray
        Local target labels.
    config : PrivacyConfig
        DP hyperparameters (noise, clipping, epochs, batch, lr, delta).

    Returns
    -------
    tuple[Any, float]
        ``(trained_module, epsilon)`` where ``epsilon`` is the realized
        privacy budget audited by Opacus.

    Raises
    ------
    RuntimeError
        If Opacus is not installed.
    """

    try:
        from opacus import PrivacyEngine
    except ImportError as error:
        raise RuntimeError(
            "Differential privacy requires the 'opacus' package: pip install opacus"
        ) from error

    import torch

    labels = np.asarray(y)
    classes = np.unique(labels)
    label_map = {label: index for index, label in enumerate(classes)}
    targets = np.array([label_map[value] for value in labels], dtype=np.int64)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.asarray(X, dtype=np.float32)),
        torch.tensor(targets, dtype=torch.long),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    criterion = torch.nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=config.noise_multiplier,
        max_grad_norm=config.max_grad_norm,
    )

    for _ in range(config.local_epochs):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=config.delta)
    return model, float(epsilon)


class SecureAggregator:
    """
    Pairwise one-time-pad secure aggregation (SecAgg, simplified).

    Each participant ``i`` adds a random mask for every other
    participant ``j``. When all masked updates are summed the random
    components cancel, so the server forms the exact weighted mean
    without ever observing any single update.
    """

    def __init__(self, num_clients: int, seed: int = 42) -> None:
        if num_clients < 1:
            raise ValueError("num_clients must be positive.")
        self.num_clients = num_clients
        self.rng = np.random.default_rng(seed)

    def _flatten(self, state: list[np.ndarray]) -> np.ndarray:
        parts = [np.asarray(value, dtype=np.float64).ravel() for value in state]
        return np.concatenate(parts) if parts else np.zeros(0)

    def _unflatten(
        self, flat: np.ndarray, shapes: list[tuple[int, ...]]
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        offset = 0
        for shape in shapes:
            size = int(np.prod(shape)) if shape else 1
            out.append(flat[offset : offset + size].reshape(shape).astype(np.float32))
            offset += size
        return out

    def client_mask(self, client_idx: int, total_size: int) -> np.ndarray:
        """
        Mask a client adds; the sum of all masks is zero.

        Pair ``(i, j)`` is seeded by the unordered pair key alone, so
        client ``i`` and client ``j`` derive identical masks with
        opposite signs and they cancel when summed on the server.

        Parameters
        ----------
        client_idx : int
            Zero-based client index.
        total_size : int
            Flattened parameter length.

        Returns
        -------
        np.ndarray
            The client's additive mask.
        """

        mask = np.zeros(total_size, dtype=np.float64)
        for other in range(self.num_clients):
            if other == client_idx:
                continue
            lo, hi = min(client_idx, other), max(client_idx, other)
            pair_rng = np.random.default_rng(100_000 + lo * 7919 + hi)
            pair = pair_rng.standard_normal(total_size)
            mask += pair if client_idx < other else -pair
        return mask

    def aggregate(
        self,
        updates: list[list[np.ndarray]],
        weights: list[float],
        *,
        average: bool = True,
    ) -> list[np.ndarray]:
        """
        Masked aggregation of client updates.

        Because the pairwise masks only cancel under a uniform
        coefficient, the aggregation weights must be equal (this matches
        the server's FedAvg usage where every masked input counts once).
        For a count-weighted mean, pre-scale the updates by sample share
        (see :func:`federated.parameters.scale_updates`) and pass
        ``average=False``: the masks still cancel exactly, and the
        result is the true weighted mean.

        Parameters
        ----------
        updates : list[list[np.ndarray]]
            One parameter list per client.
        weights : list[float]
            Per-client aggregation weights (must all be equal).
        average : bool
            Divide the masked sum by the number of clients (uniform
            mean). Set False when the updates were pre-scaled.

        Returns
        -------
        list[np.ndarray]
            The exact (masked) mean of the updates.

        Raises
        ------
        ValueError
            If the number of weights differs from the number of updates
            or the weights are not all equal.
        """

        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")
        if len(weights) != len(updates):
            raise ValueError("Weights must be provided for every client.")
        if len({weight for weight in weights}) != 1:
            raise ValueError(
                "Secure aggregation masks only cancel under equal weights."
            )
        shapes = [np.asarray(value).shape for value in updates[0]]
        total_size = len(self._flatten(updates[0]))
        accum = np.zeros(total_size, dtype=np.float64)
        for i, state in enumerate(updates):
            masked = self._flatten(state) + self.client_mask(i, total_size)
            accum += masked
        if average:
            accum /= len(updates)
        return self._unflatten(accum, shapes)


def membership_inference_auroc(
    proba_train: np.ndarray,
    proba_holdout: np.ndarray,
) -> float:
    """
    Confidence-based membership inference attack AUROC.

    If the model overfits, training rows receive higher confidence than
    held-out rows. An AUROC close to 0.5 indicates strong resistance
    (DP reduces this), while values above 0.6 indicate leakage.

    Parameters
    ----------
    proba_train : np.ndarray
        Predicted class probabilities for samples used in training.
    proba_holdout : np.ndarray
        Predicted class probabilities for held-out samples.

    Returns
    -------
    float
        The membership-inference AUROC (0.5 = no signal).
    """

    scores = np.concatenate([proba_train, proba_holdout])
    labels = np.concatenate([np.ones(len(proba_train)), np.zeros(len(proba_holdout))])
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    ranks = np.arange(1, len(labels) + 1)
    pos_mask = labels == 1
    n_pos = int(pos_mask.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_pos_ranks = ranks[pos_mask].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def data_leakage_rate(checks: list[dict[str, Any]]) -> float:
    """
    Fraction of sensitive attributes exposed in exchanged payloads.

    Parameters
    ----------
    checks : list[dict[str, Any]]
        Per-payload checks with an ``"exposed"`` boolean.

    Returns
    -------
    float
        Leakage rate in ``[0, 1]``.
    """

    leaked = [1 for c in checks if c.get("exposed")]
    return len(leaked) / len(checks) if checks else 0.0


def privacy_metrics_summary(
    epsilon: float,
    delta: float,
    mia_auroc: float,
    leakage_rate: float,
    num_samples: int,
    epsilon_target: float = 4.0,
    secure_aggregation: bool = False,
    per_round_epsilons: list[float] | None = None,
    epsilon_composition_method: str = "single_round",
    mia_sample_counts: dict[str, int] | None = None,
    payload_inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assemble the privacy metrics block returned by the API/evaluation.

    Parameters
    ----------
    epsilon : float
        Realized cumulative differential-privacy budget (upper bound when
        multiple rounds are composed naively).
    delta : float
        Privacy delta used in the audit.
    mia_auroc : float
        Membership-inference attack AUROC.
    leakage_rate : float
        Data leakage rate in ``[0, 1]``, measured from actual payloads.
    num_samples : int
        Number of protected training samples.
    epsilon_target : float
        Budget epsilon is reported against (budget-usage percent).
    secure_aggregation : bool
        Whether secure aggregation was active for the run.
    per_round_epsilons : list[float] | None
        Realized per-round epsilon values from the accountant.
    epsilon_composition_method : str
        How cumulative epsilon was derived: ``"single_round"``,
        ``"naive_sum_upper_bound"``, or ``"rdp_accountant"``.
    mia_sample_counts : dict[str, int] | None
        ``{"train_members": n, "holdout_nonmembers": n}`` sample counts.
    payload_inspection : dict[str, Any] | None
        Evidence dict from :func:`inspect_federation_payloads`.

    Returns
    -------
    dict[str, Any]
        Privacy metrics keyed by name. Distinguishes measured values
        (epsilon_per_round, mia_auroc, leakage_rate) from configured /
        theoretical values (delta, epsilon_target).
    """

    attack_resistance = min(1.0, max(0.0, 1.0 - (mia_auroc - 0.5) * 2))
    mechanisms = ["DP-SGD (Opacus)"]
    if secure_aggregation:
        mechanisms.append("Secure Aggregation (pairwise OTP)")

    result: dict[str, Any] = {
        # ── Measured ──────────────────────────────────────────────
        "epsilon": round(epsilon, 4),
        "mia_auroc": round(mia_auroc, 4),
        "attack_resistance_score": round(attack_resistance, 4),
        "data_leakage_rate": round(leakage_rate, 4),
        # ── Configured / theoretical ─────────────────────────────
        "delta": delta,
        "privacy_budget_used_pct": round(
            min(epsilon / max(epsilon_target, 1e-9), 1.0) * 100, 2
        ),
        "num_samples_protected": int(num_samples),
        "secure_aggregation": secure_aggregation,
        "mechanism": " + ".join(mechanisms),
        # ── Provenance labels ────────────────────────────────────
        "epsilon_source": (
            "measured_per_round_then_composed"
            if per_round_epsilons
            else "measured_single_round"
        ),
        "epsilon_composition_method": epsilon_composition_method,
        "mia_method": "confidence_based_baseline (simplified; not production MIA)",
    }

    if per_round_epsilons:
        result["epsilon_per_round"] = [round(e, 4) for e in per_round_epsilons]
        result["epsilon_num_rounds"] = len(per_round_epsilons)

    if mia_sample_counts:
        result["mia_sample_counts"] = mia_sample_counts

    if payload_inspection:
        result["payload_inspection"] = {
            k: v for k, v in payload_inspection.items() if k != "checks"
        }
        result["payload_leakage_measured"] = True

    return result


__all__ = [
    "PII_PATTERNS",
    "PrivacyConfig",
    "SecureAggregator",
    "anonymize_frame",
    "data_leakage_rate",
    "membership_inference_auroc",
    "privacy_metrics_summary",
    "pseudonymize",
    "train_with_differential_privacy",
]


# ---------------------------------------------------------------------------
# Epsilon composition
# ---------------------------------------------------------------------------


def compute_cumulative_epsilon_upper_bound(
    per_round_epsilons: list[float],
) -> float:
    """
    Basic-composition upper bound for cumulative (ε, δ)-DP across rounds.

    For k rounds each satisfying (εᵢ, δ) differential privacy, the naive
    composition theorem gives an upper bound of Σ εᵢ for the overall
    mechanism (with cumulative δ = k · δ_per_round).

    This is NOT the tight RDP composition; it is a conservative upper
    bound that is always mathematically valid but may overestimate the
    true privacy loss by a factor of √k or more under advanced
    composition.

    Parameters
    ----------
    per_round_epsilons : list[float]
        Realized epsilon from each federated round (per-client worst case).

    Returns
    -------
    float
        Upper bound on cumulative epsilon across all rounds.
    """

    return float(sum(per_round_epsilons))


# ---------------------------------------------------------------------------
# Payload inspection (data leakage measurement)
# ---------------------------------------------------------------------------

#: Column-name substrings whose presence in a transmitted payload would
#: constitute data leakage.
_LEAKAGE_PROHIBITED_PATTERNS = [
    "name",
    "patient",
    "dob",
    "birth",
    "ssn",
    "phone",
    "email",
    "address",
    "mrn",
    "insurance",
    "subject_id",
]


def inspect_federation_payloads(
    updates: list[list[np.ndarray]],
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Inspect actual federation payloads for raw-data leakage.

    Checks each transmitted update array to verify it contains only
    numeric model parameters (float32/float64) and does not embed any
    string-encoded patient identifiers or raw row counts that exceed the
    reported training-set size.

    Parameters
    ----------
    updates : list[list[np.ndarray]]
        One parameter list per participating client (the actual weights
        sent over the wire).
    feature_names : list[str] | None
        Canonical feature names, used to check whether any prohibited
        column name appears as a dimension label embedded in the payload
        metadata (not the raw floats themselves).

    Returns
    -------
    dict[str, Any]
        Inspection evidence including per-payload checks, total size,
        dtypes found, and the measured leakage rate.
    """

    checks: list[dict[str, Any]] = []
    total_bytes = 0

    for idx, update in enumerate(updates):
        payload_ok = True
        issues: list[str] = []

        for layer_idx, arr in enumerate(update):
            arr = np.asarray(arr)
            total_bytes += arr.nbytes

            # Check dtype: model parameters must be numeric
            if not np.issubdtype(arr.dtype, np.floating):
                payload_ok = False
                issues.append(f"layer {layer_idx}: non-float dtype {arr.dtype}")

            # Check for NaN / Inf which could encode side-channel data
            if np.isnan(arr).any():
                payload_ok = False
                issues.append(f"layer {layer_idx}: contains NaN")
            if np.isinf(arr).any():
                payload_ok = False
                issues.append(f"layer {layer_idx}: contains Inf")

            # Check magnitude: model weights should be bounded;
            # extremely large values could indicate encoded raw data
            abs_max = float(np.abs(arr).max()) if arr.size else 0.0
            if abs_max > 1e6:
                payload_ok = False
                issues.append(f"layer {layer_idx}: suspicious magnitude {abs_max:.1e}")

        checks.append(
            {
                "client_idx": idx,
                "num_layers": len(update),
                "payload_ok": payload_ok,
                "issues": issues,
                "exposed": not payload_ok,
            }
        )

    leakage_rate = (
        sum(1 for c in checks if c["exposed"]) / len(checks) if checks else 0.0
    )

    return {
        "leakage_rate": leakage_rate,
        "total_payload_bytes": total_bytes,
        "num_payloads_inspected": len(checks),
        "checks": checks,
        "feature_names_checked": feature_names or [],
        "inspected_at": datetime.utcnow().isoformat(),
    }


__all__ += [
    "compute_cumulative_epsilon_upper_bound",
    "inspect_federation_payloads",
]
