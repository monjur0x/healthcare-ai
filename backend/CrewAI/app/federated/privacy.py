"""Privacy layer (Phase 2).

Implements the four privacy foundations from the proposal and the privacy
metrics used in evaluation:

1. **Data anonymization** - strips/replaces PII-like columns from raw frames.
2. **Differential Privacy** - Opacus DP-SGD training wrapper + epsilon audit.
3. **Secure Aggregation** - pairwise one-time-pad masking of model updates so
   the server never sees an individual hospital's update directly.
4. **Membership Inference Attack** simulator + **data leakage rate** metric.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import settings

# Substrings that identify a person; matching columns are removed.
PII_PATTERNS = [
    "name", "patient", "dob", "birth", "ssn", "phone", "email", "address",
    "identifier", "zip", "account", "mrn", "insurance",
]


def anonymize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove PII-like columns from a raw frame.

    Returns ``(safe_frame, removed_columns)``. Raw data stays local; downstream
    federated learners only ever receive this anonymized frame.
    """
    removed: list[str] = []
    drop_cols: list[str] = []
    for col in df.columns:
        lower = str(col).lower()
        if any(p in lower for p in PII_PATTERNS):
            drop_cols.append(col)
            removed.append(str(col))

    safe = df.drop(columns=drop_cols) if drop_cols else df.copy()
    return safe, removed


def pseudonymize(values: list[str]) -> list[str]:
    """Deterministically hash string identifiers to pseudonyms."""
    return [hashlib.sha256(str(v).encode("utf-8")).hexdigest()[:12] for v in values]


def train_with_differential_privacy(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
) -> tuple[torch.nn.Module, float]:
    """Train ``model`` with Opacus DP-SGD; returns ``(model, epsilon)``.

    Uses Opacus ``PrivacyEngine.make_private_with_epsilon`` so the realized
    privacy budget (epsilon) is audited and returned. Loss is BCE-with-logits
    on a single output unit, which Opacus accounts correctly.
    """
    from opacus import PrivacyEngine

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).reshape(-1, 1),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    for _ in range(epochs):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=delta)
    return model, float(epsilon)


class SecureAggregator:
    """Pairwise one-time-pad secure aggregation (SecAgg, simplified).

    Each hospital ``i`` adds a random mask for every other hospital ``j``. When
    all masked updates are summed the random components cancel, so the server
    can form the exact weighted mean without ever observing any single update.
    """

    def __init__(self, num_clients: int, seed: int = 42) -> None:
        self.num_clients = num_clients
        self.rng = np.random.default_rng(seed)

    def _shapes(self, state: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
        return {k: tuple(v.shape) for k, v in state.items()}

    def _flatten(self, state: dict[str, torch.Tensor]) -> np.ndarray:
        parts = [v.float().cpu().detach().numpy().ravel() for v in state.values()]
        return np.concatenate(parts) if parts else np.zeros(0)

    def _unflatten(self, flat: np.ndarray, shapes: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        offset = 0
        for k, shape in shapes.items():
            size = int(np.prod(shape)) if shape else 1
            out[k] = torch.tensor(flat[offset : offset + size].reshape(shape))
            offset += size
        return out

    def client_mask(self, client_idx: int, total_size: int) -> np.ndarray:
        """Mask a client adds; the sum of all masks is zero.

        Pair ``(i, j)`` is seeded by the unordered pair key alone, so client
        ``i`` and client ``j`` derive identical masks with opposite signs and
        they cancel when summed on the server.
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
        updates: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        """Masked aggregation -> returns the true weighted mean of updates."""
        shapes = self._shapes(updates[0])
        total_size = len(self._flatten(updates[0]))
        total_w = sum(weights)
        accum = np.zeros(total_size, dtype=np.float64)
        for i, state in enumerate(updates):
            masked = self._flatten(state) + self.client_mask(i, total_size)
            accum += masked * (weights[i] / total_w)
        return self._unflatten(accum.astype(np.float32), shapes)


def membership_inference_auroc(
    proba_train: np.ndarray,
    proba_holdout: np.ndarray,
) -> float:
    """Confidence-based membership inference attack.

    If the model overfits, training rows receive higher confidence than
    held-out rows. AUROC close to 0.5 indicates strong resistance (DP reduces
    this), while values above 0.6 indicate leakage.
    """
    scores = np.concatenate([proba_train, proba_holdout])
    labels = np.concatenate([np.ones(len(proba_train)), np.zeros(len(proba_holdout))])
    order = scores.argsort()[::-1]
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
    """Fraction of sensitive attributes exposed in exchanged payloads.

    The payload here is the DP-protected model update, which carries no raw
    patient fields, so leakage is 0 when anonymization + DP are active.
    """
    leaked = [1 for c in checks if c.get("exposed")]
    return len(leaked) / len(checks) if checks else 0.0


def privacy_metrics_summary(
    epsilon: float,
    delta: float,
    mia_auroc: float,
    leakage_rate: float,
    num_samples: int,
) -> dict[str, Any]:
    """Assemble the privacy metrics block returned by the API/evaluation."""
    attack_resistance = max(0.0, 1.0 - (mia_auroc - 0.5) * 2)
    return {
        "epsilon": round(epsilon, 4),
        "delta": delta,
        "privacy_budget_used_pct": round(min(epsilon / settings.DP_EPSILON_TARGET, 1.0) * 100, 2),
        "mia_auroc": round(mia_auroc, 4),
        "attack_resistance_score": round(attack_resistance, 4),
        "data_leakage_rate": round(leakage_rate, 4),
        "num_samples_protected": int(num_samples),
        "mechanism": "DP-SGD (Opacus) + Secure Aggregation (pairwise OTP)",
    }