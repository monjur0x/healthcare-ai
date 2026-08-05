"""Local model definitions for the federated fleet (Phase 3).

- ``HealthMLP``: small PyTorch MLP used as the default tabular learner and fed
  through FedAvg weight aggregation (optionally trained with Opacus DP-SGD).
- ``HealthCNN``: lightweight CNN stub for the image modality. In production this
  is replaced by EfficientNetV2 / DenseNet121 / Swin-Transformer; the training
  loop is identical (state-dict averaging).
- ``XGBoostLearner``: gradient-boosted local learner used as an alternative
  tabular baseline (federated ensemble score-averaging).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .data import FEATURE_COLUMNS

INPUT_DIM = len(FEATURE_COLUMNS)

# Fixed clinical reference statistics used to standardize features. These are
# public clinical norms, so applying them locally at every hospital introduces
# no data sharing and keeps training and inference consistent.
FEATURE_MEANS = [50.0, 27.0, 125.0, 80.0, 110.0, 210.0, 48.0, 80.0, 1.1, 13.5, 96.0, 37.0, 7.5, 250.0, 5.0]
FEATURE_STDS = [18.0, 6.0, 20.0, 12.0, 35.0, 45.0, 12.0, 20.0, 0.8, 2.0, 3.0, 0.7, 3.0, 70.0, 5.0]


def standardize(X: np.ndarray) -> np.ndarray:
    """Deterministic standardization using fixed clinical norms."""
    X = np.asarray(X, dtype=np.float32)
    means = np.array(FEATURE_MEANS, dtype=np.float32)
    stds = np.array(FEATURE_STDS, dtype=np.float32)
    return (X - means) / stds


class HealthMLP(nn.Module):
    """Small multi-layer perceptron for unified clinical risk prediction."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_sizes: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HealthCNN(nn.Module):
    """Stub CNN for image modality. Swap for EfficientNetV2/DenseNet/Swin-T."""

    def __init__(self, out_dim: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.head(feat)


@dataclass
class XGBoostLearner:
    """Thin wrapper around a scikit-learn GradientBoostingClassifier.

    Federated aggregation for boosted trees is performed via score-averaging
    (an ensemble), since gradient-boosted trees do not expose a weight vector.
    """

    model: Any | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostLearner":
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Learner not trained")
        return self.model.predict_proba(X)[:, 1]

    def _state(self) -> dict[str, Any]:
        return {"model": self.model}

    def _load(self, state: dict[str, Any]) -> None:
        self.model = state["model"]


def build_tensor_dataset(X: np.ndarray, y: np.ndarray) -> torch.utils.data.TensorDataset:
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    return torch.utils.data.TensorDataset(x_t, y_t)


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Sigmoid probabilities for the MLP/CNN."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(standardize(X), dtype=torch.float32)
        if isinstance(model, HealthCNN):
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        logits = model(x)
        return torch.sigmoid(logits).numpy().ravel()


def _mlp_from_state(state: dict[str, torch.Tensor]) -> HealthMLP:
    model = HealthMLP()
    model.load_state_dict(state)
    return model


def extract_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract a clean state dict, stripping Opacus's ``_module.`` prefix."""
    raw = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        key = k[8:] if k.startswith("_module.") else k
        out[key] = v.cpu().clone()
    return out


def average_state_dicts(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    """Weighted FedAvg of a list of state dicts."""
    summed: dict[str, torch.Tensor] = {}
    total_weight = sum(weights)
    for key in states[0].keys():
        acc = None
        for state, w in zip(states, weights):
            term = state[key].float() * (w / total_weight)
            acc = term if acc is None else acc + term
        summed[key] = acc
    return summed