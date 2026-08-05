"""Federated Averaging (FedAvg) server and local trainers (Phase 3).

Each hospital keeps its raw data; only model state is exchanged. The server
aggregates the (optionally DP-protected and securely-masked) updates into a
global model and broadcasts it back each round.

``model_type``:
  - ``mlp``: PyTorch MLP, state-dict FedAvg (default).
  - ``xgboost``: local GradientBoosted learners, federated score-averaging.
  - ``cnn``: state-dict FedAvg on the stub CNN (image modality placeholder).

Metrics tracked per round: local/global validation accuracy & AUC, comm cost
in bytes, round wall time, and convergence curve.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from ..config import settings
from . import models
from .data import FEATURE_COLUMNS, feature_matrix
from .privacy import SecureAggregator, train_with_differential_privacy


def _validate_split(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    val = df.sample(frac=val_frac, random_state=seed)
    train = df.drop(val.index)
    return train, val


def _binary_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score

    y = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, y)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = 0.5
    return {"accuracy": round(float(acc), 4), "auc": round(float(auc), 4)}


class LocalMLPTrainer:
    """Trains an MLP locally for a hospital, optionally with DP-SGD."""

    def __init__(
        self,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        use_dp: bool,
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_dp = use_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def train(
        self,
        init_state: dict[str, torch.Tensor],
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        model = models.HealthMLP()
        model.load_state_dict(init_state)
        epsilon: float | None = None
        X = models.standardize(X)

        if self.use_dp:
            model, epsilon = train_with_differential_privacy(
                model, X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            dataset = models.build_tensor_dataset(X, y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            model.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    optimizer.step()

        state = models.extract_state(model)
        metrics = {
            "local_epochs": self.epochs,
            "dp_enabled": self.use_dp,
            "epsilon": epsilon,
        }
        return state, metrics


def _local_train_xgboost(X: np.ndarray, y: np.ndarray) -> models.XGBoostLearner:
    learner = models.XGBoostLearner()
    return learner.fit(X, y)


@dataclass
class FedAvgServer:
    """Runs the federated training rounds and records all metrics."""

    hospital_datasets: list[tuple[str, pd.DataFrame]]
    num_rounds: int = settings.FL_NUM_ROUNDS
    local_epochs: int = settings.FL_LOCAL_EPOCHS
    batch_size: int = settings.FL_BATCH_SIZE
    learning_rate: float = settings.FL_LEARNING_RATE
    model_type: str = settings.FL_MODEL_TYPE
    use_dp: bool = settings.DP_ENABLED
    noise_multiplier: float = settings.DP_NOISE_MULTIPLIER
    max_grad_norm: float = settings.DP_MAX_GRAD_NORM
    val_frac: float = 0.2
    seed: int = settings.FL_SEED
    secure: bool = True
    artifact_dir: str = settings.FL_ARTIFACT_DIR
    logger: Callable[[str], None] | None = None

    rounds_log: list[dict[str, Any]] = field(default_factory=list)
    comm_cost_bytes: int = 0
    total_time: float = 0.0
    global_model: torch.nn.Module | None = None
    ensemble_learners: list[models.XGBoostLearner] = field(default_factory=list)

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger(msg)

    def run(self) -> dict[str, Any]:
        start = time.time()
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.model_type == "xgboost":
            result = self._run_xgboost_federation()
        else:
            result = self._run_mlp_federation()

        self.total_time = time.time() - start
        result.update({
            "model_type": self.model_type,
            "num_hospitals": len(self.hospital_datasets),
            "num_rounds": self.num_rounds,
            "total_comm_cost_bytes": self.comm_cost_bytes,
            "total_time_seconds": round(self.total_time, 2),
            "secure_aggregation": self.secure,
            "differential_privacy": self.use_dp if self.model_type != "xgboost" else "not-applicable",
            "rounds_log": self.rounds_log,
        })
        return result

    def _run_mlp_federation(self) -> dict[str, Any]:
        # Split each hospital into train/val once (val never leaves the site).
        splits = [_validate_split(df, self.val_frac, self.seed) for _, df in self.hospital_datasets]
        trains = [t for t, _ in splits]
        vals = [v for _, v in splits]
        print(f"DEBUG after split: types={[type(v).__name__ for v in vals]}")
        sizes = [len(t) for t in trains]

        global_model = models.HealthMLP()
        init_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        aggregator = SecureAggregator(len(trains), seed=self.seed) if self.secure else None

        global_proba_val: np.ndarray | None = None
        global_y_val: np.ndarray | None = None

        for rnd in range(1, self.num_rounds + 1):
            round_start = time.time()
            updates: list[dict[str, torch.Tensor]] = []
            update_weights: list[float] = []
            round_eps: list[float | None] = []

            for i, (t, v) in enumerate(zip(trains, vals)):
                print(f"DEBUG rnd={rnd} i={i} type(t)={type(t).__name__} type(v)={type(v).__name__}")
                X_t, y_t = feature_matrix(t), t[["label"]].to_numpy().ravel().astype(int)
                state, metrics = LocalMLPTrainer(
                    learning_rate=self.learning_rate,
                    epochs=self.local_epochs,
                    batch_size=self.batch_size,
                    use_dp=self.use_dp,
                    noise_multiplier=self.noise_multiplier,
                    max_grad_norm=self.max_grad_norm,
                ).train(init_state, X_t, y_t)

                updates.append(state)
                update_weights.append(float(len(t)))
                round_eps.append(metrics["epsilon"])

                # record local val metrics (computed at site, stored globally)
                proba = models.predict_proba(models._mlp_from_state(state), feature_matrix(v))
                self.rounds_log.append({
                    "round": rnd, "hospital": self.hospital_datasets[i][0],
                    "phase": "local_val", "metric": "accuracy",
                    "value": _binary_metrics(v[["label"]].to_numpy().ravel().astype(int), proba)["accuracy"],
                })

            if self.secure and aggregator is not None:
                aggregated = aggregator.aggregate(updates, update_weights)
            else:
                aggregated = models.average_state_dicts(updates, update_weights)

            init_state = aggregated
            global_model = models._mlp_from_state(aggregated)

            # Global validation across all hospital val sets (simulated global eval).
            all_y = []
            all_p = []
            for v in vals:
                all_y.append(v[["label"]].to_numpy().ravel().astype(int))
                all_p.append(models.predict_proba(global_model, feature_matrix(v)))
            y_true = np.concatenate(all_y)
            proba = np.concatenate(all_p)
            global_proba_val, global_y_val = proba, y_true
            g_metrics = _binary_metrics(y_true, proba)

            self.comm_cost_bytes += sum(
                len(state_bytes) for state_bytes in (self._serialize(s) for s in updates)
            ) * 2  # up + broadcast
            mean_eps = None
            if any(round_eps):
                vals = [e for e in round_eps if e is not None]
                if vals:
                    mean_eps = round(float(np.mean(vals)), 4)
            self._log(
                f"Round {rnd}/{self.num_rounds} acc={g_metrics['accuracy']:.4f} "
                f"auc={g_metrics['auc']:.4f} eps={mean_eps or 'n/a'} "
                f"comm={self.comm_cost_bytes / 1024:.1f}KB"
            )
            self.rounds_log.append({
                "round": rnd, "hospital": "GLOBAL", "phase": "global_val",
                "accuracy": g_metrics["accuracy"], "auc": g_metrics["auc"],
                "comm_cost_bytes": self.comm_cost_bytes,
                "round_time_seconds": round(time.time() - round_start, 2),
                "mean_epsilon": mean_eps,
            })

        self.global_model = global_model
        return {
            "global_val_accuracy": g_metrics["accuracy"],
            "global_val_auc": g_metrics["auc"],
            "n_global_val": len(global_y_val) if global_y_val is not None else 0,
            "model_size_bytes": self.comm_cost_bytes // max(2 * self.num_rounds, 1),
            "per_client_epsilon": [float(e) if e is not None else None for e in round_eps],
        }

    def _run_xgboost_federation(self) -> dict[str, Any]:
        learners = []
        weights = []
        all_y, all_p = [], []
        for i, (name, df) in enumerate(self.hospital_datasets):
            train, val = _validate_split(df, self.val_frac, self.seed)
            learner = _local_train_xgboost(feature_matrix(train), train[["label"]].to_numpy().ravel().astype(int))
            learners.append(learner)
            weights.append(len(train))
            all_y.append(val[["label"]].to_numpy().ravel().astype(int))
            all_p.append(learner.predict_proba(feature_matrix(val)))
            self.rounds_log.append({
                "round": 1, "hospital": name, "phase": "local_val",
                "metric": "accuracy",
                "value": _binary_metrics(all_y[-1], all_p[-1])["accuracy"],
            })

        y_true = np.concatenate(all_y)
        w = np.array(weights, dtype=float) / sum(weights)
        proba = sum(p * w[i] for i, p in enumerate(all_p))
        g_metrics = _binary_metrics(y_true, proba)
        self.ensemble_learners = learners
        self._log(f"XGBoost federation acc={g_metrics['accuracy']:.4f} auc={g_metrics['auc']:.4f}")
        self.rounds_log.append({
            "round": 1, "hospital": "GLOBAL", "phase": "global_val",
            "accuracy": g_metrics["accuracy"], "auc": g_metrics["auc"],
            "comm_cost_bytes": self.comm_cost_bytes, "round_time_seconds": 0.0,
            "mean_epsilon": None,
        })
        return {"global_val_accuracy": g_metrics["accuracy"], "global_val_auc": g_metrics["auc"]}

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Global-model inference for new features."""
        if self.model_type == "xgboost":
            if not self.ensemble_learners:
                raise RuntimeError("Federation not run")
            return np.mean([l.predict_proba(features) for l in self.ensemble_learners], axis=0)
        if self.global_model is None:
            raise RuntimeError("Federation not run")
        return models.predict_proba(self.global_model, features)

    def _serialize(self, state: dict[str, torch.Tensor]) -> bytes:
        import io

        buf = io.BytesIO()
        torch.save(state, buf)
        return buf.getvalue()

    def save_artifacts(self, out_dir: str | Path | None = None) -> Path:
        out = Path(out_dir or self.artifact_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.global_model is not None:
            torch.save(self.global_model.state_dict(), out / "global_model.pt")
        elif self.ensemble_learners:
            import joblib

            for i, l in enumerate(self.ensemble_learners):
                joblib.dump(l.model, out / f"local_xgb_{i}.joblib")
        with open(out / "federation_summary.json", "w") as fh:
            json.dump(self.rounds_log, fh, indent=2, default=str)
        return out


def load_global_model(model_type: str, artifact_dir: str | Path) -> torch.nn.Module:
    """Load a previously saved global MLP/CNN model from disk."""
    path = Path(artifact_dir) / "global_model.pt"
    model = models.HealthMLP() if model_type == "mlp" else models.HealthCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model