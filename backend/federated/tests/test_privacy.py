"""
Tests for the federated privacy layer (paper Section 8).

Covers anonymization, pseudonymization, Opacus DP-SGD with an epsilon
audit, pairwise secure aggregation, the membership-inference simulator,
the leakage-rate metric, and the assembled privacy metrics summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from federated.privacy import (
    PII_PATTERNS,
    PrivacyConfig,
    SecureAggregator,
    anonymize_frame,
    data_leakage_rate,
    membership_inference_auroc,
    privacy_metrics_summary,
    pseudonymize,
    train_with_differential_privacy,
)
from models import TorchMLPClassifier


def test_pii_patterns_cover_identifiers() -> None:
    assert any("name" in pattern for pattern in PII_PATTERNS)
    assert any("ssn" in pattern for pattern in PII_PATTERNS)
    assert any("phone" in pattern for pattern in PII_PATTERNS)


def test_anonymize_frame_removes_pii_columns() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "full_name": ["a", "b"],
            "glucose": [100.0, 120.0],
            "Outcome": [0, 1],
        }
    )
    safe, removed = anonymize_frame(frame)
    assert removed == ["patient_id", "full_name"]
    assert list(safe.columns) == ["glucose", "Outcome"]
    assert safe["glucose"].tolist() == [100.0, 120.0]


def test_anonymize_frame_no_pii_is_copy() -> None:
    frame = pd.DataFrame({"glucose": [1.0], "Outcome": [0]})
    safe, removed = anonymize_frame(frame)
    assert removed == []
    assert list(safe.columns) == ["glucose", "Outcome"]


def test_pseudonymize_is_deterministic_and_lossy() -> None:
    first = pseudonymize(["alice", "bob"])
    second = pseudonymize(["alice", "bob"])
    assert first == second
    assert first[0] != "alice"
    assert len(first[0]) == 12


def test_train_with_differential_privacy_reports_epsilon() -> None:
    rng = np.random.default_rng(4)
    X = rng.uniform(0, 1, (100, 4))
    y = (X[:, 0] > 0.5).astype(int)
    model = TorchMLPClassifier(n_features=4, n_classes=2, seed=4)
    model.fit(X, y)
    _, epsilon = train_with_differential_privacy(
        model.module,
        X,
        y,
        PrivacyConfig(
            enabled=True,
            noise_multiplier=2.0,
            max_grad_norm=1.0,
            local_epochs=1,
        ),
    )
    assert epsilon > 0.0


def test_train_with_differential_privacy_requires_opacus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "opacus":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (20, 2))
    y = (X[:, 0] > 0.5).astype(int)
    model = TorchMLPClassifier(n_features=2, n_classes=2, seed=1)
    with pytest.raises(RuntimeError, match="opacus"):
        train_with_differential_privacy(
            model.module,
            X,
            y,
            PrivacyConfig(enabled=True, noise_multiplier=1.0, local_epochs=1),
        )


def test_secure_aggregation_cancels_masks_to_weighted_mean() -> None:
    aggregator = SecureAggregator(num_clients=3, seed=7)
    updates = [
        [np.ones(6, dtype=np.float32)],
        [np.full(6, 2.0, dtype=np.float32)],
        [np.full(6, 3.0, dtype=np.float32)],
    ]
    result = aggregator.aggregate(updates, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(result[0], 2.0, atol=1e-6)


def test_secure_aggregation_requires_equal_weights() -> None:
    aggregator = SecureAggregator(num_clients=2, seed=11)
    updates = [[np.zeros(4, dtype=np.float32)], [np.ones(4, dtype=np.float32)]]
    with pytest.raises(ValueError, match="equal weights"):
        aggregator.aggregate(updates, [3.0, 1.0])


def test_secure_aggregation_rejects_weight_mismatch() -> None:
    aggregator = SecureAggregator(num_clients=2, seed=11)
    updates = [[np.zeros(4)], [np.ones(4)]]
    with pytest.raises(ValueError, match="every client"):
        aggregator.aggregate(updates, [1.0])


def test_secure_aggregation_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        SecureAggregator(num_clients=2, seed=1).aggregate([], [])


def test_membership_inference_auroc_signal() -> None:
    rng = np.random.default_rng(2)
    proba_member = rng.uniform(0.7, 1.0, 80)
    proba_holdout = rng.uniform(0.0, 0.5, 80)
    auc = membership_inference_auroc(proba_member, proba_holdout)
    assert auc > 0.7


def test_membership_inference_auroc_no_signal() -> None:
    rng = np.random.default_rng(3)
    proba = rng.uniform(0.3, 0.7, 100)
    auc = membership_inference_auroc(proba.copy(), proba.copy())
    assert 0.35 <= auc <= 0.65


def test_data_leakage_rate() -> None:
    assert data_leakage_rate([{"exposed": False}, {"exposed": False}]) == 0.0
    assert data_leakage_rate([{"exposed": True}, {"exposed": False}]) == 0.5
    assert data_leakage_rate([]) == 0.0


def test_privacy_metrics_summary_shape() -> None:
    summary = privacy_metrics_summary(
        epsilon=2.0,
        delta=1e-5,
        mia_auroc=0.52,
        leakage_rate=0.0,
        num_samples=100,
        epsilon_target=4.0,
        secure_aggregation=True,
    )
    assert summary["epsilon"] == 2.0
    assert summary["privacy_budget_used_pct"] == 50.0
    assert summary["mia_auroc"] == 0.52
    assert summary["attack_resistance_score"] == pytest.approx(0.96, abs=1e-6)
    assert summary["data_leakage_rate"] == 0.0
    assert summary["num_samples_protected"] == 100
    assert summary["secure_aggregation"] is True
    assert "DP-SGD" in summary["mechanism"]
    assert "Secure Aggregation" in summary["mechanism"]
