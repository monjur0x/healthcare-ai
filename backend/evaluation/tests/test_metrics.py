"""
Tests for the evaluation metrics module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification

from evaluation import classification_metrics, evaluate_classifier
from models import FusionClassifier, ImageClassifier, TabularClassifier
from preprocessing.multimodal import MultimodalFusion


def _binary_data(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y_true = np.repeat([0, 1], n // 2)
    logits = np.where(y_true == 1, rng.normal(1.0, 0.3, n), rng.normal(-1.0, 0.3, n))
    score = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (score >= 0.5).astype(int)
    return y_true, score, y_pred


def test_metrics_perfect_prediction() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = y_true.copy()
    metrics = classification_metrics(y_true, y_pred, labels=np.array([0, 1]))
    assert metrics.accuracy == 1.0
    assert metrics.precision_macro == 1.0
    assert metrics.recall_macro == 1.0
    assert metrics.f1_macro == 1.0
    assert metrics.mcc == 1.0
    assert metrics.n_samples == 5
    assert metrics.roc_auc is None  # no score matrix


def test_metrics_binary_with_scores() -> None:
    y_true, score, y_pred = _binary_data()
    metrics = classification_metrics(y_true, y_pred, y_score=score)
    assert metrics.accuracy > 0.8
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.8
    assert metrics.pr_auc is not None and metrics.pr_auc > 0.5
    assert metrics.log_loss_value is not None and metrics.log_loss_value >= 0.0


def test_metrics_multiclass_with_scores() -> None:
    _X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_classes=3,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
    )
    rng = np.random.default_rng(1)
    score = np.column_stack(
        [
            rng.uniform(0.4, 0.6, 90),
            rng.uniform(0.0, 0.2, 90),
            rng.uniform(0.0, 0.2, 90),
        ]
    )
    score = score / score.sum(axis=1, keepdims=True)
    y_pred = np.argmax(score, axis=1)
    metrics = classification_metrics(y, y_pred, y_score=score)
    assert metrics.accuracy > 0.0
    assert metrics.roc_auc is not None
    assert metrics.pr_auc is not None


def test_metrics_single_class_target() -> None:
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    score = np.column_stack([np.full(10, 0.5), np.full(10, 0.5)])
    metrics = classification_metrics(
        y_true, y_pred, y_score=score, labels=np.array([0, 1])
    )
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc is None
    assert metrics.pr_auc is None
    assert metrics.log_loss_value is None


def test_metrics_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        classification_metrics(np.array([0, 1]), np.array([0]))


def test_metrics_1d_score_requires_two_classes() -> None:
    y_true = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError):
        classification_metrics(y_true, y_true, y_score=score, labels=np.array([0]))


def test_evaluate_classifier_tabular() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    model = TabularClassifier(model_name="logistic").fit(X, y)
    metrics = evaluate_classifier(model, X, y)
    assert metrics.n_samples == 120
    assert metrics.accuracy > 0.7
    assert metrics.roc_auc is not None


def test_evaluate_classifier_cnn() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 12)
    images = rng.normal(size=(24, 12, 12, 3)).astype(np.float32)
    for index, label in enumerate(labels):
        images[index, ..., label] += 2.0 * (label + 1)
    model = ImageClassifier(epochs=3, batch_size=8).fit(images, labels)
    metrics = evaluate_classifier(model, images, labels)
    assert metrics.n_samples == 24
    assert metrics.accuracy > 0.6
    assert metrics.roc_auc is not None


def test_evaluate_classifier_fusion() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 12)
    dataframe = pd.DataFrame(
        {
            "age": np.concatenate([rng.normal(40, 3, 12), rng.normal(60, 3, 12)]),
        }
    )
    images = rng.normal(size=(24, 12, 12, 3)).astype(np.float32)
    for index, label in enumerate(labels):
        images[index, ..., label] += 3.0 * (label + 1)
    result = MultimodalFusion(image_reduction="summary").transform(dataframe, images)
    model = FusionClassifier().fit(result, labels)
    metrics = evaluate_classifier(model, result, labels)
    assert metrics.n_samples == 24
    assert metrics.accuracy > 0.7
    assert metrics.roc_auc is not None


def test_evaluate_classifier_requires_fit() -> None:
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    with pytest.raises(ValueError):
        evaluate_classifier(TabularClassifier(model_name="logistic"), X, y)


def test_metrics_serialization() -> None:
    y_true, score, y_pred = _binary_data()
    metrics = classification_metrics(y_true, y_pred, y_score=score)
    payload = metrics.to_dict()
    assert payload["n_samples"] == 100
    assert "roc_auc" in payload
    assert "mcc" in payload
