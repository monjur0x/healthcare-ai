"""
Evaluation metrics for healthcare prediction models.

Implements the prediction metrics from the research proposal (§12
Evaluation Metrics): accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
and MCC. Scores are computed from model outputs (``predict`` /
``predict_proba``) so any :class:`BaseModel` can be evaluated uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.base import BaseModel
from preprocessing.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassificationMetrics:
    """
    Aggregated classification results for a single evaluation run.

    Probability-based metrics (``roc_auc``, ``pr_auc``,
    ``log_loss_value``) are ``None`` when no score matrix is available or
    the target cannot support them (e.g. a single observed class).
    ``mcc`` is ``None`` when the Matthews coefficient is undefined.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    mcc: float | None
    roc_auc: float | None
    pr_auc: float | None
    log_loss_value: float | None
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the metrics to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, Any]
            Metric values keyed by name.
        """

        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "mcc": self.mcc,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "log_loss": self.log_loss_value,
            "n_samples": self.n_samples,
        }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> ClassificationMetrics:
    """
    Compute classification metrics from ground truth and model outputs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels (from ``model.predict``).
    y_score : np.ndarray | None
        Probability matrix of shape (n_samples, n_classes) aligned with
        ``labels``, or a 1D binary score for the positive class. When
        omitted, probability-based metrics are skipped.
    labels : np.ndarray | None
        Class labels defining column order of ``y_score`` and the macro
        average. Inferred from the observed labels when omitted.

    Returns
    -------
    ClassificationMetrics
        Computed metrics.

    Raises
    ------
    ValueError
        If input arrays are malformed or misaligned.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true has {len(y_true)} samples but y_pred has {len(y_pred)}."
        )

    observed = np.unique(y_true)
    if labels is None:
        labels = np.unique(np.concatenate([observed, np.unique(y_pred)]))
    label_list = labels.tolist()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(
        precision_score(
            y_true, y_pred, average="macro", labels=label_list, zero_division=0
        )
    )
    recall_macro = float(
        recall_score(
            y_true, y_pred, average="macro", labels=label_list, zero_division=0
        )
    )
    f1_macro = float(
        f1_score(y_true, y_pred, average="macro", labels=label_list, zero_division=0)
    )
    mcc = _matthews_correlation(y_true, y_pred)

    roc_auc, pr_auc, log_loss_value = _score_metrics(
        y_true, y_score, observed, label_list
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        mcc=mcc,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        log_loss_value=log_loss_value,
        n_samples=len(y_true),
    )


def evaluate_classifier(
    model: BaseModel,
    X: Any,
    y_true: np.ndarray,
) -> ClassificationMetrics:
    """
    Score a fitted model on a dataset using its prediction interface.

    Parameters
    ----------
    model : BaseModel
        Fitted model exposing ``predict`` and, optionally,
        ``predict_proba``.
    X : Any
        Input samples matching the model's expected layout.
    y_true : np.ndarray
        Ground-truth labels.

    Returns
    -------
    ClassificationMetrics
        Computed metrics.

    Raises
    ------
    ValueError
        If the model is not fitted.
    """

    if not model.is_fitted:
        raise ValueError(
            f"{model.__class__.__name__} must be fitted before evaluation."
        )

    y_pred = np.asarray(model.predict(X))
    y_score: np.ndarray | None
    try:
        y_score = np.asarray(model.predict_proba(X), dtype=np.float64)
    except NotImplementedError:
        logger.info(
            "%s does not expose predict_proba; AUC metrics omitted.",
            model.__class__.__name__,
        )
        y_score = None

    labels = getattr(model, "classes_", None)
    return classification_metrics(y_true, y_pred, y_score, labels=labels)


def _matthews_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Compute MCC, returning None when the coefficient is undefined."""
    if set(np.unique(y_pred)) != set(np.unique(y_true)):
        return None
    value = matthews_corrcoef(y_true, y_pred)
    if np.isnan(value):
        return None
    return float(value)


def _score_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray | None,
    observed: np.ndarray,
    label_list: list[Any],
) -> tuple[float | None, float | None, float | None]:
    """Compute ROC-AUC, PR-AUC, and log loss from a score matrix."""
    if y_score is None:
        return None, None, None

    score = np.asarray(y_score, dtype=np.float64)
    if score.ndim == 1:
        if len(label_list) != 2:
            raise ValueError("A 1D y_score requires exactly two classes.")
        score = np.column_stack([1.0 - score, score])
    if score.ndim != 2:
        raise ValueError(f"y_score must be 1D or 2D, got {score.ndim} dimensions.")
    if score.shape[0] != len(y_true):
        raise ValueError(
            f"y_score has {score.shape[0]} rows but y_true has {len(y_true)}."
        )
    if score.shape[1] != len(label_list):
        raise ValueError(
            f"y_score has {score.shape[1]} columns but labels has {len(label_list)}."
        )

    if len(observed) < 2:
        logger.warning(
            "Cannot compute AUC/log-loss metrics with a single observed class."
        )
        return None, None, None

    try:
        if len(label_list) == 2:
            binary = (y_true == label_list[1]).astype(int)
            if np.unique(binary).size < 2:
                return None, None, None
            positive_score = score[:, 1]
            roc_auc = float(roc_auc_score(binary, positive_score))
            pr_auc = float(average_precision_score(binary, positive_score))
        else:
            roc_auc = float(
                roc_auc_score(y_true, score, multi_class="ovr", labels=label_list)
            )
            pr_auc = float(average_precision_score(y_true, score, average="macro"))
        log_loss_value = float(log_loss(y_true, score, labels=label_list))
    except ValueError as exc:
        logger.warning("Skipped AUC/log-loss metrics: %s", exc)
        return None, None, None

    return roc_auc, pr_auc, log_loss_value


__all__ = ["ClassificationMetrics", "classification_metrics", "evaluate_classifier"]
