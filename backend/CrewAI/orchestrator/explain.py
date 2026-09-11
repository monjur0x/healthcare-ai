"""
Model-derived explanations for the clinical crew (P2.2).

This module replaces the magnitude-sort heuristic ("largest raw feature
value drives the prediction") with attributions derived from the fitted
model itself:

- Tabular models: SHAP values via the ``shap`` package
  (``LinearExplainer`` for logistic regression, ``KernelExplainer`` for
  anything else). Class ``1`` is always the disease class (see
  ``_preset_binary_labels``), so a positive SHAP value pushes *toward* a
  disease prediction and a negative one pushes *away*.
- Image CNNs: Grad-CAM heatmaps from the last convolutional layer via
  native torch hooks (no extra dependency).

Callers that cannot supply a fitted model (or a background sample for
SHAP) must keep using the documented magnitude-sort fallback in
``build_explanation`` — a missing explanation is worse than a heuristic
one, but a heuristic one must never be presented as model-derived.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from preprocessing.csv.scaler import CSVScaler

from .exceptions import ExplanationError

#: Cap on SHAP background rows: KernelExplainer cost scales with the
#: background size, and 25 rows is plenty for a stable reference.
MAX_BACKGROUND_ROWS = 25

#: KernelExplainer evaluations per explained row. Scales mildly with
#: feature count; keeps a single-row explanation to ~1s on CPU.
KERNEL_NSAMPLES_BASE = 100


@dataclass(frozen=True)
class FeatureAttribution:
    """
    SHAP attribution of one feature for the disease class.

    Parameters
    ----------
    feature : str
        Feature name.
    shap_value : float
        SHAP value: positive pushes toward disease, negative away.
    abs_rank : int
        1-based rank by descending |SHAP|.
    direction : Literal["toward", "away"]
        Clinical reading of the sign.
    """

    feature: str
    shap_value: float
    abs_rank: int
    direction: Literal["toward", "away"]


@dataclass(frozen=True)
class TabularAttribution:
    """
    Per-row SHAP attribution for the disease class.

    Parameters
    ----------
    method : str
        ``"shap_linear"`` (exact, logistic) or ``"shap_kernel"``
        (sampling, all other estimators).
    expected_value : float
        Model's baseline output (reference) for the disease class.
    predicted_class : str
        Disease-class label attributed (always class ``"1"``).
    attributions : tuple[FeatureAttribution, ...]
        One entry per feature, ranked by descending |SHAP|.
    """

    method: str
    expected_value: float
    predicted_class: str
    attributions: tuple[FeatureAttribution, ...] = field(default_factory=tuple)

    def top(self, k: int = 3) -> tuple[FeatureAttribution, ...]:
        """Return the top-k attributions by |SHAP|."""
        return self.attributions[: max(k, 0)]

    def text(self, disease: str | None = None, top_k: int = 3) -> str:
        """
        Render the clinical explanation sentence.

        Parameters
        ----------
        disease : str | None
            Disease name for the sentence; generic wording when None.
        top_k : int
            Number of top features to name.

        Returns
        -------
        str
            e.g. ``"SHAP (shap_linear): glucose +0.231 toward ..."``.
        """
        label = (disease or "the condition").replace("_", " ")
        parts = [
            f"{attr.feature} {attr.shap_value:+.3f} {attr.direction}"
            for attr in self.top(top_k)
        ]
        # LinearExplainer explains log-odds; KernelExplainer explains the
        # disease-class probability — label the baseline honestly.
        scale = "log-odds" if self.method == "shap_linear" else "probability"
        return (
            f"SHAP ({self.method}): top drivers of {label} risk — "
            + "; ".join(parts)
            + f". Baseline {scale} {self.expected_value:.3f}."
        )


def _require_binary_disease_model(model: Any) -> list[str]:
    """Validate the model is a fitted binary tabular model; return names."""
    if not getattr(model, "is_fitted", False):
        raise ExplanationError("The tabular model is not fitted.")
    names = list(getattr(model, "feature_names", None) or [])
    if not names:
        raise ExplanationError("The model has no recorded feature columns.")
    classes = [str(label) for label in getattr(model, "classes_", [])]
    if len(classes) != 2:
        raise ExplanationError(
            f"SHAP attribution needs a binary model, got classes {classes}."
        )
    return names


def _align_row(features: Mapping[str, float], names: list[str]) -> dict[str, float]:
    """Order the feature mapping by model column; fail loudly on gaps."""
    missing = [name for name in names if name not in features]
    if missing:
        raise ExplanationError(f"Missing feature values for columns: {missing}.")
    try:
        return {name: float(features[name]) for name in names}
    except (TypeError, ValueError) as error:
        raise ExplanationError(f"Non-numeric feature values: {error}") from error


def _scale_row(model: Any, row: dict[str, float], preprocessed: bool) -> np.ndarray:
    """Apply the model's persisted scaler unless already preprocessed."""
    if preprocessed or getattr(model, "scaler_params", None) is None:
        return np.array([row[name] for name in row], dtype=np.float64)
    try:
        scaled = CSVScaler.from_params(model.scaler_params).transform(
            pd.DataFrame([row])
        )[0]
        return np.array([float(scaled.iloc[0][name]) for name in row], dtype=np.float64)
    except Exception as error:
        raise ExplanationError(f"Feature scaling failed: {error}") from error


def _background_matrix(
    background: pd.DataFrame | np.ndarray, names: list[str]
) -> np.ndarray:
    """Validate and cap the SHAP background sample."""
    if isinstance(background, pd.DataFrame):
        missing = [name for name in names if name not in background.columns]
        if missing:
            raise ExplanationError(f"Background sample lacks columns: {missing}.")
        matrix = background[names].to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(background, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(names):
            raise ExplanationError(
                f"Background must be (n_rows, {len(names)}); got {matrix.shape}."
            )
    if matrix.shape[0] == 0:
        raise ExplanationError("Background sample is empty.")
    return matrix[:MAX_BACKGROUND_ROWS]


def _positive_proba_fn(model: Any) -> Any:
    """Return ``matrix -> P(disease)`` over scaled rows for SHAP."""
    predict = model.predict_proba

    def fn(matrix: np.ndarray) -> np.ndarray:
        proba = np.asarray(predict(np.asarray(matrix, dtype=np.float64)))
        if proba.ndim == 1:
            return proba
        return proba[:, 1]

    return fn


def attribute_tabular(
    model: Any,
    features: Mapping[str, float],
    *,
    background: pd.DataFrame | np.ndarray,
    preprocessed: bool = False,
    seed: int = 42,
    nsamples: int | None = None,
) -> TabularAttribution:
    """
    Attribute a disease prediction to input features with SHAP.

    Parameters
    ----------
    model : Any
        Fitted ``TabularClassifier`` (sklearn) or ``TorchMLPClassifier``.
        ``model_name == "logistic"`` uses the exact ``LinearExplainer``;
        anything else uses the sampling ``KernelExplainer``.
    features : Mapping[str, float]
        Single feature row (raw or preprocessed per ``preprocessed``).
    background : pd.DataFrame | np.ndarray
        Reference sample of *preprocessed* training rows (capped
        internally at ``MAX_BACKGROUND_ROWS``).
    preprocessed : bool
        True when ``features`` already went through the training
        pipeline; False applies the model's persisted scaler.
    seed : int
        Seed for the KernelExplainer sampler (LinearExplainer is exact).
    nsamples : int | None
        KernelExplainer evaluations; defaults to
        ``min(KERNEL_NSAMPLES_BASE + 2 * n_features, 500)``.

    Returns
    -------
    TabularAttribution
        Ranked per-feature SHAP values for the disease class.

    Raises
    ------
    ExplanationError
        If the model, features, or background cannot support SHAP.
    """

    try:
        import shap
    except ImportError as error:
        raise ExplanationError(
            "The 'shap' package is required for model-derived "
            "explanations; install it or use the magnitude-sort fallback."
        ) from error

    names = _require_binary_disease_model(model)
    row = _align_row(features, names)
    scaled = _scale_row(model, row, preprocessed).reshape(1, -1)
    matrix = _background_matrix(background, names)

    estimator = getattr(model, "estimator", None)
    if getattr(model, "model_name", "") == "logistic" and estimator is not None:
        try:
            explainer = shap.LinearExplainer(estimator, matrix)
            values = np.asarray(explainer.shap_values(scaled))
        except Exception as error:
            raise ExplanationError(f"LinearExplainer failed: {error}") from error
        method = "shap_linear"
        expected = explainer.expected_value
    else:
        n_features = len(names)
        draws = (
            min(KERNEL_NSAMPLES_BASE + 2 * n_features, 500)
            if nsamples is None
            else int(nsamples)
        )
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            explainer = shap.KernelExplainer(
                _positive_proba_fn(model), matrix, nsamples=draws
            )
            values = np.asarray(explainer.shap_values(scaled, nsamples=draws))
        except Exception as error:
            raise ExplanationError(f"KernelExplainer failed: {error}") from error
        finally:
            np.random.set_state(state)
        method = "shap_kernel"
        expected = explainer.expected_value

    # Both explainers return one row per explained sample for a scalar
    # disease-class output; flatten the single explained row.
    values = np.asarray(values).reshape(-1)
    if values.shape[0] != len(names):
        raise ExplanationError(
            f"SHAP returned {values.shape[0]} values for {len(names)} features."
        )
    try:
        baseline = float(np.asarray(expected).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        baseline = 0.0

    order = np.argsort(-np.abs(values))
    attributions = tuple(
        FeatureAttribution(
            feature=names[index],
            shap_value=float(values[index]),
            abs_rank=rank + 1,
            direction="toward" if values[index] >= 0 else "away",
        )
        for rank, index in enumerate(order)
    )
    return TabularAttribution(
        method=method,
        expected_value=baseline,
        predicted_class="1",
        attributions=attributions,
    )


def grad_cam_heatmap(
    image_model: Any,
    image: np.ndarray,
    *,
    class_index: int | None = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for an image prediction.

    Hooks the last convolutional layer: channel weights are the
    global-average-pooled gradients of the target logit, and the heatmap
    is the ReLU of the weighted activation sum, upsampled to the input
    resolution and normalized to ``[0, 1]``.

    Parameters
    ----------
    image_model : Any
        Fitted ``ImageClassifier`` exposing ``torch_module``.
    image : np.ndarray
        Preprocessed image, ``(H, W, C)`` channels-last float32 (the same
        convention as ``run_image_prediction``).
    class_index : int | None
        Logit to explain; defaults to the predicted (argmax) class.

    Returns
    -------
    np.ndarray
        ``(H, W)`` float32 heatmap in ``[0, 1]``; larger means the region
        pushed the model harder toward the explained class.

    Raises
    ------
    ExplanationError
        If the model is unfitted, has no conv layer, or grads fail.
    """

    try:
        import torch  # noqa: I001
        from torch import nn
    except ImportError as error:
        raise ExplanationError(
            "PyTorch is required for Grad-CAM explanations."
        ) from error

    if not getattr(image_model, "is_fitted", False):
        raise ExplanationError("The image model is not fitted.")
    module = getattr(image_model, "torch_module", None)
    if module is None:
        raise ExplanationError("The image model exposes no torch module.")
    target = None
    for candidate in module.modules():
        if isinstance(candidate, nn.Conv2d):
            target = candidate
    if target is None:
        raise ExplanationError("The image model has no convolutional layer.")

    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3:
        raise ExplanationError(f"Expected an (H, W, C) image, got shape {array.shape}.")
    height, width, channels = array.shape
    if channels != int(getattr(image_model, "in_channels", channels)):
        raise ExplanationError(
            f"Image has {channels} channels; model expects {image_model.in_channels}."
        )
    device = next(module.parameters()).device
    batch = torch.from_numpy(np.moveaxis(array, -1, 0)).unsqueeze(0).to(device)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_layer: Any, _inputs: Any, output: torch.Tensor) -> None:
        activations["value"] = output.detach()

    def backward_hook(
        _layer: Any, _grad_in: Any, grad_out: tuple[torch.Tensor, ...]
    ) -> None:
        gradients["value"] = grad_out[0].detach()

    handle_f = target.register_forward_hook(forward_hook)
    handle_b = target.register_full_backward_hook(backward_hook)
    was_training = module.training
    try:
        module.eval()
        batch.requires_grad_(True)
        logits = module(batch)
        if class_index is None:
            target_index = int(torch.argmax(logits, dim=1).item())
        else:
            target_index = int(class_index)
            if not 0 <= target_index < logits.shape[1]:
                raise ExplanationError(
                    f"class_index {target_index} out of range "
                    f"for {logits.shape[1]} classes."
                )
        module.zero_grad(set_to_none=True)
        logits[0, target_index].backward()
    except ExplanationError:
        raise
    except Exception as error:
        raise ExplanationError(f"Grad-CAM backward pass failed: {error}") from error
    finally:
        handle_f.remove()
        handle_b.remove()
        if was_training:
            module.train()

    if "value" not in activations or "value" not in gradients:
        raise ExplanationError("Grad-CAM hooks captured no tensors.")
    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations["value"]).sum(dim=1, keepdim=True).clamp(min=0.0)
    cam = cam - cam.min()
    if float(cam.max()) > 0:
        cam = cam / float(cam.max())
    upsampled = nn.functional.interpolate(
        cam, size=(height, width), mode="bilinear", align_corners=False
    )
    return upsampled.squeeze().detach().cpu().numpy().astype(np.float32)


def summarize_grad_cam(heatmap: np.ndarray, class_label: str) -> str:
    """
    Render a one-line clinical summary of a Grad-CAM heatmap.

    Parameters
    ----------
    heatmap : np.ndarray
        ``(H, W)`` normalized heatmap from :func:`grad_cam_heatmap`.
    class_label : str
        Explained class label.

    Returns
    -------
    str
        Peak-quadrant location plus focus coverage above half-maximum.
    """

    array = np.asarray(heatmap, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ExplanationError("Grad-CAM heatmap must be a non-empty (H, W) array.")
    peak = np.unravel_index(int(np.argmax(array)), array.shape)
    vertical = "upper" if peak[0] < array.shape[0] / 2 else "lower"
    horizontal = "left" if peak[1] < array.shape[1] / 2 else "right"
    coverage = float((array > 0.5).mean())
    return (
        f"Grad-CAM for class {class_label}: peak activation in the "
        f"{vertical}-{horizontal} quadrant, "
        f"{coverage:.0%} of the image above half-maximum focus."
    )


__all__ = [
    "KERNEL_NSAMPLES_BASE",
    "MAX_BACKGROUND_ROWS",
    "FeatureAttribution",
    "TabularAttribution",
    "attribute_tabular",
    "grad_cam_heatmap",
    "summarize_grad_cam",
]
