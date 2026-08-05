"""Global-model inference for the federated pipeline (Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings
from . import models
from .data import FEATURE_COLUMNS


def predict_risk(features: dict[str, float] | list[float], model_type: str | None = None) -> dict[str, Any]:
    """Predict clinical risk with the trained global model.

    ``features`` may be a dict of ``{feature: value}`` or a flat list aligned to
    FEATURE_COLUMNS. Returns risk score, risk level, and contributing features.
    """
    model_type = model_type or settings.FL_MODEL_TYPE
    artifact_dir = Path(settings.FL_ARTIFACT_DIR)

    if isinstance(features, dict):
        row = [features.get(col, _default(col)) for col in FEATURE_COLUMNS]
    else:
        row = list(features)

    if len(row) != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} features ({', '.join(FEATURE_COLUMNS)}), got {len(row)}"
        )

    model = models.HealthMLP()
    state_path = artifact_dir / "global_model.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            "No trained global model found. Run federated training first "
            "(POST /api/federated/train)."
        )
    model.load_state_dict(torch_load(state_path))
    model.eval()

    proba = float(models.predict_proba(model, np.array([row], dtype=np.float32))[0])
    return _format_prediction(proba, dict(zip(FEATURE_COLUMNS, row)))


def _format_prediction(proba: float, features: dict[str, float]) -> dict[str, Any]:
    if proba >= 0.7:
        level, category = "high", "High clinical risk - medical intervention recommended"
    elif proba >= 0.4:
        level, category = "medium", "Moderate risk - lifestyle modifications recommended"
    else:
        level, category = "low", "Low risk - continue healthy lifestyle"

    top = sorted(features.items(), key=lambda kv: _risk_contribution(kv[0], kv[1]), reverse=True)[:5]
    return {
        "risk_score": round(proba, 4),
        "risk_percentage": round(proba * 100, 2),
        "risk_level": level,
        "risk_category": category,
        "top_contributing_features": [
            {"feature": name, "value": round(value, 2)} for name, value in top
        ],
        "model": "federated_global",
        "num_features": len(features),
    }


def _risk_contribution(feature: str, value: float) -> float:
    """Heuristic contribution for interpretability (approximates SHAP)."""
    thresholds = {
        "age": (45, 0.5), "bmi": (25, 0.6), "sbp": (130, 0.5), "glucose": (110, 0.7),
        "cholesterol": (200, 0.5), "creatinine": (1.2, 0.7), "hemoglobin": (12, 0.5),
        "spo2": (95, -0.6), "heart_rate": (100, 0.5), "los_days": (7, 0.5),
    }
    if feature not in thresholds:
        return 0.0
    base, weight = thresholds[feature]
    deviation = (value - base) / base if base else 0.0
    if feature == "spo2":
        return max(0.0, -deviation * weight)
    return max(0.0, deviation * weight)


def _default(feature: str) -> float:
    return {
        "age": 45.0, "bmi": 25.0, "sbp": 120.0, "dbp": 80.0, "glucose": 100.0,
        "cholesterol": 200.0, "hdl": 50.0, "heart_rate": 75.0, "creatinine": 1.0,
        "hemoglobin": 14.0, "spo2": 97.0, "temp": 37.0, "wbc": 7.5,
        "platelet": 250.0, "los_days": 3.0,
    }.get(feature, 0.0)


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)