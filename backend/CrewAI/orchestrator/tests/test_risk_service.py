"""
Tests for the deterministic risk assessment service.
"""

from __future__ import annotations

import pytest

from CrewAI.orchestrator.exceptions import RiskToolError
from CrewAI.orchestrator.schemas import PredictionResult
from CrewAI.orchestrator.services import assess_risk


def make_prediction(confidence: float) -> PredictionResult:
    return PredictionResult(
        predicted_class="diabetes",
        probabilities={"healthy": round(1 - confidence, 4), "diabetes": confidence},
        confidence=confidence,
        model_name="logistic",
    )


def test_low_risk_below_threshold() -> None:
    result = assess_risk(make_prediction(0.10))
    assert result.risk_level == "low"
    assert result.risk_score == pytest.approx(0.10)


def test_medium_risk_between_thresholds() -> None:
    assert assess_risk(make_prediction(0.45)).risk_level == "medium"


def test_high_risk_above_threshold() -> None:
    assert assess_risk(make_prediction(0.85)).risk_level == "high"


def test_markers_add_factors() -> None:
    result = assess_risk(
        make_prediction(0.5),
        markers={"glucose": 150.0, "bmi": 24.0},
    )
    assert any("glucose" in factor for factor in result.risk_factors)
    assert not any("bmi" in factor for factor in result.risk_factors)


def test_non_numeric_marker_raises() -> None:
    with pytest.raises(RiskToolError):
        assess_risk(make_prediction(0.5), markers={"glucose": "high"})


def test_monitoring_schedule_by_level() -> None:
    assert assess_risk(make_prediction(0.85)).monitoring_schedule
    assert assess_risk(make_prediction(0.10)).monitoring_schedule
