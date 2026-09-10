"""
Tests for the CrewAI tools (thin wrappers over the services).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from CrewAI.orchestrator.schemas import PredictionResult
from CrewAI.orchestrator.services import assess_risk
from CrewAI.orchestrator.tools import (
    ClinicalReportTool,
    PredictionTool,
    RiskAssessmentTool,
)
from models import TabularClassifier


@pytest.fixture
def model() -> TabularClassifier:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 2))
    y = (x[:, 0] > 0).astype(int)
    return TabularClassifier(model_name="logistic").fit(
        pd.DataFrame(x, columns=["a", "b"]), y
    )


def test_prediction_tool_run(model) -> None:
    tool = PredictionTool(model=model)
    payload = tool._run({"a": 0.5, "b": -0.2})
    assert payload["predicted_class"] in {"0", "1"}
    assert "confidence" in payload


def test_risk_tool_run() -> None:
    prediction = PredictionResult(
        predicted_class="x", probabilities={"x": 0.9, "y": 0.1}, confidence=0.9
    )
    tool = RiskAssessmentTool()
    payload = tool._run(prediction.model_dump(), markers={"glucose": 200.0})
    assert payload["risk_level"] == "high"
    assert any("glucose" in factor for factor in payload["risk_factors"])


def test_report_tool_run() -> None:
    prediction = PredictionResult(
        predicted_class="x", probabilities={"x": 1.0}, confidence=1.0
    )
    risk = assess_risk(prediction)
    tool = ClinicalReportTool()
    payload = tool._run(
        patient={"name": "A", "id": "p1", "age": 40},
        prediction=prediction.model_dump(),
        risk=risk.model_dump(),
    )
    assert payload["patient"]["id"] == "p1"
    assert payload["prediction"]["predicted_class"] == "x"
