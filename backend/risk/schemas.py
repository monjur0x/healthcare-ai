"""
Pydantic schemas for the risk history / monitoring module.

Describes stored risk records, trend analysis, and escalation alerts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]


class RiskHistoryRecord(BaseModel):
    """
    A single persisted risk assessment for a patient.

    Parameters
    ----------
    id : int
        Store row id.
    patient_id : str
        Patient study id.
    preset : str
        Dataset preset used for the analysis.
    risk_score : float
        Risk score in [0, 1] (probability of positive class).
    risk_level : RiskLevel
        Risk level: "low" / "medium" / "high".
    prediction : int | None
        Predicted class (0/1).
    confidence : float | None
        Model confidence.
    markers : dict[str, float] | None
        Raw clinical markers used for risk assessment.
    created_at : datetime
        When the analysis was performed.
    """

    id: int
    patient_id: str
    preset: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    prediction: int | None = None
    confidence: float | None = None
    markers: dict[str, float] | None = None
    created_at: datetime


class RiskTrend(BaseModel):
    """
    Computed risk trend for a patient over recent analyses.

    Parameters
    ----------
    patient_id : str
        Patient study id.
    preset : str
        Dataset preset.
    recent_scores : list[float]
        Risk scores (newest first).
    trend_direction : Literal["improving", "stable", "worsening"]
        Whether risk is decreasing, flat, or increasing.
    slope : float
        Linear regression slope (score change per analysis).
    avg_score : float
        Average of recent scores.
    latest_score : float
        Most recent risk score.
    latest_level : RiskLevel
        Most recent risk level.
    escalation_alert : bool
        True if the latest increase exceeds ESCALATION_THRESHOLD.
    n_points : int
        Number of data points in the trend.
    """

    patient_id: str
    preset: str
    recent_scores: list[float]
    trend_direction: Literal["improving", "stable", "worsening"]
    slope: float
    avg_score: float
    latest_score: float
    latest_level: RiskLevel
    escalation_alert: bool = False
    n_points: int


class RiskHistorySummary(BaseModel):
    """
    Summary of risk history for one patient-preset combination.

    Parameters
    ----------
    patient_id : str
        Patient study id.
    preset : str
        Dataset preset.
    total_analyses : int
        Total number of stored analyses.
    trend : RiskTrend | None
        Computed trend (None if insufficient data).
    latest : RiskHistoryRecord | None
        Most recent analysis record.
    """

    patient_id: str
    preset: str
    total_analyses: int
    trend: RiskTrend | None = None
    latest: RiskHistoryRecord | None = None


class RiskHistoryResponse(BaseModel):
    """
    Response for risk history queries across patients/presets.

    Parameters
    ----------
    summaries : list[RiskHistorySummary]
        Per-patient-preset summaries.
    alert_count : int
        Number of patients with active escalation alerts.
    """

    summaries: list[RiskHistorySummary] = Field(default_factory=list)
    alert_count: int = 0


class EscalationAlert(BaseModel):
    """
    An escalation alert for a patient whose risk has worsened.

    Parameters
    ----------
    patient_id : str
        Patient study id.
    preset : str
        Dataset preset.
    previous_score : float
        Risk score from the previous analysis.
    current_score : float
        Current risk score.
    delta : float
        Score increase (current - previous).
    threshold : float
        Configured escalation threshold.
    timestamp : datetime
        When the alert was generated.
    """

    patient_id: str
    preset: str
    previous_score: float
    current_score: float
    delta: float
    threshold: float
    timestamp: datetime


__all__ = [
    "EscalationAlert",
    "RiskHistoryRecord",
    "RiskHistoryResponse",
    "RiskHistorySummary",
    "RiskTrend",
]
