"""
Risk history / monitoring module.

Persists risk assessments from clinical analyses
(:mod:`risk.store`) and exposes trend analysis and escalation alerts
used by the API and n8n orchestration.
"""

from .config import RiskHistorySettings, settings
from .schemas import (
    EscalationAlert,
    RiskHistoryRecord,
    RiskHistoryResponse,
    RiskHistorySummary,
    RiskTrend,
)
from .store import RiskHistoryStore, RiskHistoryStoreError

__all__ = [
    "EscalationAlert",
    "RiskHistoryRecord",
    "RiskHistoryResponse",
    "RiskHistorySettings",
    "RiskHistoryStore",
    "RiskHistoryStoreError",
    "RiskHistorySummary",
    "RiskTrend",
    "settings",
]
