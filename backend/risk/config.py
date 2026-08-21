"""
Configuration for the risk history / monitoring module.

Tracks patient risk levels over time for longitudinal monitoring
and trend-based alerting.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskHistorySettings(BaseSettings):
    """
    Risk history store settings.

    Environment variables use the ``RISK_HISTORY_`` prefix, e.g.
    ``RISK_HISTORY_DB_PATH`` or ``RISK_HISTORY_TREND_WINDOW``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RISK_HISTORY_",
        case_sensitive=False,
        extra="ignore",
    )

    #: Path to the SQLite risk history database.
    DB_PATH: str = "artifacts/risk_history.db"

    #: Number of recent analyses to consider for trend computation.
    TREND_WINDOW: int = 5

    #: Threshold for risk score increase that triggers an alert
    #: (absolute difference, e.g., 0.2 = 20 percentage points).
    ESCALATION_THRESHOLD: float = 0.2

    #: Minimum number of data points required before trend analysis.
    MIN_TREND_POINTS: int = 3

    #: Whether automated escalation alerts are enabled.
    ALERTS_ENABLED: bool = True


settings = RiskHistorySettings()

__all__ = ["RiskHistorySettings", "settings"]
