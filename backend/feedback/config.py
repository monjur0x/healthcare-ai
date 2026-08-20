"""
Configuration for the feedback / retrain loop module.

Only feedback-loop settings belong here. The feedback store persists
clinician-confirmed labels for past analyses so the model can be
retrained on real-world corrections once enough samples accumulate.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class FeedbackSettings(BaseSettings):
    """
    Feedback store settings.

    Environment variables use the ``FEEDBACK_`` prefix, e.g.
    ``FEEDBACK_DB_PATH`` or ``FEEDBACK_RETRAIN_THRESHOLD``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FEEDBACK_",
        case_sensitive=False,
        extra="ignore",
    )

    #: Path to the SQLite feedback store database.
    DB_PATH: str = "artifacts/feedback.db"

    #: Minimum number of pending feedback samples required before a
    #: retrain is allowed for a preset.
    RETRAIN_THRESHOLD: int = 5

    #: Whether the retrain step should be allowed at all. Set to False
    #: in read-only / demo deployments to keep feedback recording but
    #: disable automated retraining.
    RETRAIN_ENABLED: bool = True


settings = FeedbackSettings()

__all__ = ["FeedbackSettings", "settings"]
