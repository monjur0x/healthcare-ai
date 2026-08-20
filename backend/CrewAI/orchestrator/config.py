"""
Configuration for the CrewAI orchestration module.

Only orchestration settings belong here. Model and retrieval settings
stay in their owning modules (``models/config.py``, ``rag/config.py``).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class CrewSettings(BaseSettings):
    """
    Orchestration settings.

    Environment variables use the ``CREW_`` prefix, e.g.
    ``CREW_LLM_MODEL`` or ``CREW_RISK_LOW_THRESHOLD``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CREW_",
        case_sensitive=False,
        extra="ignore",
    )

    ##########################################
    # LLM orchestration (optional)
    ##########################################

    LLM_PROVIDER: str = "google"

    LLM_MODEL: str = "gemini-3.7-flash"

    LLM_API_KEY: str = ""

    LLM_TEMPERATURE: float = 0.3

    LLM_MAX_ITERATIONS: int = 10

    LLM_BASE_URL: str = ""

    CREW_VERBOSE: bool = False

    CREW_MEMORY: bool = False

    ##########################################
    # Retrieval wiring
    ##########################################

    RAG_TOP_K: int = 3

    ##########################################
    # Risk assessment
    ##########################################

    RISK_LOW_THRESHOLD: float = 0.3

    RISK_MEDIUM_THRESHOLD: float = 0.6

    MARKER_THRESHOLDS: dict[str, float] = {
        "age": 65.0,
        "bmi": 30.0,
        "blood_pressure_systolic": 140.0,
        "glucose": 126.0,
        "cholesterol": 240.0,
    }


settings = CrewSettings()

__all__ = ["CrewSettings", "settings"]
