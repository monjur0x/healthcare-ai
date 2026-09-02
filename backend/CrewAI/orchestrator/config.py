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

    # Capped low because each iteration is a full LLM round-trip (very
    # slow on free-tier providers); 4 leaves an agent room to
    # self-correct without dominating wall-clock time. Not a correctness
    # requirement.
    LLM_MAX_ITERATIONS: int = 4

    # Cap on completion tokens per agent call; bounds the worst-case
    # latency of a single round-trip on slow/free providers. Keep above
    # ~1k: smaller caps truncate tool-call JSON mid-emit and providers
    # return empty/invalid completions.
    LLM_MAX_TOKENS: int = 1024

    # Per-call timeout in seconds for the provider client; bounds how
    # long one hung request can stall an agent. Must stay above
    # free-tier queue+generation latency (~1-2 min), or every call dies
    # and the crew silently falls back to the deterministic report.
    LLM_TIMEOUT_SECONDS: int = 120

    # Retries per LLM call (with exponential backoff) so shared-pool
    # 429 rate limits wait out their window instead of failing the
    # kickoff.
    LLM_MAX_RETRIES: int = 4

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
