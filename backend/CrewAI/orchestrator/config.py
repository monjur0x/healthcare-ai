"""
Configuration for the CrewAI orchestration module.

Only orchestration settings belong here. Model and retrieval settings
stay in their owning modules (``models/config.py``, ``rag/config.py``).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CrewSettings(BaseSettings):
    """
    Orchestration settings.

    Environment variables use the ``CREW_`` prefix, e.g.
    ``CREW_LLM_MODEL``. Fields whose names carry their own prefix bind
    the plain documented name instead (e.g. ``RISK_LOW_THRESHOLD``,
    not ``CREW_RISK_LOW_THRESHOLD``).
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

    # NOTE: the attribute names below carry their own prefix for
    # readability at call sites (``settings.RISK_...``); the
    # ``validation_alias`` entries bind the documented single-prefix
    # environment names, since ``env_prefix="CREW_"`` would otherwise
    # demand doubled names like ``CREW_CREW_VERBOSE``.
    CREW_VERBOSE: bool = Field(default=False, validation_alias="CREW_VERBOSE")

    CREW_MEMORY: bool = Field(default=False, validation_alias="CREW_MEMORY")

    ##########################################
    # Retrieval wiring
    ##########################################

    # NOTE: the retrieval default for the rag package itself lives in
    # rag.config (RAG_TOP_K, default 5); this crew-side default (used
    # when agent/RAG calls omit top_k) binds the distinct CREW_RAG_TOP_K
    # variable so the two knobs never collide.
    RAG_TOP_K: int = Field(default=3, validation_alias="CREW_RAG_TOP_K")

    ##########################################
    # Risk assessment
    ##########################################

    RISK_LOW_THRESHOLD: float = Field(
        default=0.3, validation_alias="RISK_LOW_THRESHOLD"
    )

    RISK_MEDIUM_THRESHOLD: float = Field(
        default=0.6, validation_alias="RISK_MEDIUM_THRESHOLD"
    )

    # Scale for the marker-evidence component of the risk score. Each
    # elevated marker contributes clamp(value/threshold - 1, 0, 1);
    # the score takes max(model P(disease), weight * max elevation).
    # 0.5 means markers alone can reach the medium band (but never
    # high), while severe elevations reliably lift a low model
    # probability so the score matches the reported risk factors.
    RISK_MARKER_WEIGHT: float = Field(
        default=0.5, validation_alias="RISK_MARKER_WEIGHT"
    )

    MARKER_THRESHOLDS: dict[str, float] = {
        "age": 65.0,
        "bmi": 30.0,
        "blood_pressure_systolic": 140.0,
        "glucose": 126.0,
        "cholesterol": 240.0,
    }


settings = CrewSettings()

__all__ = ["CrewSettings", "settings"]
