"""
Configuration for the FastAPI module.

Only server-level settings belong here. Prediction, retrieval, and
orchestration settings stay in their owning modules (``models/config.py``,
``rag/config.py``, ``CrewAI/orchestrator/config.py``).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """
    API server settings.

    Environment variables use the ``API_`` prefix, e.g. ``API_MODEL_PATH``
    or ``API_CORPUS_DIR``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="API_",
        case_sensitive=False,
    )

    ##########################################
    # Server metadata
    ##########################################

    APP_NAME: str = "Healthcare AI Backend"

    APP_VERSION: str = "1.0.0"

    DEBUG: bool = False

    ##########################################
    # Analysis wiring
    ##########################################

    #: Path to a persisted ``TabularClassifier`` artifact (joblib).
    #: Empty disables the prediction step until configured.
    MODEL_PATH: str = ""

    #: Path to a directory of ``.txt`` / ``.md`` knowledge documents for
    #: RAG. Empty uses a small built-in medical corpus.
    CORPUS_DIR: str = ""

    ##########################################
    # Security
    ##########################################

    #: Optional bearer token. When set, ``/api/v1`` routes require an
    #: ``Authorization: Bearer <token>`` header.
    API_TOKEN: str = ""

    CORS_ALLOW_ORIGINS: str = "*"


settings = APISettings()

__all__ = ["APISettings", "settings"]
