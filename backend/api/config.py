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
        extra="ignore",
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

    #: Path to a persisted ``ImageClassifier`` artifact (torch).
    #: Empty disables image-based analysis until configured.
    IMAGE_MODEL_PATH: str = ""

    #: Path to a directory of ``.txt`` / ``.md`` knowledge documents for
    #: RAG. Empty uses the repository's bundled medical corpus
    #: (``backend/rag/corpus/``).
    CORPUS_DIR: str = ""

    #: Base directory where ``/api/v1/train`` writes trained model
    #: artifacts (one sub-directory per dataset).
    ARTIFACTS_DIR: str = "artifacts"

    #: Base directory for preset datasets (``diabetes.csv``, ...). When
    #: empty, falls back to the ``DATASET_DIR`` environment variable and
    #: then the current working directory.
    DATASET_DIR: str = ""

    ##########################################
    # Security
    ##########################################

    #: Optional bearer token. When set, ``/api/v1`` routes require an
    #: ``Authorization: Bearer <token>`` header.
    API_TOKEN: str = ""

    CORS_ALLOW_ORIGINS: str = "*"


settings = APISettings()

__all__ = ["APISettings", "settings"]
