"""Application configuration using environment variables."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Healthcare AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # LLM Configuration (Gemini)
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "medical_knowledge"
    QDRANT_API_KEY: str = ""

    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # File Upload
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/png", "image/dicom"]
    ALLOWED_CSV_TYPES: list[str] = ["text/csv", "application/vnd.ms-excel"]

    # Risk Thresholds
    RISK_LOW_THRESHOLD: float = 0.3
    RISK_MEDIUM_THRESHOLD: float = 0.6
    RISK_HIGH_THRESHOLD: float = 0.8

    # Crew Configuration
    CREW_VERBOSE: bool = True
    CREW_MAX_ITERATIONS: int = 10
    CREW_MEMORY: bool = True

    # Federated Learning Configuration
    FL_NUM_HOSPITALS: int = 4
    FL_NUM_ROUNDS: int = 10
    FL_LOCAL_EPOCHS: int = 4
    FL_BATCH_SIZE: int = 64
    FL_LEARNING_RATE: float = 0.001
    FL_CLIENT_FRACTION: float = 1.0
    FL_SEED: int = 42
    FL_ARTIFACT_DIR: str = "artifacts"
    FL_DATA_DIR: str = "data"
    FL_MODEL_TYPE: str = "mlp"  # mlp | xgboost | cnn

    # Differential Privacy Configuration
    DP_ENABLED: bool = True
    DP_EPSILON_TARGET: float = 4.0
    DP_DELTA: float = 1e-5
    DP_MAX_GRAD_NORM: float = 1.0
    DP_NOISE_MULTIPLIER: float = 1.1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
