"""
Global configuration for preprocessing.

Every preprocessing module should read settings from here instead of
hardcoding values.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PreprocessingSettings(BaseSettings):
    """
    Configuration used throughout preprocessing.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PREPROCESS_",
        case_sensitive=False,
    )

    ##########################################
    # General
    ##########################################

    RANDOM_SEED: int = 42

    ##########################################
    # CSV
    ##########################################

    MAX_MISSING_RATIO: float = 0.30

    REMOVE_DUPLICATES: bool = True

    ENABLE_FEATURE_ENGINEERING: bool = True

    SCALER: str = "standard"

    ##########################################
    # Image
    ##########################################

    IMAGE_WIDTH: int = 224

    IMAGE_HEIGHT: int = 224

    IMAGE_CHANNELS: int = 3

    NORMALIZE_IMAGES: bool = True

    ##########################################
    # Supported Files
    ##########################################

    SUPPORTED_IMAGE_TYPES: tuple = (
        ".png",
        ".jpg",
        ".jpeg",
        ".dcm",
    )

    SUPPORTED_CSV_TYPES: tuple = (".csv",)

    ##########################################
    # Logging
    ##########################################

    LOG_LEVEL: str = "INFO"

    LOG_DIR: Path = Path("logs")


settings = PreprocessingSettings()
