"""
Configuration for prediction models.

Model hyperparameters are read from environment variables (prefixed
``MODEL_``) or the project ``.env`` file. Values should never be
hardcoded inside model code.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """
    Configuration used throughout the models package.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MODEL_",
        case_sensitive=False,
        extra="ignore",
    )

    ##########################################
    # General
    ##########################################

    RANDOM_SEED: int = 42

    ##########################################
    # Image
    ##########################################

    IMAGE_TRAIN_EPOCHS: int = 10

    IMAGE_TRAIN_BATCH_SIZE: int = 16

    IMAGE_TRAIN_LEARNING_RATE: float = 1e-3

    IMAGE_DEVICE: str = "auto"


settings = ModelSettings()
