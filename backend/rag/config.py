"""
Global configuration for the RAG (retrieval) module.

Every RAG module should read settings from here instead of hardcoding
values.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGSettings(BaseSettings):
    """
    Configuration used throughout retrieval.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    RANDOM_SEED: int = 42

    ##########################################
    # Chunking
    ##########################################

    CHUNK_SIZE: int = 512

    CHUNK_OVERLAP: int = 64

    ##########################################
    # Embedding
    ##########################################

    EMBEDDING_MODEL: str = "tfidf"

    MAX_FEATURES: int = 5000

    ##########################################
    # Retrieval
    ##########################################

    TOP_K: int = 5

    SIMILARITY_METRIC: str = "cosine"


settings = RAGSettings()

__all__ = ["RAGSettings", "settings"]
