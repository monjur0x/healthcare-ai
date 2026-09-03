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

    #: Sentence-transformer model name used when ``EMBEDDING_MODEL ==
    #: "sentence-transformer"``. Keep it small and CPU-friendly.
    SENTENCE_TRANSFORMER_MODEL: str = "BAAI/bge-small-en-v1.5"

    ##########################################
    # Retrieval
    ##########################################

    TOP_K: int = 5

    SIMILARITY_METRIC: str = "cosine"

    #: Minimum best cosine similarity for a generated-answer sentence to
    #: count as grounded in retrieved context (rag.metrics.faithfulness).
    #: Calibrated for the default TF-IDF embedder; raise it (~0.5) for
    #: dense embedders.
    FAITHFULNESS_THRESHOLD: float = 0.3

    ##########################################
    # Vector store
    ##########################################

    #: Vector store backend: ``"memory"`` (in-process NumPy) or
    #: ``"chroma"`` (persistent ChromaDB).
    VECTOR_STORE: str = "memory"

    #: Directory for the persistent ChromaDB collection when
    #: ``VECTOR_STORE == "chroma"``. Empty means a fresh ephemeral
    #: collection per process.
    CHROMA_PERSIST_DIR: str = ""

    #: ChromaDB collection name for the persistent store.
    CHROMA_COLLECTION: str = "healthcare_rag"


settings = RAGSettings()

__all__ = ["RAGSettings", "settings"]
