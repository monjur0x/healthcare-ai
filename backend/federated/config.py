"""
Configuration for the federated learning module.

Server-level settings for the distributed Flower deployment and the
hospital data layer. Environment variables use the ``FED_`` prefix,
e.g. ``FED_REGISTRY_PATH`` or ``FED_SERVER_ADDRESS``.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class FederationSettings(BaseSettings):
    """
    Distributed federation settings.

    Environment variables use the ``FED_`` prefix (e.g.
    ``FED_REGISTRY_PATH``). Falls back to the shared ``DATASET_DIR``
    environment variable for hospital dataset sources.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FED_",
        case_sensitive=False,
        extra="ignore",
    )

    #: gRPC address the Flower server binds to (e.g. ``0.0.0.0:8080``).
    SERVER_ADDRESS: str = "0.0.0.0:8080"

    #: Path to the SQLite model registry.
    REGISTRY_PATH: str = "artifacts/federation.db"

    #: Root directory holding per-hospital local datasets.
    HOSPITALS_DIR: str = "data/hospitals"

    #: Base directory for preset datasets (diabetes.csv, ...). When
    #: empty, falls back to the ``DATASET_DIR`` environment variable and
    #: then the current working directory.
    DATASET_DIR: str = ""

    #: Root directory where trained global model artifacts are written.
    ARTIFACTS_DIR: str = "artifacts"

    #: Fixed seed for hospital data partitioning and secure aggregation.
    SEED: int = 42

    #: Wall-clock cap (seconds) for the orchestrated distributed-training
    #: subprocess launched through the API. A hung Flower server fails
    #: the request instead of blocking the API worker forever.
    SUBPROCESS_TIMEOUT: int = 1800

    #: Enable TLS for gRPC connections (default False).
    TLS_ENABLED: bool = False

    #: Path to the CA certificate PEM file (required when TLS_ENABLED=true).
    TLS_CA_CERT: str = ""

    #: Path to the server certificate PEM file (required when TLS_ENABLED=true).
    TLS_SERVER_CERT: str = ""

    #: Path to the server private key PEM file (required when TLS_ENABLED=true).
    TLS_SERVER_KEY: str = ""

    #: Path to the client certificate PEM file (optional, for mutual TLS).
    TLS_CLIENT_CERT: str = ""

    #: Path to the client private key PEM file (optional, for mutual TLS).
    TLS_CLIENT_KEY: str = ""


settings = FederationSettings()

__all__ = ["FederationSettings", "settings"]
