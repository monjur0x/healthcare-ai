"""
API-layer exceptions.

The service layer catches domain exceptions (prediction, risk,
retrieval, orchestration) at its boundary and re-raises them as typed
``APIError`` subclasses so routes and handlers never depend on domain
details. Handlers in ``api/main.py`` map each subclass to an HTTP status.
"""


class APIError(Exception):
    """
    Base exception for API-layer failures.

    Attributes
    ----------
    message : str
        Human-readable description of the failure.
    """

    status_code: int = 500
    code: str = "api_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ServiceUnavailableError(APIError):
    """Raised when a required dependency (model / RAG) is not configured."""

    status_code = 503
    code = "service_unavailable"


class InvalidInputError(APIError):
    """Raised when a request cannot be processed (invalid or inconsistent input)."""

    status_code = 422
    code = "invalid_input"


class AuthenticationError(APIError):
    """Raised when a required bearer token is missing or invalid."""

    status_code = 401
    code = "unauthorized"


class NotFoundError(APIError):
    """Raised when a requested resource does not exist."""

    status_code = 404
    code = "not_found"


__all__ = [
    "APIError",
    "AuthenticationError",
    "InvalidInputError",
    "NotFoundError",
    "ServiceUnavailableError",
]
