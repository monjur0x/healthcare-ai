"""
FastAPI application factory for the Healthcare AI backend.

``create_app`` wires the analysis service and optional bearer-token
authentication into app state and registers exception handlers that map
``APIError`` subclasses to HTTP responses. The module-level ``app`` is
the uvicorn entry point: ``uvicorn api.main:app``.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import APISettings
from .config import settings as default_settings
from .exceptions import APIError
from .routes import router
from .schemas import HealthResponse
from .services import AnalysisService


def create_app(
    cfg: APISettings | None = None,
    service: AnalysisService | None = None,
) -> FastAPI:
    """
    Build and configure the FastAPI application.

    Parameters
    ----------
    cfg : APISettings | None
        Settings to use; defaults to the module-level ``settings``.
    service : AnalysisService | None
        Analysis service to attach; when None, built from ``cfg``.

    Returns
    -------
    FastAPI
        The configured application.
    """

    cfg = cfg or default_settings
    service = service or AnalysisService.from_settings(cfg)

    app = FastAPI(title=cfg.APP_NAME, version=cfg.APP_VERSION, debug=cfg.DEBUG)
    app.state.analysis_service = service
    app.state.api_token = cfg.API_TOKEN

    if cfg.CORS_ALLOW_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                origin.strip() for origin in cfg.CORS_ALLOW_ORIGINS.split(",")
            ],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    _register_exception_handlers(app)
    app.include_router(router)

    @app.get("/", response_model=HealthResponse, tags=["meta"])
    def root() -> HealthResponse:
        """Return server metadata."""
        return HealthResponse(
            status="running", name=cfg.APP_NAME, version=cfg.APP_VERSION
        )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Return a liveness check."""
        return HealthResponse(
            status="healthy", name=cfg.APP_NAME, version=cfg.APP_VERSION
        )

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """
    Map API-layer exceptions to JSON error responses.

    Parameters
    ----------
    app : FastAPI
        Application to register handlers on.
    """

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, error: APIError) -> JSONResponse:
        """Serialize an ``APIError`` to a JSON response."""
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": {"code": error.code, "message": error.message}},
        )


app = create_app()

__all__ = ["app", "create_app"]
