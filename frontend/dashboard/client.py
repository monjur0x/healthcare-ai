"""
Thin HTTP client for the Healthcare AI FastAPI backend.

The client performs no reasoning: it serializes requests, parses
responses, and surfaces API errors as ``HealthcareAPIError``. All
business logic lives in the backend (``backend/api/services.py``) and the
CrewAI clinical crew.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx


class HealthcareAPIError(Exception):
    """
    Raised when the backend returns a non-2xx response.

    Attributes
    ----------
    status_code : int
        HTTP status returned by the backend.
    code : str
        Machine error code from the API error detail.
    """

    def __init__(self, message: str, status_code: int, code: str = "api_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class APIConfig:
    """
    Connection settings for the backend.

    Parameters
    ----------
    base_url : str
        Backend origin, e.g. ``http://localhost:8000``.
    api_token : str
        Optional bearer token; sent as ``Authorization: Bearer <token>``
        when non-empty (matches the backend ``API_TOKEN`` setting).
    timeout : float
        Per-request timeout in seconds.
    """

    base_url: str = "http://localhost:8000"
    api_token: str = ""
    timeout: float = 60.0


class HealthcareAPIClient:
    """
    Typed client over the backend REST API.

    Parameters
    ----------
    config : APIConfig
        Connection settings.
    transport : httpx.BaseTransport | None
        Optional transport for tests (e.g. ``httpx.MockTransport``).
    """

    def __init__(
        self,
        config: APIConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config or APIConfig()
        headers = (
            {"Authorization": f"Bearer {self.config.api_token}"}
            if self.config.api_token
            else {}
        )
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout,
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def health(self) -> dict[str, Any]:
        """
        Query the backend health endpoint.

        Returns
        -------
        dict[str, Any]
            ``{status, name, version}`` metadata.
        """
        return self._get_json("/health")

    def predict(self, features: Mapping[str, float]) -> dict[str, Any]:
        """
        Classify a single feature row.

        Parameters
        ----------
        features : Mapping[str, float]
            Feature values keyed by column name.

        Returns
        -------
        dict[str, Any]
            ``PredictionResult`` payload.
        """
        return self._post_json("/api/v1/predict", {"features": dict(features)})

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve evidence chunks for a query.

        Parameters
        ----------
        query : str
            Query text.
        top_k : int | None
            Result count; defaults to the backend RAG top-k.

        Returns
        -------
        list[dict[str, Any]]
            List of ``EvidenceItem`` payloads.
        """
        return self._post_json("/api/v1/retrieve", {"query": query, "top_k": top_k})

    def analyze(
        self,
        patient: Mapping[str, Any],
        features: Mapping[str, float],
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
        input_type: str = "csv",
    ) -> dict[str, Any]:
        """
        Run the clinical analysis and return the report.

        Parameters
        ----------
        patient : Mapping[str, Any]
            Patient context (``name``, ``id``, ``age``, ``notes``).
        features : Mapping[str, float]
            Preprocessed feature row for the prediction step.
        markers : Mapping[str, float] | None
            Raw clinical markers for the risk assessment.
        recommendations : list[str] | None
            Recommendation strings for the report.
        input_type : str
            Data modality analyzed (``"csv"`` / ``"image"`` / ...).

        Returns
        -------
        dict[str, Any]
            ``ClinicalReport`` payload.
        """
        payload: dict[str, Any] = {
            "patient": dict(patient),
            "features": dict(features),
            "input_type": input_type,
        }
        if markers is not None:
            payload["markers"] = dict(markers)
        if recommendations is not None:
            payload["recommendations"] = list(recommendations)
        return self._post_json("/api/v1/analyze", payload)

    def _get_json(self, path: str) -> dict[str, Any]:
        response = self._client.get(path)
        self._raise_for_error(response)
        return response.json()

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        response = self._client.post(path, json=payload)
        self._raise_for_error(response)
        return response.json()

    def _raise_for_error(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        detail: Any = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            detail = body.get("detail")
        message = "Backend request failed"
        code = "api_error"
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("code") or message
            code = detail.get("code") or code
        elif detail:
            message = str(detail)
        raise HealthcareAPIError(message, response.status_code, code)


__all__ = ["APIConfig", "HealthcareAPIClient", "HealthcareAPIError"]
