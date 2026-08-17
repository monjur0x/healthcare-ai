"""
Thin HTTP client for the Healthcare AI FastAPI backend.

The client performs no reasoning: it serializes requests, parses
responses, and surfaces API errors as ``HealthcareAPIError``. All
business logic lives in the backend (``backend/api/services.py``) and the
CrewAI clinical crew.
"""

from __future__ import annotations

import base64

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

#: n8n webhook path that drives the end-to-end workflow (train + analyze +
#: store + respond). See ``n8n/healthcare-endtoend.json``.
N8N_ANALYZE_WEBHOOK = "healthcare-endtoend"

#: Path of the n8n health probe endpoint (plain ``200 OK`` when reachable).
N8N_HEALTH_PATH = "healthz"


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

    def model_info(self) -> dict[str, Any]:
        """
        Describe the configured prediction models.

        Returns
        -------
        dict[str, Any]
            ``ModelInfo`` payload with ``available``, ``model_type``,
            ``model_name``, ``classes``, ``feature_names``, and ``preset``.
        """
        return self._get_json("/api/v1/model")

    def presets(self) -> list[dict[str, Any]]:
        """
        Describe the named dataset presets and their feature schemas.

        Returns
        -------
        list[dict[str, Any]]
            ``PresetInfo`` payloads ordered by name, each with
            ``available`` / ``feature_names`` / ``classes``.
        """
        return self._get_json("/api/v1/presets")

    def train(self, preset: str, model: str = "mlp") -> dict[str, Any]:
        """
        Train (or retrain) a preset model and serve it immediately.

        Parameters
        ----------
        preset : str
            Dataset preset name (``"diabetes"`` / ``"heart"`` / ...).
        model : str
            Model family to fit (``"mlp"`` or ``"logistic"``).

        Returns
        -------
        dict[str, Any]
            ``TrainResponse`` payload with artifact path and hold-out
            metrics.
        """
        return self._post_json("/api/v1/train", {"preset": preset, "model": model})

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
        return self._post_json(
            "/api/v1/analyze",
            self._analyze_payload(
                patient, features, markers, recommendations, input_type
            ),
        )

    def analyze_via_n8n(
        self,
        n8n_base_url: str,
        patient: Mapping[str, Any],
        features: Mapping[str, float],
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
        input_type: str = "csv",
        webhook: str = N8N_ANALYZE_WEBHOOK,
        preset: str | None = None,
        train: bool = False,
    ) -> dict[str, Any]:
        """
        Run the clinical analysis through the n8n end-to-end webhook.

        The webhook orchestrates the FastAPI call (optionally training a
        model first) and returns the full clinical report when it succeeds.

        Parameters
        ----------
        n8n_base_url : str
            Origin of the n8n instance, e.g. ``http://localhost:5678``.
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
        webhook : str
            n8n webhook path to call (defaults to the end-to-end workflow).
        preset : str | None
            When given, the workflow trains this preset before analyzing.
        train : bool
            When true (and ``preset`` is set), the workflow trains the
            preset model before the analysis step.

        Returns
        -------
        dict[str, Any]
            ``ClinicalReport`` payload extracted from the webhook response.

        Raises
        ------
        HealthcareAPIError
            If the webhook is unreachable, the workflow reports an error,
            or the response omits the clinical report.
        """
        response = self._client.post(
            f"{n8n_base_url.rstrip('/')}/webhook/{webhook}",
            json=self._analyze_payload(
                patient,
                features,
                markers,
                recommendations,
                input_type,
                preset=preset,
                train=train,
            ),
        )
        self._raise_for_error(response)
        body = response.json()
        if not isinstance(body, dict):
            raise HealthcareAPIError(
                "n8n returned an unexpected response.",
                response.status_code,
                "n8n_error",
            )
        if body.get("status") != "success":
            message = (
                body.get("error_message")
                or body.get("message")
                or "n8n workflow did not complete successfully."
            )
            raise HealthcareAPIError(
                str(message), response.status_code, "n8n_workflow_error"
            )
        report = body.get("report")
        if not isinstance(report, dict):
            raise HealthcareAPIError(
                "n8n response did not include a clinical report.",
                response.status_code,
                "n8n_report_missing",
            )
        return report

    def n8n_health(self, n8n_base_url: str, path: str = N8N_HEALTH_PATH) -> bool:
        """
        Probe an n8n instance health endpoint.

        Parameters
        ----------
        n8n_base_url : str
            Origin of the n8n instance, e.g. ``http://localhost:5678``.
        path : str
            Health endpoint path (default ``healthz``).

        Returns
        -------
        bool
            True when n8n answers with a successful status.
        """
        try:
            response = self._client.get(f"{n8n_base_url.rstrip('/')}/{path}")
        except httpx.HTTPError:
            return False
        return response.is_success

    def _analyze_payload(
        self,
        patient: Mapping[str, Any],
        features: Mapping[str, float],
        markers: Mapping[str, float] | None,
        recommendations: list[str] | None,
        input_type: str,
        preset: str | None = None,
        train: bool = False,
    ) -> dict[str, Any]:
        """Build the shared ``/api/v1/analyze`` request body."""
        payload: dict[str, Any] = {
            "patient": dict(patient),
            "features": dict(features),
            "input_type": input_type,
        }
        if markers is not None:
            payload["markers"] = dict(markers)
        if recommendations is not None:
            payload["recommendations"] = list(recommendations)
        if train:
            payload["train"] = True
        if preset:
            payload["preset"] = preset
        return payload

    def analyze_image(
        self,
        patient: Mapping[str, Any],
        image: bytes,
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run the clinical analysis on an uploaded image and return the report.

        Parameters
        ----------
        patient : Mapping[str, Any]
            Patient context (``name``, ``id``, ``age``, ``notes``).
        image : bytes
            Raw image file bytes (PNG / JPEG).
        markers : Mapping[str, float] | None
            Optional raw clinical markers for the risk assessment.
        recommendations : list[str] | None
            Recommendation strings for the report.

        Returns
        -------
        dict[str, Any]
            ``ClinicalReport`` payload.
        """
        payload: dict[str, Any] = {
            "patient": dict(patient),
            "image": base64.b64encode(image).decode("ascii"),
        }
        if markers is not None:
            payload["markers"] = dict(markers)
        if recommendations is not None:
            payload["recommendations"] = list(recommendations)
        return self._post_json("/api/v1/analyze/image", payload)

    def analyze_csv(
        self,
        patient: Mapping[str, Any],
        csv: bytes,
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
        input_type: str = "csv",
    ) -> dict[str, Any]:
        """
        Run the clinical analysis on an uploaded CSV and return the report.

        The raw CSV bytes are sent as-is; all parsing and preprocessing
        happens on the backend (``preprocessing.csv.CSVPipeline``).

        Parameters
        ----------
        patient : Mapping[str, Any]
            Patient context (``name``, ``id``, ``age``, ``notes``).
        csv : bytes
            Raw UTF-8 CSV file bytes.
        markers : Mapping[str, float] | None
            Optional raw clinical markers for the risk assessment.
        recommendations : list[str] | None
            Recommendation strings for the report.
        input_type : str
            Data modality analyzed (default ``"csv"``).

        Returns
        -------
        dict[str, Any]
            ``ClinicalReport`` payload.
        """
        payload: dict[str, Any] = {
            "patient": dict(patient),
            "csv": base64.b64encode(csv).decode("ascii"),
            "input_type": input_type,
        }
        if markers is not None:
            payload["markers"] = dict(markers)
        if recommendations is not None:
            payload["recommendations"] = list(recommendations)
        return self._post_json("/api/v1/analyze/csv", payload)

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


__all__ = [
    "N8N_ANALYZE_WEBHOOK",
    "N8N_HEALTH_PATH",
    "APIConfig",
    "HealthcareAPIClient",
    "HealthcareAPIError",
]
