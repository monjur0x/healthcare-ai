"""
Healthcare AI Dashboard (Streamlit).

A thin view layer over the FastAPI backend. All reasoning happens
server-side (``backend/api`` -> CrewAI clinical crew); this app only
sends requests and renders responses.

Run from the repository root (or from ``frontend/``):

    streamlit run frontend/streamlit_app.py

Configure the backend origin and optional bearer token in the sidebar.
"""

from __future__ import annotations

import json

import httpx
import streamlit as st
from dashboard.client import APIConfig, HealthcareAPIClient, HealthcareAPIError

DEFAULT_FEATURES = '{"glucose": 148.0, "bmi": 27.3, "age": 54.0}'
DEFAULT_MARKERS = '{"glucose": 148.0, "bmi": 27.3}'


def build_client(base_url: str, api_token: str) -> HealthcareAPIClient:
    """
    Build a cached API client for the given sidebar configuration.

    Parameters
    ----------
    base_url : str
        Backend origin.
    api_token : str
        Optional bearer token.

    Returns
    -------
    HealthcareAPIClient
        Configured client.
    """

    @st.cache_resource
    def _cached(base: str, token: str) -> HealthcareAPIClient:
        return HealthcareAPIClient(APIConfig(base_url=base, api_token=token))

    return _cached(base_url, api_token)


def parse_json_field(raw: str, field_name: str) -> dict:
    """
    Parse a JSON text-area value into a dict, showing errors inline.

    Parameters
    ----------
    raw : str
        Raw textarea content.
    field_name : str
        Human-readable field name for the error message.

    Returns
    -------
    dict
        Parsed mapping (empty on invalid input).
    """

    raw = raw.strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        st.error(f"Invalid JSON for {field_name}: {error}")
        return {}
    if not isinstance(value, dict):
        st.error(f"{field_name} must be a JSON object.")
        return {}
    return {str(key): value for key, value in value.items()}


def parse_lines(raw: str) -> list[str]:
    """Split a text area into non-empty stripped lines."""
    return [line.strip() for line in raw.splitlines() if line.strip()]


def render_prediction(prediction: dict) -> None:
    """Render a ``PredictionResult`` payload."""
    st.subheader("Prediction")
    left, middle, right = st.columns(3)
    left.metric("Predicted class", prediction.get("predicted_class"))
    middle.metric("Confidence", f"{prediction.get('confidence', 0.0):.1%}")
    right.metric("Model", prediction.get("model_name"))
    probabilities = prediction.get("probabilities", {})
    if probabilities:
        st.bar_chart(
            {str(label): float(value) for label, value in probabilities.items()}
        )


def render_risk(risk: dict | None) -> None:
    """Render a ``RiskResult`` payload."""
    if risk is None:
        return
    level = risk.get("risk_level", "unknown")
    if level == "high":
        st.error(f"Risk level: {level.upper()} (score {risk.get('risk_score')})")
    elif level == "medium":
        st.warning(f"Risk level: {level.upper()} (score {risk.get('risk_score')})")
    else:
        st.success(f"Risk level: {level.upper()} (score {risk.get('risk_score')})")
    if risk.get("risk_factors"):
        st.markdown("**Risk factors**")
        for factor in risk["risk_factors"]:
            st.write(f"- {factor}")
    schedule = risk.get("monitoring_schedule") or []
    if schedule:
        st.markdown("**Monitoring schedule**")
        st.dataframe(schedule, width="stretch")


def render_evidence(evidence: list[dict]) -> None:
    """Render retrieved evidence items."""
    for item in evidence:
        with st.expander(
            f"[{item.get('document_id')}] {item.get('source')} "
            f"(score {item.get('score', 0.0):.3f})"
        ):
            st.progress(min(max(float(item.get("score", 0.0)), 0.0), 1.0))
            st.write(item.get("text") or "")


def run_analysis_tab(client: HealthcareAPIClient) -> None:
    """Render the clinical analysis form and report."""
    st.header("Clinical Analysis")
    st.caption(
        "Runs the backend clinical crew: prediction -> risk -> RAG evidence -> report."
    )

    with st.form("analysis_form"):
        patient_col, input_col = st.columns(2)
        with patient_col:
            patient_name = st.text_input("Patient name", value="Patient")
            patient_id = st.text_input("Patient ID", value="patient-1")
            patient_age = st.number_input(
                "Age", min_value=0, max_value=120, value=45, step=1
            )
            notes = st.text_area("Clinical notes", value="", height=80)
        with input_col:
            input_type = st.selectbox("Input type", options=["csv", "image"], index=0)
            features = st.text_area(
                "Features (JSON)", value=DEFAULT_FEATURES, height=110
            )
            markers = st.text_area(
                "Clinical markers (JSON)", value=DEFAULT_MARKERS, height=110
            )

        recommendations = st.text_area(
            "Recommendations (one per line)",
            value="Review the report with a licensed physician before acting.",
            height=80,
        )
        submitted = st.form_submit_button("Run analysis")

    if submitted:
        feature_map = parse_json_field(features, "features")
        marker_map = parse_json_field(markers, "markers")
        patient = {
            "name": patient_name,
            "id": patient_id,
            "age": int(patient_age) if patient_age else None,
            "notes": notes,
        }
        try:
            report = client.analyze(
                patient=patient,
                features=feature_map,
                markers=marker_map or None,
                recommendations=parse_lines(recommendations),
                input_type=input_type,
            )
        except (HealthcareAPIError, httpx.HTTPError) as error:
            st.error(str(error))
            return

        st.divider()
        st.subheader(f"Report — {patient_id}")
        st.write(report.get("patient_summary") or "")
        render_prediction(report.get("prediction") or {})
        render_risk(report.get("risk"))
        if report.get("evidence"):
            st.subheader("Evidence")
            render_evidence(report["evidence"])
        if report.get("recommendations"):
            st.subheader("Recommendations")
            for recommendation in report["recommendations"]:
                st.write(f"- {recommendation}")
        st.caption(report.get("limitations") or "")
        st.download_button(
            "Download report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"clinical_report_{patient_id}.json",
            mime="application/json",
        )


def run_prediction_tab(client: HealthcareAPIClient) -> None:
    """Render the single-row prediction form."""
    st.header("Prediction")
    st.caption("Classify one feature row with the backend model.")
    with st.form("predict_form"):
        features = st.text_area("Features (JSON)", value=DEFAULT_FEATURES, height=160)
        submitted = st.form_submit_button("Predict")
    if submitted:
        feature_map = parse_json_field(features, "features")
        try:
            prediction = client.predict(feature_map)
        except (HealthcareAPIError, httpx.HTTPError) as error:
            st.error(str(error))
            return
        render_prediction(prediction)


def run_retrieval_tab(client: HealthcareAPIClient) -> None:
    """Render the evidence retrieval form."""
    st.header("Evidence Retrieval")
    st.caption("Query the RAG knowledge base for supporting evidence.")
    with st.form("retrieval_form"):
        query = st.text_input(
            "Query",
            value="clinical evidence and management for diabetes",
        )
        top_k = st.slider("Top-k", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("Retrieve")
    if submitted:
        try:
            evidence = client.retrieve(query, top_k=top_k)
        except (HealthcareAPIError, httpx.HTTPError) as error:
            st.error(str(error))
            return
        if not evidence:
            st.info("No evidence retrieved for this query.")
        else:
            st.subheader(f"{len(evidence)} item(s) retrieved")
            render_evidence(evidence)


def run_info_tab(client: HealthcareAPIClient) -> None:
    """Render backend metadata and endpoint reference."""
    st.header("Backend status")
    try:
        health = client.health()
        st.success("Backend reachable")
        st.json(health)
    except (HealthcareAPIError, httpx.HTTPError) as error:
        st.error(f"Backend unreachable: {error}")

    st.subheader("Endpoints")
    st.dataframe(
        [
            {"method": "GET", "path": "/health", "description": "Liveness check"},
            {
                "method": "POST",
                "path": "/api/v1/predict",
                "description": "Classify a feature row",
            },
            {
                "method": "POST",
                "path": "/api/v1/retrieve",
                "description": "Retrieve evidence",
            },
            {
                "method": "POST",
                "path": "/api/v1/analyze",
                "description": "Full clinical analysis",
            },
        ],
        width="stretch",
    )


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(page_title="Healthcare AI Dashboard", layout="wide")

    with st.sidebar:
        st.title("Healthcare AI")
        base_url = st.text_input("Backend URL", value="http://localhost:8000")
        api_token = st.text_input("API token (optional)", value="", type="password")
        client = build_client(base_url, api_token)
        try:
            health = client.health()
            st.success(f"Connected — {health.get('name')} v{health.get('version')}")
        except (HealthcareAPIError, httpx.HTTPError):
            st.warning("Backend not reachable")

    tab_analyze, tab_predict, tab_retrieve, tab_info = st.tabs(
        ["Clinical Analysis", "Prediction", "Evidence Retrieval", "Info"]
    )
    with tab_analyze:
        run_analysis_tab(client)
    with tab_predict:
        run_prediction_tab(client)
    with tab_retrieve:
        run_retrieval_tab(client)
    with tab_info:
        run_info_tab(client)


if __name__ == "__main__":
    main()
