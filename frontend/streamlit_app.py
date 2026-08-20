"""
Healthcare AI — Clinical Decision Support Dashboard (Streamlit).

A thin view layer over the FastAPI backend and the n8n end-to-end
workflow. All reasoning happens server-side (``backend/api`` -> CrewAI
clinical crew); this app only collects doctor-facing inputs, sends
requests, and renders the structured report.

The dashboard is organised around the research workflow:

    Patient Data -> Federated Prediction -> Disease Prediction Agent ->
    RAG Retrieval -> Treatment Agent -> Explainability -> n8n Workflow
    -> Doctor Dashboard

Pages
-----
- Overview — what the system does, what is supported, workflow recap.
- Clinical Assessment — research-defined clinical inputs grouped
  logically (Patient Information / Vital Signs / Clinical Measurements /
  Medical History), one clear **Analyze Patient** action, and inline
  results.
- Imaging — upload -> preview -> analyze -> prediction -> confidence ->
  explanation, when an image model is configured.
- Results — the six research outputs rendered doctor-friendly.
- System Status — live checks of FastAPI, the ML model, RAG, the CrewAI
  crew, and n8n.
- Federation — the multi-hospital registry: run overview, per-condition
  global models, distributed training trigger, and per-run round charts.

Run from the repository root (or from ``frontend/``):

    streamlit run frontend/streamlit_app.py

Configure the backend origin, n8n origin, optional bearer token, and the
analysis route (n8n workflow / direct to FastAPI) in the sidebar.
"""

from __future__ import annotations

import json

from typing import Any

import httpx
import streamlit as st

from dashboard.client import APIConfig, HealthcareAPIClient, HealthcareAPIError
from dashboard.clinical import (
    analysis_stages,
    assessment_summary,
    explanation_sections,
    feature_bounds,
    feature_label,
    feature_unit,
    group_features,
    is_flag_feature,
    is_integer_feature,
    normalize_feature_name,
    parse_blood_pressure,
    validate_feature_values,
)

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_N8N_URL = "http://localhost:5678"
IMAGE_TYPES = ["png", "jpg", "jpeg"]

ROUTE_AUTOMATIC = "Automatic (recommended)"
ROUTE_N8N = "Via n8n workflow"
ROUTE_DIRECT = "Direct to FastAPI"

#: Clinical-safety disclaimer shown near the assessment action and results.
CLINICAL_DISCLAIMER = (
    "AI-assisted clinical decision support. Results should be reviewed "
    "by a qualified clinician."
)

#: Doctor-friendly labels for the assessment types (backed by the presets
#: reported by the backend).
ASSESSMENT_LABELS = {
    "diabetes": "Diabetes Risk",
    "heart": "Heart Disease Risk",
    "kidney": "Chronic Kidney Disease Risk",
    "sepsis": "Sepsis Risk",
}

#: Known dataset presets (fallback when the backend registry is empty).
PRESET_PRESETS: tuple[str, ...] = ("diabetes", "heart", "kidney", "sepsis")


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


def fetch_model_info(client: HealthcareAPIClient) -> dict[str, Any]:
    """
    Fetch model metadata, returning an empty dict when unreachable.

    Parameters
    ----------
    client : HealthcareAPIClient
        Backend client.

    Returns
    -------
    dict[str, Any]
        ``ModelInfo`` payload or ``{}`` on error.
    """
    try:
        return client.model_info()
    except (HealthcareAPIError, httpx.HTTPError):
        return {}


def fetch_presets_info(client: HealthcareAPIClient) -> list[dict[str, Any]]:
    """
    Fetch the named dataset presets, returning an empty list on error.

    Parameters
    ----------
    client : HealthcareAPIClient
        Backend client.

    Returns
    -------
    list[dict[str, Any]]
        ``PresetInfo`` payloads or ``[]`` when the backend does not
        expose them (older backend / unreachable).
    """
    try:
        return client.presets()
    except (HealthcareAPIError, httpx.HTTPError):
        return []


def fetch_federation_status(client: HealthcareAPIClient) -> dict[str, Any]:
    """
    Fetch the federation registry overview, returning an empty dict on error.

    Parameters
    ----------
    client : HealthcareAPIClient
        Backend client.

    Returns
    -------
    dict[str, Any]
        ``FederationStatus`` payload or ``{}`` when the backend does not
        expose the endpoint (older backend / unreachable).
    """
    try:
        return client.federation_status()
    except (HealthcareAPIError, httpx.HTTPError):
        return {}


def assessment_type_label(preset_name: str) -> str:
    """
    Doctor-friendly label for a preset name.

    Parameters
    ----------
    preset_name : str
        Backend preset name (``"diabetes"`` / ...).

    Returns
    -------
    str
        Known label or a Title-cased fallback.
    """
    return ASSESSMENT_LABELS.get(preset_name, preset_name.replace("_", " ").title())


def model_matches_preset(model: dict[str, Any], preset_info: dict[str, Any]) -> bool:
    """
    Whether the currently served model already corresponds to a preset.

    When the served model records its ``preset`` it is authoritative;
    otherwise the feature schemas are compared as a fallback (e.g. for a
    model loaded from ``API_MODEL_PATH`` whose preset is unknown).

    Parameters
    ----------
    model : dict[str, Any]
        ``ModelInfo`` payload.
    preset_info : dict[str, Any]
        ``PresetInfo`` payload for the selected assessment type.

    Returns
    -------
    bool
        True when the served model already matches the preset.
    """
    served_preset = model.get("preset")
    if served_preset:
        return served_preset == preset_info["name"]
    served_features = model.get("feature_names")
    schema = preset_info.get("feature_names")
    if served_features and schema:
        return list(served_features) == list(schema)
    return False


def resolve_route(
    client: HealthcareAPIClient, n8n_base_url: str, requested: str
) -> str:
    """
    Resolve the effective analysis route.

    Parameters
    ----------
    client : HealthcareAPIClient
        Backend client (used for the n8n health probe).
    n8n_base_url : str
        n8n origin.
    requested : str
        Sidebar selection (automatic / via n8n / direct).

    Returns
    -------
    str
        ``"n8n"`` or ``"direct"``.
    """
    if requested == ROUTE_N8N:
        return "n8n"
    if requested == ROUTE_DIRECT:
        return "direct"
    if client.n8n_health(n8n_base_url):
        return "n8n"
    return "direct"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def risk_badge_html(level: str | None) -> str:
    """
    Render a colored risk-level badge (HTML snippet).

    Parameters
    ----------
    level : str | None
        Risk level from the backend (``low`` / ``medium`` / ``high``).

    Returns
    -------
    str
        HTML badge.
    """
    colors = {
        "low": "#1b5e20",
        "medium": "#b26a00",
        "high": "#b71c1c",
        "unknown": "#37474f",
    }
    color = colors.get(str(level), colors["unknown"])
    label = str(level).upper() if level else "UNKNOWN"
    return (
        f'<span style="background:{color};color:white;padding:2px 12px;'
        f'border-radius:12px;font-weight:600;font-size:0.95em;">{label}</span>'
    )


def render_evidence(evidence: list[dict[str, Any]]) -> None:
    """
    Render retrieved clinical evidence readably.

    Raw vector ids, similarity scores, and API internals are deliberately
    hidden; only the knowledge-source label and text are shown.

    Parameters
    ----------
    evidence : list[dict[str, Any]]
        ``EvidenceItem`` payloads.
    """
    for index, item in enumerate(evidence, start=1):
        source = item.get("source") or "Retrieved clinical knowledge"
        st.markdown(f"**{index}. {source}**")
        st.write(item.get("text") or "")
        st.divider()


def render_stages(report: dict[str, Any]) -> None:
    """
    Render the pipeline stages that actually completed for a report.

    Stages are only marked complete when the corresponding output is
    present in the report — nothing is assumed.

    Parameters
    ----------
    report : dict[str, Any]
        ``ClinicalReport`` payload.
    """
    st.markdown("#### Analysis pipeline")
    for stage in analysis_stages(report):
        icon = "✅" if stage["done"] else "⭕"
        detail = stage["detail"]
        suffix = f" — {detail}" if detail else ""
        st.markdown(f"{icon} **{stage['label']}**{suffix}")


def render_explanation(report: dict[str, Any]) -> None:
    """
    Render the explainable decision report from actual model outputs.

    Parameters
    ----------
    report : dict[str, Any]
        ``ClinicalReport`` payload.
    """
    st.markdown("### Explainable Decision Report")
    st.caption(
        "Concise explanation derived from model outputs (prediction, "
        "confidence, risk). It is not a diagnosis and does not expose "
        "internal model reasoning."
    )
    for section in explanation_sections(report):
        st.markdown(f"**{section['title']}**")
        st.write(section["body"])


def render_clinical_results(
    report: dict[str, Any],
    route: str,
    subtitle: str | None = None,
    download_key: str = "download_report",
) -> None:
    """
    Render the six research-defined clinical outputs for a report.

    Sections that the current backend does not support are shown
    explicitly as unavailable rather than fabricated.

    Parameters
    ----------
    report : dict[str, Any]
        ``ClinicalReport`` payload.
    route : str
        ``"n8n"`` or ``"direct"`` (how the analysis was routed).
    subtitle : str | None
        Optional context line above the results.
    download_key : str
        Unique widget key for the report download button (the results may
        be rendered on more than one page in the same run).
    """
    st.divider()
    st.subheader("Clinical Results")
    if subtitle:
        st.caption(subtitle)
    st.caption(CLINICAL_DISCLAIMER)
    route_label = (
        "Orchestrated through the n8n end-to-end workflow"
        if route == "n8n"
        else "Routed directly to the FastAPI backend"
    )
    st.caption(route_label)
    st.warning(report.get("doctor_notice") or "")

    render_stages(report)

    prediction = report.get("prediction")
    risk = report.get("risk")

    st.markdown("### Disease Risk Score")
    if risk:
        score = float(risk.get("risk_score", 0.0))
        level = risk.get("risk_level")
        confidence = (
            float(prediction.get("confidence", 0.0))
            if isinstance(prediction, dict)
            else None
        )
        left, middle, right = st.columns(3)
        left.metric("Model-estimated risk score", f"{score:.2f}")
        middle.markdown("**Risk level**")
        middle.markdown(risk_badge_html(level), unsafe_allow_html=True)
        if confidence is not None:
            right.metric("Model confidence", f"{confidence:.1%}")
        factors = risk.get("risk_factors") or []
        if factors:
            st.markdown("**Contributing factors**")
            for factor in factors:
                st.markdown(f"- {factor}")
        schedule = risk.get("monitoring_schedule") or []
        if schedule:
            st.markdown("**Suggested monitoring**")
            for item in schedule:
                if isinstance(item, dict):
                    st.markdown(
                        f"- {item.get('test', '')} — {item.get('frequency', '')}"
                    )
    else:
        st.info(
            "Risk score not available — no prediction model was configured "
            "for this analysis."
        )
    if isinstance(prediction, dict):
        st.caption(
            "Model-estimated primary condition: "
            f"**{prediction.get('predicted_class')}** "
            f"(confidence {prediction.get('confidence', 0.0):.0%})."
        )

    st.markdown("### Mortality Risk")
    st.info(
        "Not estimated — the current model does not predict mortality risk. "
        "This is documented as future work."
    )

    st.markdown("### Readmission Risk")
    st.info(
        "Not estimated — the current model does not predict readmission risk. "
        "This is documented as future work."
    )

    st.markdown("### Treatment Recommendation")
    recommendations = report.get("recommendations") or []
    if recommendations:
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")
        st.caption(
            "AI-assisted decision support. Recommendations must be reviewed "
            "by a qualified clinician before any action is taken."
        )
    else:
        st.info(
            "No AI treatment recommendation was generated for this analysis. "
            "The Treatment Agent did not produce output in the current "
            "deterministic mode."
        )

    st.markdown("### Clinical Evidence")
    evidence = report.get("evidence") or []
    if evidence:
        st.caption(
            "Retrieved from the clinical knowledge base. Distinguish this "
            "from the AI-generated recommendation above."
        )
        render_evidence(evidence)
    else:
        st.info("No clinical evidence was retrieved from the knowledge base.")

    render_explanation(report)

    st.caption(report.get("limitations") or "")
    patient_id = str(report.get("patient", {}).get("id", "patient"))
    st.download_button(
        "Download report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"clinical_report_{patient_id}.json",
        mime="application/json",
        key=f"{download_key}_{patient_id}",
    )


# ---------------------------------------------------------------------------
# Form widgets
# ---------------------------------------------------------------------------


def feature_widget(name: str) -> float | None:
    """
    Render the appropriate input widget for a model feature.

    Binary flag features become checkboxes, ``sex`` / ``gender`` become a
    two-option selector, bounded features get sensible ranges, integer
    features step by whole numbers, and the rest are plain numeric
    inputs. Verified units are shown in the label.

    Parameters
    ----------
    name : str
        Model feature column name.

    Returns
    -------
    float | None
        The entered value, or None when the entry could not be parsed
        (e.g. an invalid blood-pressure string).
    """
    label = feature_label(name)
    unit = feature_unit(name)
    if unit:
        label = f"{label} ({unit})"
    key = f"feature_{name}"
    if name == "bloodpressure":
        return _blood_pressure_widget(label, key)
    if is_flag_feature(name):
        return float(st.checkbox(label, value=False, key=key))
    if name in {"sex", "gender"}:
        return float(
            st.selectbox(
                label,
                ["0", "1"],
                key=key,
                help="Encoded by the model as 0 / 1.",
            )
        )
    bounds = feature_bounds(name)
    integer = is_integer_feature(name)
    if bounds:
        low, high = bounds
        return st.number_input(
            label,
            min_value=low,
            max_value=high,
            value=low,
            step=1.0,
            format="%d" if integer else "%.1f",
            key=key,
        )
    return st.number_input(
        label,
        min_value=0.0,
        value=0.0,
        step=1.0,
        format="%d" if integer else "%.2f",
        key=key,
    )


def _blood_pressure_widget(label: str, key: str) -> float | None:
    """
    Render a systolic/diastolic blood-pressure entry.

    Accepts ``SYS/DIA`` (e.g. ``120/90``) or a lone value. The model's
    ``bloodpressure`` feature is the diastolic reading (the PIMA diabetes
    "Blood Pressure (mm Hg)" column), so ``120/90`` maps to ``90``.

    Parameters
    ----------
    label : str
        Feature display label.
    key : str
        Session-state key for the text input.

    Returns
    -------
    float | None
        The parsed value (diastolic for ``SYS/DIA`` input), or None when
        the entry is invalid (the caller reports it; nothing is silently
        substituted).
    """
    raw = st.text_input(
        label,
        value="120/80",
        key=key,
        help="Enter as SYS/DIA, e.g. 120/90, or a single value. The "
        "model's Blood Pressure feature uses the diastolic reading.",
    )
    return parse_blood_pressure(raw)


def render_feature_groups(feature_names: list[str]) -> dict[str, float | None]:
    """
    Render the clinical inputs grouped per the research specification.

    The patient-context ``age`` feature is excluded here; it is collected
    once in the Patient Context section and mapped to the model feature
    in a single well-defined place on submission.

    Parameters
    ----------
    feature_names : list[str]
        Model feature columns reported by the backend for the selected
        assessment type.

    Returns
    -------
    dict[str, float | None]
        Feature name to entered value (None marks unparseable entries).
    """
    names = [name for name in feature_names if normalize_feature_name(name) != "age"]
    values: dict[str, float | None] = {}
    for group, grouped_names in group_features(names):
        if group == "Additional Model Features":
            with st.expander("Additional model features"):
                _render_group_inputs(group, grouped_names, values)
        else:
            st.markdown(f"**{group}**")
            _render_group_inputs(group, grouped_names, values)
    return values


def _render_group_inputs(
    group: str, names: list[str], values: dict[str, float | None]
) -> None:
    """
    Render the inputs of one feature group in a multi-column grid.

    Parameters
    ----------
    group : str
        Group label (unused, kept for clarity).
    names : list[str]
        Feature names in the group.
    values : dict[str, float | None]
        Output mapping to fill.
    """
    columns = st.columns(min(len(names), 3))
    for index, name in enumerate(names):
        with columns[index % len(columns)]:
            values[name] = feature_widget(name)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def run_overview_tab(client: HealthcareAPIClient, n8n_base_url: str) -> None:
    """Render the Overview page."""
    st.header("Clinical Decision Support — Overview")
    st.caption(
        "Research prototype. Model outputs are estimates and must be "
        "reviewed by a licensed clinician."
    )

    st.markdown(
        "This dashboard is the **Clinical Decision Support interface for "
        "doctors** in the federated multi-agent healthcare framework."
    )
    st.markdown(
        "### Research workflow\n\n"
        "Patient Data → Federated Prediction → Disease Prediction Agent → "
        "RAG Retrieval → Treatment Agent → Explainability → "
        "**n8n Workflow** → Doctor Dashboard"
    )

    st.markdown("### What you can do here")
    st.markdown(
        "- **Clinical Assessment** — enter the supported clinical data and "
        "press **Analyze Patient** to run the workflow.\n"
        "- **Imaging** — analyze a medical image (when an image model is "
        "configured).\n"
        "- **Results** — review disease risk, mortality / readmission risk "
        "(where supported), treatment recommendations, clinical evidence, "
        "and the explainable decision report.\n"
        "- **System Status** — check that FastAPI, the ML model, RAG, the "
        "CrewAI crew, and n8n are operational."
    )

    st.markdown("### Current configuration")
    try:
        health = client.health()
        st.markdown(
            f"- FastAPI backend: **reachable** "
            f"({health.get('name')} v{health.get('version')})"
        )
    except (HealthcareAPIError, httpx.HTTPError):
        st.markdown("- FastAPI backend: **not reachable**")
    model = fetch_model_info(client)
    if model.get("available"):
        st.markdown(
            f"- ML model: **configured** "
            f"({model.get('model_type')} / {model.get('model_name')})"
        )
    else:
        st.markdown("- ML model: **not configured**")
    if client.n8n_health(n8n_base_url):
        st.markdown(f"- n8n: **reachable** ({n8n_base_url})")
    else:
        st.markdown("- n8n: **not reachable** (direct FastAPI routing will be used)")

    st.caption(
        "Persistent patient records are not implemented — each assessment "
        "is entered fresh. Storing patients and previous assessments is "
        "recorded as future work."
    )


def run_assessment_tab(client: HealthcareAPIClient) -> None:
    """Render the Clinical Assessment page and run the analysis."""
    st.header("Clinical Assessment")
    st.caption(
        "Enter the clinical information for a model-specific risk "
        "assessment. Only the measurements the selected model actually "
        "uses are shown; the analysis runs through the FastAPI / n8n "
        "workflow and returns a structured clinical decision-support "
        "report."
    )

    model = fetch_model_info(client)
    presets = fetch_presets_info(client)

    selected: str | None = None
    schema: list[str] = []
    need_train = False
    if presets:
        names = [item["name"] for item in presets]
        served_preset = model.get("preset")
        default_index = names.index(served_preset) if served_preset in names else 0
        st.markdown("#### Assessment Type")
        selected = st.selectbox(
            "Select the clinical condition to assess",
            options=names,
            index=default_index,
            format_func=assessment_type_label,
            help="The form shows exactly the measurements the selected model requires.",
        )
        preset_info = next(item for item in presets if item["name"] == selected)
        schema = list(preset_info.get("feature_names") or [])
        if not schema:
            st.info(
                f"No trained model is available for "
                f"**{assessment_type_label(selected)}** yet. Train it first "
                f"(POST /api/v1/train with preset='{selected}', or via the "
                "runner) so the dashboard can show its required measurements."
            )
            return
        need_train = not model_matches_preset(model, preset_info)
        if need_train:
            st.caption(
                f"This assessment uses the **{assessment_type_label(selected)}** "
                "model. Running it will train/serve that model first; the "
                "currently served model is replaced for subsequent analyses."
            )
    else:
        schema = list(model.get("feature_names") or [])
        if not schema:
            st.info(
                "No tabular model with known features is configured. You can "
                "still run an analysis — the report will contain clinical "
                "evidence without a prediction. (Train a model via the API or "
                "set API_MODEL_PATH.)"
            )
            return

    input_mode = st.radio(
        "Input method",
        options=["Manual Entry", "CSV Upload"],
        index=0,
        horizontal=True,
        key="assessment_input_mode",
    )
    if input_mode == "CSV Upload":
        st.caption(
            "Upload a CSV file whose columns match the model feature names "
            "(the first row is analyzed). Parsing and preprocessing happen "
            "on the backend; the n8n workflow forwards the file when the "
            "analysis route uses n8n."
        )

    with st.form("assessment_form"):
        st.markdown("**Patient context**")
        st.caption(
            "Patient identification and audit metadata. It is never sent to "
            "the ML model as a feature."
        )
        context_left, context_right = st.columns(2)
        patient_name = context_left.text_input(
            "Patient name", value="Patient", key="patient_name"
        )
        patient_id = context_right.text_input(
            "Patient ID", value="patient-1", key="patient_id"
        )
        patient_age = st.number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            value=45,
            step=1,
            key="patient_age",
        )
        st.divider()

        csv_file = None
        feature_values: dict[str, float | None] = {}
        if input_mode == "CSV Upload":
            st.markdown("**Clinical data (CSV)**")
            csv_file = st.file_uploader(
                "Clinical data CSV",
                type=["csv"],
                key="assessment_csv",
            )
        else:
            st.markdown("**Clinical measurements**")
            st.caption(
                "All fields are required by the selected model. Units are "
                "shown where verified by the dataset; unverified units are "
                "left unspecified."
            )
            feature_values = render_feature_groups(schema)
        st.divider()

        st.markdown("**Optional clinical notes**")
        notes = st.text_area(
            "Clinical notes (optional)", value="", height=70, key="patient_notes"
        )
        st.divider()

        st.markdown("**Assessment summary**")
        summary_rows = assessment_summary(
            patient={"name": patient_name, "id": patient_id},
            preset_label=assessment_type_label(selected or "current model"),
            schema=schema,
            values=feature_values,
            notes_provided=bool(notes.strip()),
            patient_age=int(patient_age),
        )
        if input_mode == "CSV Upload":
            summary_rows = [
                (
                    (label, "CSV upload" if csv_file is not None else "Not uploaded")
                    if label == "Clinical data"
                    else (label, value)
                )
                for label, value in summary_rows
            ]
        for label, value in summary_rows:
            st.markdown(f"- **{label}**: {value}")
        st.caption(
            "Medical image analysis is a separate workflow (Imaging tab); "
            "multimodal fusion is not implemented, so no image is combined "
            "with this assessment."
        )
        st.caption(CLINICAL_DISCLAIMER)
        submitted = st.form_submit_button("Run Clinical Assessment", type="primary")

    if not submitted:
        return

    patient = {
        "name": patient_name,
        "id": patient_id,
        "age": int(patient_age),
        "notes": notes,
    }

    if input_mode == "CSV Upload":
        if csv_file is None:
            st.error("Please upload a CSV file before running the assessment.")
            return
        route = resolve_route(
            client,
            st.session_state.get("n8n_base_url", DEFAULT_N8N_URL),
            st.session_state.get("analysis_route", ROUTE_AUTOMATIC),
        )
        try:
            with st.spinner("Running the clinical analysis pipeline…"):
                if route == "n8n":
                    n8n_kwargs: dict[str, Any] = {}
                    if need_train and selected is not None:
                        n8n_kwargs = {"preset": selected, "train": True}
                    report = client.analyze_csv_via_n8n(
                        st.session_state.get("n8n_base_url", DEFAULT_N8N_URL),
                        patient=patient,
                        csv=csv_file.getvalue(),
                        **n8n_kwargs,
                    )
                else:
                    report = client.analyze_csv(
                        patient=patient, csv=csv_file.getvalue()
                    )
        except HealthcareAPIError as error:
            st.error(str(error))
            if route == "n8n":
                st.caption(
                    "The n8n workflow did not complete. Check that n8n is "
                    "running and the workflow is active, or switch the "
                    "analysis route to 'Direct to FastAPI' in the sidebar."
                )
            return
        except httpx.HTTPError as error:
            st.error(f"Could not reach the analysis workflow: {error}")
            return
        st.session_state["clinical_report"] = report
        st.session_state["report_route"] = route
        render_clinical_results(
            report,
            route,
            subtitle=f"Patient {patient_id}",
            download_key="download_assessment_csv",
        )
        st.caption(CLINICAL_DISCLAIMER)
        return

    errors = validate_feature_values(
        feature_values, schema, patient_age=int(patient_age)
    )
    if errors:
        st.error("Please correct the following before running the assessment:")
        for message in errors:
            st.error(f"- {message}")
        return

    features = {
        name: float(value)
        for name, value in feature_values.items()
        if value is not None
    }
    if "age" in schema:
        features["age"] = float(patient_age)
    markers = dict(features) if features else None
    route = resolve_route(
        client,
        st.session_state.get("n8n_base_url", DEFAULT_N8N_URL),
        st.session_state.get("analysis_route", ROUTE_AUTOMATIC),
    )

    try:
        with st.spinner("Running the clinical analysis pipeline…"):
            if route == "n8n":
                n8n_kwargs: dict[str, Any] = {}
                if need_train and selected is not None:
                    n8n_kwargs = {"preset": selected, "train": True}
                report = client.analyze_via_n8n(
                    st.session_state.get("n8n_base_url", DEFAULT_N8N_URL),
                    patient=patient,
                    features=features,
                    markers=markers,
                    input_type="csv",
                    **n8n_kwargs,
                )
            else:
                if need_train and selected is not None:
                    client.train(selected, model="mlp")
                report = client.analyze(
                    patient=patient,
                    features=features,
                    markers=markers,
                    input_type="csv",
                )
    except HealthcareAPIError as error:
        st.error(str(error))
        if route == "n8n":
            st.caption(
                "The n8n workflow did not complete. Check that n8n is "
                "running and the workflow is active, or switch the analysis "
                "route to 'Direct to FastAPI' in the sidebar."
            )
        return
    except httpx.HTTPError as error:
        st.error(f"Could not reach the analysis workflow: {error}")
        return

    st.session_state["clinical_report"] = report
    st.session_state["report_route"] = route
    render_clinical_results(
        report,
        route,
        subtitle=f"Patient {patient_id}",
        download_key="download_assessment",
    )
    st.caption(CLINICAL_DISCLAIMER)


def run_imaging_tab(client: HealthcareAPIClient) -> None:
    """Render the Imaging page (upload -> preview -> analyze)."""
    st.header("Imaging")
    model = fetch_model_info(client)
    image_available = bool(
        model.get("available")
        and model.get("model_type") in {"image", "tabular_and_image"}
    )

    if not image_available:
        st.info(
            "Medical image analysis is **not currently available**: the "
            "backend has no image model configured. To enable it, train the "
            "image model with `scripts/train_image_model.py` and set "
            "`API_IMAGE_MODEL_PATH` (the runner wires this automatically)."
        )
        return

    classes = model.get("classes") or []
    st.caption(
        "Upload a medical image to run the image-based clinical analysis: "
        "upload → preview → analyze → prediction → confidence → explanation."
    )
    if classes:
        st.caption(f"Supported classes: {', '.join(classes)}")

    context_left, context_right = st.columns(2)
    patient_name = context_left.text_input(
        "Patient name", value="Patient", key="image_patient_name"
    )
    patient_id = context_right.text_input(
        "Patient ID", value="patient-image", key="image_patient_id"
    )
    patient_age = st.number_input(
        "Age", min_value=0, max_value=120, value=45, step=1, key="image_patient_age"
    )

    uploaded = st.file_uploader(
        "Medical image (PNG / JPG / JPEG)",
        type=IMAGE_TYPES,
        help="Analyzed by the backend image model.",
    )
    if uploaded is not None:
        st.image(uploaded, caption=uploaded.name, width=280)

    analyze = st.button("Analyze Image", type="primary", disabled=uploaded is None)
    if not (analyze and uploaded is not None):
        return

    patient = {
        "name": patient_name,
        "id": patient_id,
        "age": int(patient_age),
        "notes": "",
    }
    try:
        with st.spinner("Running the image analysis pipeline…"):
            report = client.analyze_image(patient=patient, image=uploaded.getvalue())
    except (HealthcareAPIError, httpx.HTTPError) as error:
        st.error(str(error))
        return

    st.session_state["clinical_report"] = report
    st.session_state["report_route"] = "direct"
    render_clinical_results(
        report,
        "direct",
        subtitle=f"Image analysis — {uploaded.name} (patient {patient_id})",
        download_key="download_imaging",
    )


def run_results_tab() -> None:
    """Render the most recent analysis results."""
    st.header("Results")
    report = st.session_state.get("clinical_report")
    if report is None:
        st.info(
            "No analysis has been run yet. Enter patient data on the "
            "**Clinical Assessment** page and press **Analyze Patient**."
        )
        return
    route = st.session_state.get("report_route", "direct")
    patient_id = str(report.get("patient", {}).get("id", "patient"))
    render_clinical_results(
        report,
        route,
        subtitle=f"Patient {patient_id} — most recent analysis",
        download_key="download_results",
    )


def run_system_status_tab(client: HealthcareAPIClient, n8n_base_url: str) -> None:
    """Render live system-health checks."""
    st.header("System Status")
    st.caption(
        "Live checks against the running services. Only statuses that can "
        "actually be probed are reported."
    )

    rows: list[tuple[str, str, str, str]] = []

    try:
        health = client.health()
        rows.append(
            (
                "FastAPI",
                "ok",
                "Operational",
                (
                    f"{health.get('name')} v{health.get('version')} — "
                    f"{health.get('status')}"
                ),
            )
        )
    except (HealthcareAPIError, httpx.HTTPError) as error:
        rows.append(("FastAPI", "bad", "Unreachable", str(error)))

    model = fetch_model_info(client)
    if model.get("available"):
        classes = ", ".join(model.get("classes") or []) or "—"
        rows.append(
            (
                "ML model",
                "ok",
                "Configured",
                (
                    f"{model.get('model_type')} · {model.get('model_name')} · "
                    f"classes: {classes}"
                ),
            )
        )
    else:
        rows.append(
            (
                "ML model",
                "warn",
                "Not configured",
                "Train a model via /api/v1/train or set API_MODEL_PATH.",
            )
        )

    try:
        evidence = client.retrieve("clinical evidence and management", top_k=3)
        rows.append(
            (
                "RAG",
                "ok" if evidence else "warn",
                "Operational" if evidence else "No evidence",
                (
                    f"{len(evidence)} evidence item(s) returned for a probe query"
                    if evidence
                    else "Knowledge base returned no evidence"
                ),
            )
        )
    except (HealthcareAPIError, httpx.HTTPError) as error:
        rows.append(("RAG", "bad", "Unavailable", str(error)))

    fastapi_ok = any(row[0] == "FastAPI" and row[1] == "ok" for row in rows)
    rows.append(
        (
            "CrewAI",
            "ok" if fastapi_ok else "unknown",
            "Operational" if fastapi_ok else "Unknown",
            (
                "Deterministic clinical crew embedded in the backend; exercised "
                "on every analysis."
                if fastapi_ok
                else "Cannot be probed while the backend is down."
            ),
        )
    )

    if client.n8n_health(n8n_base_url):
        rows.append(
            (
                "n8n",
                "ok",
                "Operational",
                f"{n8n_base_url} — webhook /webhook/healthcare-endtoend",
            )
        )
    else:
        rows.append(
            (
                "n8n",
                "warn",
                "Unavailable",
                (
                    f"{n8n_base_url} not reachable. Analyses fall back to the "
                    "direct FastAPI route (N8N_ENABLED=0 dev mode)."
                ),
            )
        )

    tone = {"ok": "🟢", "warn": "🟡", "bad": "🔴", "unknown": "⚪"}
    for component, verdict, status, detail in rows:
        st.markdown(
            f"{tone.get(verdict, '⚪')} **{component}** — {status} "
            f"<span style='color:#6b7280'>{detail}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Current analysis route**")
    requested = st.session_state.get("analysis_route", ROUTE_AUTOMATIC)
    effective = resolve_route(client, n8n_base_url, requested)
    if effective == "n8n":
        st.success(
            f"Analyses are routed through the n8n workflow (selection: {requested})."
        )
    else:
        st.warning(
            f"Analyses are routed directly to FastAPI (selection: {requested}). "
            "n8n is part of the architecture; start it to route through the workflow."
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def run_federation_tab(client: HealthcareAPIClient) -> None:
    """Render the federation registry overview and distributed training."""
    st.header("Federated Learning")
    st.caption(
        "Multi-hospital federation overview: distributed Flower gRPC runs, "
        "versioned global models, and per-round metrics from the SQLite "
        "model registry."
    )

    status = fetch_federation_status(client)
    if not status:
        st.info(
            "The federation registry is not reachable. Start the backend "
            "and run a distributed training (POST /api/v1/train with "
            "`distributed: true`) to populate it."
        )
        return

    st.markdown("### Registry overview")
    overview_left, overview_middle, overview_right = st.columns(3)
    overview_left.metric("Federation runs", status.get("n_runs", 0))
    overview_middle.metric("Registered models", status.get("n_models", 0))
    overview_right.caption("Registry")
    overview_right.caption(str(status.get("registry_path") or "not configured"))

    st.markdown("### Models by condition")
    presets = status.get("presets") or []
    available_presets = [preset for preset in presets if preset.get("available")]
    if not available_presets:
        st.info("No federated global model has been registered yet.")
    else:
        rows = []
        for preset in available_presets:
            model = preset.get("latest_model") or {}
            rows.append(
                {
                    "Condition": ASSESSMENT_LABELS.get(
                        preset.get("name"), preset.get("name", "")
                    ),
                    "Version": int(model.get("version", 0)),
                    "Accuracy": model.get("accuracy"),
                    "ROC-AUC": model.get("roc_auc"),
                    "DP epsilon": model.get("epsilon"),
                    "Secure aggregation": (
                        "on" if model.get("secure_aggregation") else "off"
                    ),
                    "Differential privacy": (
                        "on" if model.get("differential_privacy") else "off"
                    ),
                }
            )
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("### Distributed training")
    st.caption(
        "Run a new federated round across hospital client processes "
        "(server + one process per hospital on this machine, over gRPC)."
    )
    train_left, train_middle, train_right = st.columns(3)
    fed_preset = train_left.selectbox(
        "Condition",
        options=[item["name"] for item in presets] if presets else list(PRESET_PRESETS),
        format_func=assessment_type_label,
        key="fed_preset",
    )
    fed_clients = train_middle.number_input(
        "Hospital clients",
        min_value=2,
        max_value=8,
        value=3,
        step=1,
        key="fed_clients",
    )
    fed_rounds = train_right.number_input(
        "Rounds",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="fed_rounds",
    )
    privacy_left, privacy_right = st.columns(2)
    fed_secure = privacy_left.checkbox(
        "Secure aggregation",
        value=False,
        help="Mask client updates with the pairwise one-time-pad aggregator.",
        key="fed_secure",
    )
    fed_dp = privacy_right.checkbox(
        "Differential privacy (DP-SGD)",
        value=False,
        help="Clients train with Opacus DP-SGD; the run reports epsilon.",
        key="fed_dp",
    )

    if st.button(
        "Run distributed training",
        type="primary",
        key="run_federation",
        help="Launches the distributed Flower server + hospital clients.",
    ):
        try:
            with st.spinner(
                f"Running {fed_rounds} federated rounds across "
                f"{fed_clients} hospital processes…"
            ):
                response = client.train_distributed(
                    preset=fed_preset,
                    clients=int(fed_clients),
                    rounds=int(fed_rounds),
                    secure_aggregation=fed_secure,
                    differential_privacy=fed_dp,
                )
        except (HealthcareAPIError, httpx.HTTPError) as error:
            st.error(str(error))
            return
        st.success("Distributed training completed.")
        metrics = response.get("federated_metrics") or {}
        st.markdown("**Run result**")
        st.markdown(
            f"- Run id: `{metrics.get('run_id', '—')}` · version "
            f"{metrics.get('version', '—')}"
        )
        st.markdown(f"- Hold-out accuracy: **{metrics.get('accuracy', '—')}**")
        if metrics.get("roc_auc") is not None:
            st.markdown(f"- Hold-out ROC-AUC: **{metrics.get('roc_auc'):.3f}**")
        if metrics.get("epsilon") is not None:
            st.markdown(f"- Worst-case DP epsilon: **{metrics.get('epsilon'):.3f}**")

    st.divider()
    st.markdown("### Recent runs")
    try:
        runs = client.federation_runs()
    except (HealthcareAPIError, httpx.HTTPError):
        runs = []
    if not runs:
        st.caption("No distributed runs recorded yet.")
    else:
        run_labels = {
            run["run_id"]: (
                f"{run['run_id']} — "
                f"{ASSESSMENT_LABELS.get(run.get('preset'), run.get('preset', ''))}"
            )
            for run in runs
        }
        selected_run = st.selectbox(
            "Run",
            options=[run["run_id"] for run in runs],
            format_func=lambda run_id: run_labels.get(run_id, run_id),
            key="fed_run_select",
        )
        run = next((item for item in runs if item["run_id"] == selected_run), None)
        if run is not None:
            st.markdown(
                f"- **Status**: {run.get('status')} · "
                f"**Hospitals**: {run.get('n_hospitals')} · "
                f"**Rounds**: {run.get('n_rounds')}"
            )
            st.markdown(
                f"- Secure aggregation: "
                f"**{'on' if run.get('secure_aggregation') else 'off'}**"
                f" · Differential privacy: "
                f"**{'on' if run.get('differential_privacy') else 'off'}**"
            )
            try:
                rounds = client.federation_rounds(run["run_id"])
            except (HealthcareAPIError, httpx.HTTPError):
                rounds = []
            if rounds:
                chart = [
                    {
                        "Round": round_info.get("round_index", 0),
                        "Accuracy": round_info.get("accuracy"),
                    }
                    for round_info in rounds
                ]
                st.line_chart(
                    chart,
                    x="Round",
                    y="Accuracy",
                    color="#1b5e20",
                )


def render_sidebar() -> tuple[HealthcareAPIClient, str, str]:
    """
    Render the sidebar configuration and return (client, n8n URL, route).

    Returns
    -------
    tuple[HealthcareAPIClient, str, str]
        The configured client, the n8n base URL, and the selected route.
    """
    with st.sidebar:
        st.title("Healthcare AI")
        st.caption("Clinical Decision Support")

        base_url = st.text_input(
            "Backend URL", value=DEFAULT_BACKEND_URL, key="backend_url"
        )
        api_token = st.text_input(
            "API token (optional)", value="", type="password", key="api_token"
        )
        n8n_base_url = st.text_input(
            "n8n URL",
            value=DEFAULT_N8N_URL,
            key="n8n_base_url",
            help="n8n orchestrates the end-to-end workflow.",
        )
        with st.expander("Advanced"):
            route = st.radio(
                "Analysis route",
                options=[ROUTE_AUTOMATIC, ROUTE_N8N, ROUTE_DIRECT],
                key="analysis_route",
                help=(
                    "Automatic uses the n8n workflow when it is reachable "
                    "and falls back to the FastAPI backend. N8N_ENABLED=0 "
                    "may be used for development/testing."
                ),
            )

        st.divider()
        client = build_client(base_url, api_token)
        try:
            health = client.health()
            st.success(
                f"Backend connected — {health.get('name')} v{health.get('version')}"
            )
        except (HealthcareAPIError, httpx.HTTPError):
            st.warning("Backend not reachable")

    return client, n8n_base_url, route


def main() -> None:
    """Render the dashboard."""
    st.set_page_config(
        page_title="Healthcare AI — Clinical Decision Support",
        page_icon="🏥",
        layout="wide",
    )

    client, n8n_base_url, _route = render_sidebar()

    tab_overview, tab_assessment, tab_imaging, tab_results, tab_status, tab_fed = (
        st.tabs(
            [
                "Overview",
                "Clinical Assessment",
                "Imaging",
                "Results",
                "System Status",
                "Federation",
            ]
        )
    )
    with tab_overview:
        run_overview_tab(client, n8n_base_url)
    with tab_assessment:
        run_assessment_tab(client)
    with tab_imaging:
        run_imaging_tab(client)
    with tab_results:
        run_results_tab()
    with tab_status:
        run_system_status_tab(client, n8n_base_url)
    with tab_fed:
        run_federation_tab(client)


if __name__ == "__main__":
    main()
