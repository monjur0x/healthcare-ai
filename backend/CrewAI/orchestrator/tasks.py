"""
Task definitions for the healthcare crew.

Tasks follow the architecture's data flow: patient analysis → disease
prediction → evidence retrieval → treatment → explanation → risk
monitoring → report generation. Each task passes its output to the next
via the CrewAI ``context`` mechanism.
"""

from __future__ import annotations

from collections.abc import Mapping

from .prompts import REPORT_SCHEMA_INSTRUCTIONS, TASK_DESCRIPTIONS
from .schemas import PatientInfo


def _clinical_context_block(
    features: Mapping[str, float] | None,
    markers: Mapping[str, float] | None,
    disease_context: Mapping[str, object] | None,
) -> str:
    """
    Render the patient's clinical values as a prompt context block.

    Task descriptions previously embedded only the patient demographics
    (``PatientInfo``), so LLM agents never saw the actual feature or
    marker values and narratives could contradict the inputs. This
    block injects them verbatim; values are bounded to keep prompts
    compact.

    Parameters
    ----------
    features : Mapping[str, float] | None
        Feature row fed to the prediction model.
    markers : Mapping[str, float] | None
        Raw clinical markers used by the risk assessment.
    disease_context : Mapping[str, object] | None
        Disease registry entry for the assessed preset.

    Returns
    -------
    str
        Multi-line context block (possibly empty).
    """

    lines: list[str] = []
    if disease_context:
        lines.append(f"Assessed condition: {disease_context.get('disease', 'unknown')}")
    if features:
        rendered = ", ".join(f"{k}={v}" for k, v in list(features.items())[:12])
        lines.append(f"Feature values: {rendered}")
    if markers:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(markers.items())[:12])
        lines.append(f"Clinical markers: {rendered}")
    return "\n".join(lines)


def create_tasks(
    agents: Mapping[str, object],
    patient: PatientInfo,
    features: Mapping[str, float] | None = None,
    markers: Mapping[str, float] | None = None,
    disease_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    Build the healthcare analysis tasks.

    Parameters
    ----------
    agents : Mapping[str, object]
        Agents keyed by role name (from ``create_agents``).
    patient : PatientInfo
        Patient context injected into the task descriptions.
    features : Mapping[str, float] | None
        Feature row fed to the prediction model; injected into the
        task descriptions so agents reason over the real values.
    markers : Mapping[str, float] | None
        Raw clinical markers; injected alongside the features.
    disease_context : Mapping[str, object] | None
        Disease registry entry for the assessed preset.

    Returns
    -------
    dict[str, object]
        Tasks keyed by task name, in execution order.
    """

    from crewai import Task

    context = {
        "patient": patient.model_dump(),
        "report_schema": REPORT_SCHEMA_INSTRUCTIONS,
    }
    clinical_block = _clinical_context_block(features, markers, disease_context)

    # Concurrency: ``evidence_retrieval`` and ``explanation`` run
    # asynchronously (concurrently). Both depend only on the two sync
    # tasks before them (patient analysis, disease prediction) and not
    # on each other, so parallel execution is safe. Any later task that
    # lists them in its ``context`` (treatment, report generation)
    # automatically waits for their results — CrewAI awaits all pending
    # async tasks before executing the next sync task.

    task_patient_analysis = Task(
        description=(
            TASK_DESCRIPTIONS["patient_analysis"]
            + f"\nPatient: {context['patient']}"
            + (f"\n{clinical_block}" if clinical_block else "")
        ),
        expected_output=(
            "A structured patient summary with data quality notes and key "
            "health indicators."
        ),
        agent=agents["patient_analyst"],
    )

    task_disease_prediction = Task(
        description=(
            TASK_DESCRIPTIONS["disease_prediction"]
            + f"\nPatient: {context['patient']}"
            + (f"\n{clinical_block}" if clinical_block else "")
        ),
        expected_output=(
            "A diagnostic assessment with primary condition, confidence, "
            "severity, and risk factors."
        ),
        agent=agents["disease_predictor"],
        context=[task_patient_analysis],
    )

    task_evidence_retrieval = Task(
        description=TASK_DESCRIPTIONS["evidence_retrieval"],
        expected_output=(
            "Verifiable clinical evidence with source labels for each finding."
        ),
        agent=agents["medical_researcher"],
        context=[task_patient_analysis, task_disease_prediction],
        async_execution=True,
    )

    task_treatment = Task(
        description=TASK_DESCRIPTIONS["treatment_recommendation"],
        expected_output=(
            "Evidence-based treatment recommendations with the physician-"
            "review disclaimer."
        ),
        agent=agents["treatment_planner"],
        context=[task_disease_prediction, task_evidence_retrieval],
    )

    task_explanation = Task(
        description=(
            TASK_DESCRIPTIONS["explanation"]
            + (f"\n{clinical_block}" if clinical_block else "")
        ),
        expected_output=(
            "A plain-language explanation of the prediction and the "
            "clinical meaning of the contributing features."
        ),
        agent=agents["explainability_expert"],
        context=[task_patient_analysis, task_disease_prediction],
        async_execution=True,
    )

    task_risk_monitoring = Task(
        description=(
            TASK_DESCRIPTIONS["risk_monitoring"]
            + (f"\n{clinical_block}" if clinical_block else "")
        ),
        expected_output=(
            "A monitoring plan with alert thresholds and screening recommendations."
        ),
        agent=agents["risk_monitor"],
        context=[task_disease_prediction, task_treatment],
    )

    task_report_generation = Task(
        description=(
            TASK_DESCRIPTIONS["report_generation"]
            + f"\nReport schema: {context['report_schema']}"
        ),
        expected_output=(
            "The complete structured clinical report as JSON matching the "
            "report schema, including all disclaimers."
        ),
        agent=agents["report_writer"],
        context=[
            task_patient_analysis,
            task_disease_prediction,
            task_evidence_retrieval,
            task_treatment,
            task_explanation,
            task_risk_monitoring,
        ],
    )

    return {
        "patient_analysis": task_patient_analysis,
        "disease_prediction": task_disease_prediction,
        "evidence_retrieval": task_evidence_retrieval,
        "treatment_recommendation": task_treatment,
        "explanation": task_explanation,
        "risk_monitoring": task_risk_monitoring,
        "report_generation": task_report_generation,
    }


__all__ = ["create_tasks"]
