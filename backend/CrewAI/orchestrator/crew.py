"""
The healthcare clinical crew.

``ClinicalCrew`` offers two execution paths over the same deterministic
services:

- ``run_analysis`` — a fully offline, deterministic pipeline
  (prediction → risk → evidence → report) that needs no LLM and is
  always reproducible (ADR-008).
- ``run_llm`` — the same pipeline plus CrewAI agent orchestration for
  narrative enrichment. Only enabled when ``CREW_LLM_API_KEY`` is set.

``run`` picks the LLM path when configured, otherwise falls back to the
deterministic pipeline.
"""

from __future__ import annotations

import json
import os

from collections.abc import Mapping

import numpy as np

from preprocessing.logger import get_logger

from .config import settings
from .exceptions import LLMNotConfiguredError, OrchestrationError
from .schemas import (
    ClinicalReport,
    EvidenceItem,
    PatientInfo,
    PredictionResult,
)
from .services import (
    assemble_clinical_report,
    assess_risk,
    retrieve_evidence,
    run_image_prediction,
    run_prediction,
)

logger = get_logger(__name__)


class ClinicalCrew:
    """
    Orchestrates a patient analysis into a structured clinical report.

    Parameters
    ----------
    patient : PatientInfo
        Patient context.
    input_type : str
        Data modality analyzed (``"csv"`` / ``"image"`` / ...).
    model : object | None
        A fitted ``TabularClassifier`` (or any model with
        ``predict_proba`` / ``classes_``) for the prediction step.
    features : Mapping[str, float] | None
        Full feature row (preprocessed) for the prediction step. Required
        when ``model`` is provided.
    image_model : object | None
        A fitted ``ImageClassifier`` for image-based prediction.
    image : np.ndarray | None
        Preprocessed image array ``(H, W, C)`` for the prediction step.
        Required when ``image_model`` is provided.
    rag_pipeline : object | None
        An ingested ``RAGPipeline`` for the evidence step.
    markers : Mapping[str, float] | None
        Optional raw clinical markers for risk assessment.
    recommendations : list[str] | None
        Optional recommendation strings for the report.
    """

    def __init__(
        self,
        patient: PatientInfo,
        input_type: str = "csv",
        model: object | None = None,
        features: Mapping[str, float] | None = None,
        image_model: object | None = None,
        image: np.ndarray | None = None,
        rag_pipeline: object | None = None,
        markers: Mapping[str, float] | None = None,
        recommendations: list[str] | None = None,
    ) -> None:
        self.patient = patient
        self.input_type = input_type
        self._model = model
        self._features = dict(features or {})
        self._image_model = image_model
        self._image = image
        self._rag_pipeline = rag_pipeline
        self._markers = dict(markers or {})
        self._recommendations = list(recommendations or [])

    def run_analysis(self) -> ClinicalReport:
        """
        Run the deterministic, LLM-free analysis pipeline.

        Returns
        -------
        ClinicalReport
            The assembled structured report.
        """

        prediction = risk = None
        if self._image_model is not None:
            if self._image is None:
                raise OrchestrationError(
                    "The image prediction step requires a preprocessed "
                    "image array; pass image to ClinicalCrew."
                )
            prediction = run_image_prediction(self._image_model, self._image)
            risk = assess_risk(prediction, self._markers)
        elif self._model is not None:
            if not self._features:
                raise OrchestrationError(
                    "The prediction step requires a full feature row; "
                    "pass features to ClinicalCrew."
                )
            prediction = run_prediction(self._model, self._features)
            risk = assess_risk(prediction, self._markers)

        evidence: list[EvidenceItem] = []
        if self._rag_pipeline is not None:
            query = self._build_query(prediction)
            evidence = retrieve_evidence(self._rag_pipeline, query)

        report = assemble_clinical_report(
            patient=self.patient,
            input_type=self.input_type,
            prediction=prediction,
            risk=risk,
            evidence=evidence,
            recommendations=self._recommendations,
        )
        logger.info("Deterministic analysis complete for patient %s", self.patient.id)
        return report

    def run_llm(self) -> ClinicalReport:
        """
        Run the CrewAI-orchestrated pipeline (requires an LLM key).

        Enriches the deterministic analysis with agent reasoning. If the
        crew's JSON output cannot be parsed, the deterministic report is
        returned instead.

        Returns
        -------
        ClinicalReport
            The assembled structured report.

        Raises
        ------
        LLMNotConfiguredError
            If ``CREW_LLM_API_KEY`` is not configured.
        OrchestrationError
            If the crew cannot be built.
        """

        if not settings.LLM_API_KEY:
            raise LLMNotConfiguredError(
                "LLM orchestration requires CREW_LLM_API_KEY; use "
                "run_analysis() for the offline deterministic path."
            )

        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get(
            "GOOGLE_API_KEY"
        ):
            os.environ["GEMINI_API_KEY"] = settings.LLM_API_KEY

        base = self.run_analysis()
        try:
            from crewai import Crew, Process

            from .agents import _agent_llm, create_agents
            from .tasks import create_tasks
        except Exception as error:
            raise OrchestrationError(f"CrewAI is not available: {error}") from error

        tool_instances = {}
        if self._model is not None:
            from .tools import PredictionTool

            tool_instances["disease_prediction"] = PredictionTool(model=self._model)
        if self._rag_pipeline is not None:
            from .tools import RAGRetrievalTool

            tool_instances["evidence_retrieval"] = RAGRetrievalTool(
                pipeline=self._rag_pipeline
            )
        agents = create_agents(tool_instances, llm=_agent_llm())
        tasks = create_tasks(agents, self.patient)
        crew = Crew(
            agents=list(agents.values()),
            tasks=list(tasks.values()),
            process=Process.sequential,
            verbose=settings.CREW_VERBOSE,
            memory=settings.CREW_MEMORY,
            planning=False,
        )
        try:
            result = crew.kickoff(inputs={"base_report": base.to_dict()})
        except Exception as error:  # noqa: BLE001 - LLM failures fall back
            logger.error("Crew kickoff failed: %s", error)
            return base

        parsed = self._parse_report(result)
        if parsed is None:
            logger.warning(
                "Could not parse crew result as a report; returning base report"
            )
            return base
        logger.info("LLM analysis complete for patient %s", self.patient.id)
        return parsed

    def run(self) -> ClinicalReport:
        """
        Run the analysis, preferring LLM orchestration when configured.

        Returns
        -------
        ClinicalReport
            The assembled structured report.
        """

        if settings.LLM_API_KEY:
            return self.run_llm()
        return self.run_analysis()

    def _build_query(self, prediction: PredictionResult | None) -> str:
        """Build an evidence query from the prediction (or a generic one)."""
        if prediction is None:
            return "clinical management and monitoring recommendations"
        return (
            f"clinical evidence and management for {prediction.predicted_class} "
            f"at {prediction.confidence:.0%} confidence"
        )

    @staticmethod
    def _parse_report(result: object) -> ClinicalReport | None:
        """Parse a crew kickoff result into a ClinicalReport if possible."""
        text = str(result)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
            return ClinicalReport.model_validate(payload)
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            logger.warning("Report parse failed: %s", error)
            return None


__all__ = ["ClinicalCrew"]
