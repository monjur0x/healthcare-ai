"""
The healthcare clinical crew.

``ClinicalCrew`` offers two execution paths over the same deterministic
services:

- ``run_analysis`` — a fully offline, deterministic pipeline executed as
  seven discrete agent steps with full per-agent tracing (M4 DoD).
- ``run_llm`` — the same pipeline plus CrewAI agent orchestration for
  narrative enrichment. Only enabled when ``CREW_LLM_API_KEY`` is set.

Both paths produce per-agent execution traces proving multi-agent
reasoning contributed to the system.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time

from collections.abc import Mapping

import numpy as np

from preprocessing.logger import get_logger

from .agent_tracing import AgentTrace, CrewTrace
from .config import settings
from .exceptions import LLMNotConfiguredError, OrchestrationError
from .metrics import compute_agent_metrics
from .schemas import ClinicalReport, PatientInfo
from .services import (
    assemble_clinical_report,
    assess_risk,
    retrieve_evidence,
    run_image_prediction,
    run_prediction,
)

logger = get_logger(__name__)

_CREWAI_AVAILABLE = importlib.util.find_spec("crewai") is not None


class ClinicalCrew:
    """Orchestrates a patient analysis into a structured clinical report.

    The deterministic path executes seven discrete agents in sequence:
        Patient Analyst → Disease Predictor → Medical Researcher →
        Treatment Planner → Explainability Expert → Risk Monitor →
        Report Writer

    Each agent's input, output, execution time, and status are logged
    and collected into a :class:`CrewTrace`.
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
        preprocessed: bool = False,
    ) -> None:
        self.patient = patient
        self.input_type = input_type
        self._model = model
        self._features = dict(features or {})
        self._preprocessed = preprocessed
        self._image_model = image_model
        self._image = image
        self._rag_pipeline = rag_pipeline
        self._markers = dict(markers or {})
        self._recommendations = list(recommendations or [])
        #: Populated after run_analysis() with per-agent traces.
        self.crew_trace: CrewTrace | None = None

    # ------------------------------------------------------------------
    # Deterministic multi-agent pipeline (M4 DoD)
    # ------------------------------------------------------------------

    def run_analysis(self) -> ClinicalReport:
        """Run the deterministic analysis as seven discrete agent steps.

        Returns
        -------
        ClinicalReport
            The assembled structured report with ``crew_trace`` attached.
        """
        self.crew_trace = CrewTrace(patient_id=self.patient.id)
        t0 = time.perf_counter()

        # ── Agent 1: Patient Analyst ────────────────────────────────
        step1 = AgentTrace(
            agent_name="Patient Analyst",
            role="Clinical Data Analyst",
            task_description="Analyze patient features and summarize clinical data",
        )
        self.crew_trace.steps.append(step1)
        s = time.perf_counter()
        try:
            feature_summary = self._patient_analyst()
            step1.input_summary = (
                f"patient={self.patient.id}, features={list(self._features.keys())[:6]}"
            )
            step1.output_summary = feature_summary
            step1.execution_time_s = time.perf_counter() - s
            step1.status = "SUCCESS"
            logger.info("[AGENT 1/7 ✓] Patient Analyst (%.4fs)", step1.execution_time_s)
        except Exception as e:
            step1.execution_time_s = time.perf_counter() - s
            step1.status = "FAILED"
            step1.output_summary = str(e)
            logger.error("[AGENT 1/7 ✗] Patient Analyst FAILED: %s", e)
            raise

        # ── Agent 2: Disease Predictor ──────────────────────────────
        step2 = AgentTrace(
            agent_name="Disease Predictor",
            role="Clinical Prediction Specialist",
            task_description="Run ML prediction on feature row",
        )
        self.crew_trace.steps.append(step2)
        s = time.perf_counter()
        prediction = risk = None
        try:
            if self._image_model is not None:
                if self._image is None:
                    raise OrchestrationError("Image model requires image array.")
                prediction = run_image_prediction(self._image_model, self._image)
            elif self._model is not None:
                if not self._features:
                    raise OrchestrationError("Prediction requires feature row.")
                prediction = run_prediction(
                    self._model, self._features, preprocessed=self._preprocessed
                )
            risk = assess_risk(prediction, self._markers)
            step2.input_summary = f"features={len(self._features)} cols"
            step2.output_summary = (
                f"pred={prediction.predicted_class} conf={prediction.confidence:.4f} "
                f"risk={risk.risk_level}"
            )
            step2.output_data = {"prediction": prediction, "risk": risk}
            step2.execution_time_s = time.perf_counter() - s
            step2.status = "SUCCESS" if prediction else "SKIPPED"
            icon = "✓" if prediction else "○"
            logger.info(
                "[AGENT 2/7 %s] Disease Predictor (%.4fs) pred=%s conf=%.4f",
                icon,
                step2.execution_time_s,
                prediction.predicted_class if prediction else "N/A",
                prediction.confidence if prediction else 0,
            )
        except Exception as e:
            step2.execution_time_s = time.perf_counter() - s
            step2.status = "FAILED"
            step2.output_summary = str(e)
            logger.error("[AGENT 2/7 ✗] Disease Predictor FAILED: %s", e)
            raise

        # ── Agent 3: Medical Researcher (RAG) ───────────────────────
        step3 = AgentTrace(
            agent_name="Medical Researcher",
            role="Clinical Research Specialist",
            task_description="Retrieve clinical evidence from RAG knowledge base",
        )
        self.crew_trace.steps.append(step3)
        s = time.perf_counter()
        evidence = []
        try:
            if self._rag_pipeline is not None:
                query = self._build_query(prediction)
                evidence = retrieve_evidence(self._rag_pipeline, query)
            step3.input_summary = f"query={query!r}" if prediction else "(no query)"
            step3.output_summary = f"{len(evidence)} evidence items"
            step3.output_data = evidence
            step3.execution_time_s = time.perf_counter() - s
            step3.status = "SUCCESS" if evidence else "SKIPPED"
            icon = "✓" if evidence else "○"
            logger.info(
                "[AGENT 3/7 %s] Medical Researcher (%.4fs) evidence=%d",
                icon,
                step3.execution_time_s,
                len(evidence),
            )
        except Exception as e:  # noqa: BLE001
            step3.execution_time_s = time.perf_counter() - s
            step3.status = "FAILED"
            step3.output_summary = str(e)
            logger.error("[AGENT 3/7 ✗] Medical Researcher FAILED: %s", e)

        # ── Agent 4: Treatment Planner ──────────────────────────────
        step4 = AgentTrace(
            agent_name="Treatment Planner",
            role="Treatment Planner",
            task_description="Generate evidence-based treatment recommendations",
        )
        self.crew_trace.steps.append(step4)
        s = time.perf_counter()
        recommendations = list(self._recommendations)
        monitoring_schedule = []
        try:
            if risk:
                monitoring_schedule = risk.monitoring_schedule
            for ev in evidence[:2]:
                text_snippet = ev.text[:120].replace("\n", " ").strip()
                recommendations.append(f"Evidence-based: {text_snippet}")
            step4.input_summary = (
                f"risk={risk.risk_level if risk else 'N/A'}, evidence={len(evidence)}"
            )
            step4.output_summary = (
                f"{len(recommendations)} recs, "
                f"{len(monitoring_schedule)} monitoring items"
            )
            step4.output_data = {
                "recommendations": recommendations,
                "monitoring": monitoring_schedule,
            }
            step4.execution_time_s = time.perf_counter() - s
            step4.status = "SUCCESS"
            logger.info(
                "[AGENT 4/7 ✓] Treatment Planner (%.4fs) recs=%d",
                step4.execution_time_s,
                len(recommendations),
            )
        except Exception as e:  # noqa: BLE001
            step4.execution_time_s = time.perf_counter() - s
            step4.status = "FAILED"
            step4.output_summary = str(e)
            logger.error("[AGENT 4/7 ✗] Treatment Planner FAILED: %s", e)

        # ── Agent 5: Explainability Expert ──────────────────────────
        step5 = AgentTrace(
            agent_name="Explainability Expert",
            role="Medical AI Explainer",
            task_description="Explain why the model made its prediction",
        )
        self.crew_trace.steps.append(step5)
        s = time.perf_counter()
        explanation_parts = []
        try:
            if prediction:
                top_features = sorted(
                    self._features.items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
                explanation_parts.append(
                    "Prediction driven primarily by: "
                    + ", ".join(f"{k}={v}" for k, v in top_features)
                )
            if risk and risk.risk_factors:
                explanation_parts.append(
                    "Risk factors: " + "; ".join(risk.risk_factors)
                )
            explanation_parts.append(
                f"Model confidence: {prediction.confidence:.1%}" if prediction else ""
            )
            step5.input_summary = (
                f"prediction={prediction.predicted_class if prediction else 'N/A'}"
            )
            step5.output_summary = "; ".join(explanation_parts)[:200]
            step5.output_data = {"explanation": explanation_parts}
            step5.execution_time_s = time.perf_counter() - s
            step5.status = "SUCCESS" if prediction else "SKIPPED"
            logger.info(
                "[AGENT 5/7 ✓] Explainability Expert (%.4fs)",
                step5.execution_time_s,
            )
        except Exception as e:  # noqa: BLE001
            step5.execution_time_s = time.perf_counter() - s
            step5.status = "FAILED"
            step5.output_summary = str(e)
            logger.error("[AGENT 5/7 ✗] Explainability Expert FAILED: %s", e)

        # ── Agent 6: Risk Monitor ───────────────────────────────────
        step6 = AgentTrace(
            agent_name="Risk Monitor",
            role="Risk Assessment Specialist",
            task_description="Continuous risk evaluation and alert generation",
        )
        self.crew_trace.steps.append(step6)
        s = time.perf_counter()
        alert_status = "none"
        try:
            if risk:
                alert_status = (
                    "ALERT"
                    if risk.risk_level == "high"
                    else "WATCH"
                    if risk.risk_level == "medium"
                    else "CLEAR"
                )
            step6.input_summary = f"risk_score={risk.risk_score if risk else 'N/A'}"
            step6.output_summary = (
                f"status={alert_status}, schedule={len(monitoring_schedule)} items"
            )
            step6.output_data = {
                "alert_status": alert_status,
                "monitoring_schedule": monitoring_schedule,
            }
            step6.execution_time_s = time.perf_counter() - s
            step6.status = "SUCCESS" if risk else "SKIPPED"
            icon = "✓" if risk else "○"
            logger.info(
                "[AGENT 6/7 %s] Risk Monitor (%.4fs) status=%s",
                icon,
                step6.execution_time_s,
                alert_status,
            )
        except Exception as e:  # noqa: BLE001
            step6.execution_time_s = time.perf_counter() - s
            step6.status = "FAILED"
            step6.output_summary = str(e)
            logger.error("[AGENT 6/7 ✗] Risk Monitor FAILED: %s", e)

        # ── Agent 7: Report Writer ──────────────────────────────────
        step7 = AgentTrace(
            agent_name="Report Writer",
            role="Medical Report Writer",
            task_description="Merge all prior outputs into structured clinical report",
        )
        self.crew_trace.steps.append(step7)
        s = time.perf_counter()
        try:
            report = assemble_clinical_report(
                patient=self.patient,
                input_type=self.input_type,
                prediction=prediction,
                risk=risk,
                evidence=evidence,
                recommendations=recommendations,
            )

            # Attach agent metrics from real execution traces
            successful = [s for s in self.crew_trace.steps if s.status == "SUCCESS"]
            outputs = [s.output_data for s in successful]
            predicted_str = str(prediction.predicted_class) if prediction else ""
            report.agent_metrics = compute_agent_metrics(
                outputs, [predicted_str]
            ).to_dict()

            # Attach crew trace to report context
            report.context = "\n".join(explanation_parts)

            step7.input_summary = f"prediction+risk+evidence({len(evidence)})+recs({len(recommendations)})"
            step7.output_summary = (
                f"ClinicalReport with {len(report.evidence)} evidence items"
            )
            step7.output_data = report.to_dict()
            step7.execution_time_s = time.perf_counter() - s
            step7.status = "SUCCESS"
            logger.info(
                "[AGENT 7/7 ✓] Report Writer (%.4fs) evidence=%d recs=%d",
                step7.execution_time_s,
                len(report.evidence),
                len(report.recommendations),
            )
        except Exception as e:
            step7.execution_time_s = time.perf_counter() - s
            step7.status = "FAILED"
            step7.output_summary = str(e)
            logger.error("[AGENT 7/7 ✗] Report Writer FAILED: %s", e)
            raise

        # ── Finalise trace ──────────────────────────────────────────
        self.crew_trace.total_time_s = time.perf_counter() - t0
        self.crew_trace.all_succeeded = all(
            s.status == "SUCCESS" for s in self.crew_trace.steps
        )

        logger.info("\n%s", self.crew_trace.summary())
        return report

    # ------------------------------------------------------------------
    # CrewAI LLM-enriched path
    # ------------------------------------------------------------------

    def run_llm(self) -> ClinicalReport:
        """Run the CrewAI-orchestrated pipeline (requires an LLM key)."""
        if not _CREWAI_AVAILABLE:
            raise OrchestrationError(
                "CrewAI is not installed. Install with 'pip install crewai' "
                "or use run_analysis()."
            )
        if not settings.LLM_API_KEY:
            raise LLMNotConfiguredError("LLM orchestration requires CREW_LLM_API_KEY.")
        if (
            not settings.LLM_BASE_URL
            and not os.environ.get("GEMINI_API_KEY")
            and not os.environ.get("GOOGLE_API_KEY")
        ):
            os.environ["GEMINI_API_KEY"] = settings.LLM_API_KEY

        base = self.run_analysis()
        try:
            from crewai import Crew, Process

            from .agents import _agent_llm, create_agents
            from .tasks import create_tasks
        except Exception as error:
            raise OrchestrationError(f"CrewAI import failed: {error}") from error

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
        except Exception as error:
            logger.error("Crew kickoff failed: %s", error)
            return base

        parsed = self._parse_report(result)
        if parsed is None:
            logger.warning("Could not parse crew result; returning base report.")
            return base
        report = self._merge_llm_over_base(base, parsed)

        # Carry over the deterministic crew trace
        if self.crew_trace:
            report.agent_metrics = {
                **(report.agent_metrics or {}),
                "deterministic_agents_completed": self.crew_trace.completed_count,
                "deterministic_agents_total": self.crew_trace.total_agents,
            }

        logger.info("LLM analysis complete for patient %s", self.patient.id)
        return report

    def run(self) -> ClinicalReport:
        """Prefer LLM orchestration when configured; fall back to deterministic."""
        if settings.LLM_API_KEY and _CREWAI_AVAILABLE:
            return self.run_llm()
        return self.run_analysis()

    def _build_query(self, prediction) -> str:
        if prediction is None:
            return "clinical management and monitoring recommendations"
        return (
            f"clinical evidence and management for {prediction.predicted_class} "
            f"at {prediction.confidence:.0%} confidence"
        )

    def _patient_analyst(self) -> str:
        """Summarize patient features for downstream agents."""
        parts = [
            f"Patient {self.patient.name} ({self.patient.id})",
            f"age {self.patient.age}" if self.patient.age else "",
            f"input type: {self.input_type}",
        ]
        if self._markers:
            marker_strs = [f"{k}={v}" for k, v in sorted(self._markers.items())]
            parts.append("markers: " + ", ".join(marker_strs))
        if self._features:
            abnormal = [
                k
                for k, v in self._features.items()
                if isinstance(v, (int, float)) and (v > 200 or v < 0)
            ]
            if abnormal:
                parts.append(f"outlier features: {abnormal}")
        return "; ".join(p for p in parts if p)

    @staticmethod
    def _parse_report(result: object):
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

    @staticmethod
    def _merge_llm_over_base(
        base: ClinicalReport, llm: ClinicalReport
    ) -> ClinicalReport:
        merged = base.model_copy()
        merged.patient_summary = llm.patient_summary or base.patient_summary
        merged.context = llm.context or base.context
        if llm.recommendations:
            merged.recommendations = llm.recommendations
        if llm.limitations:
            merged.limitations = llm.limitations
        if llm.doctor_notice:
            merged.doctor_notice = llm.doctor_notice
        if merged.risk is None and llm.risk is not None:
            merged.risk = llm.risk
        if llm.agent_metrics is not None:
            merged.agent_metrics = llm.agent_metrics
        return merged


__all__ = ["ClinicalCrew"]
