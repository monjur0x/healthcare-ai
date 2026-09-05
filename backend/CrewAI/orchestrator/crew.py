"""
The healthcare clinical crew.

``ClinicalCrew`` offers two execution paths over the same deterministic
services:

- ``run_analysis`` — a fully offline, deterministic pipeline executed as
  seven discrete agent steps with full per-agent tracing (M4 DoD).
- ``run_llm`` — a fast single-agent CrewAI enrichment pass. Prediction,
  risk scoring, RAG retrieval, and recommendations remain deterministic;
  one LLM call is used only to polish the final narrative.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
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
    build_disease_query,
    build_rag_topic,
    build_treatment_recommendations,
    enrich_prediction,
    resolve_disease,
    retrieve_evidence,
    run_image_prediction,
    run_prediction,
    summarize_patient,
)

logger = get_logger(__name__)

_CREWAI_AVAILABLE = importlib.util.find_spec("crewai") is not None

#: Error-message markers that mean retrying is pointless (bad key,
#: unknown model, forbidden tier, denied permission). Everything else
#: (rate limits, timeouts, empty completions, connection drops) is
#: treated as transient.
_NON_RETRYABLE_LLM_MARKERS = (
    "401",
    "403",
    "invalid api key",
    "incorrect api key",
    "invalid_api_key",
    "tier_forbidden",
    "model_not_found",
    "does not exist",
    "not found",
    "access denied",
    "permissiondenied",
    "permission_denied",
    "account deactivated",
    "invalid_request_error",
)


def _retryable_llm_error(error: Exception) -> bool:
    """True when a failed kickoff is worth retrying after backoff."""
    message = str(error).lower().replace(" ", "").replace("_", "")
    markers = [m.replace(" ", "").replace("_", "") for m in _NON_RETRYABLE_LLM_MARKERS]
    return not any(marker in message for marker in markers)


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
        disease: str | None = None,
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
        #: Resolved clinical context for the assessed dataset preset;
        #: ``None`` when the analysis has no disease mapping.
        self._disease_context = resolve_disease(disease)
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
                prediction = enrich_prediction(
                    run_prediction(
                        self._model, self._features, preprocessed=self._preprocessed
                    ),
                    self._disease_context,
                )
            risk = (
                assess_risk(
                    prediction, self._markers, disease_context=self._disease_context
                )
                if prediction is not None
                else None
            )
            step2.input_summary = f"features={len(self._features)} cols"
            if prediction is None:
                step2.output_summary = "SKIPPED: no prediction model configured"
            else:
                step2.output_summary = (
                    f"disease={prediction.disease or 'N/A'} "
                    f"label={prediction.predicted_label} "
                    f"p_pos={prediction.positive_probability} risk={risk.risk_level}"
                )
            step2.output_data = {"prediction": prediction, "risk": risk}
            step2.execution_time_s = time.perf_counter() - s
            step2.status = "SUCCESS" if prediction else "SKIPPED"
            icon = "✓" if prediction else "○"
            logger.info(
                "[AGENT 2/7 %s] Disease Predictor (%.4fs) label=%s p_pos=%s risk=%s",
                icon,
                step2.execution_time_s,
                prediction.predicted_label if prediction else "N/A",
                prediction.positive_probability if prediction else "N/A",
                risk.risk_level if risk else "N/A",
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
        evidence: list = []
        query: str | None = None
        try:
            if self._rag_pipeline is not None and prediction is not None:
                query = self._build_query(prediction)
                topic = build_rag_topic(prediction)
                evidence = retrieve_evidence(self._rag_pipeline, query, topic=topic)
            step3.input_summary = (
                f"query={query!r}, topic={build_rag_topic(prediction)!r}"
                if query
                else "(no query)"
            )
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
            playbook_recs, _ = build_treatment_recommendations(prediction, risk)
            recommendations.extend(playbook_recs)
            # One evidence-derived pointer (source label, not a raw text
            # dump) so the report stays traceable to retrieved knowledge.
            for ev in evidence[:1]:
                recommendations.append(
                    f"Evidence source consulted: {ev.document_id} "
                    f"(topics: {', '.join(ev.topics) or 'general'})"
                )
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
                outcome = prediction.predicted_label or prediction.predicted_class
                if prediction.disease and prediction.positive_probability is not None:
                    disease_name = prediction.disease.replace("_", " ")
                    explanation_parts.append(
                        f"Predicted outcome: {outcome}; {disease_name} "
                        f"probability {prediction.positive_probability:.1%} "
                        f"(model confidence in this class: "
                        f"{prediction.confidence:.1%})"
                    )
                else:
                    explanation_parts.append(
                        f"Predicted outcome: {outcome} "
                        f"(model confidence {prediction.confidence:.1%})"
                    )
                explanation_parts.append(
                    "Prediction driven primarily by: "
                    + ", ".join(f"{k}={v}" for k, v in top_features)
                )
            if risk and risk.risk_factors:
                explanation_parts.append(
                    "Risk factors: " + "; ".join(risk.risk_factors)
                )
            step5.input_summary = (
                f"prediction={prediction.predicted_label if prediction else 'N/A'}"
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

            step7.input_summary = (
                f"prediction+risk+evidence({len(evidence)})"
                f"+recs({len(recommendations)})"[:200]
            )
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

    def run(self) -> ClinicalReport:
        """Prefer LLM orchestration when configured; fall back to deterministic."""
        if settings.LLM_API_KEY and _CREWAI_AVAILABLE:
            return self.run_llm()
        return self.run_analysis()

    # ------------------------------------------------------------------
    # CrewAI LLM-enriched path (lean 5-agent crew)
    # ------------------------------------------------------------------

    def run_llm(self) -> ClinicalReport:
        """Run one CrewAI LLM call over the already-verified analysis.

        The previous implementation launched five sequential agents after
        running the deterministic pipeline, resulting in up to five model
        round-trips (plus retries). This version deliberately keeps all
        factual computation in Python and uses CrewAI for one final writing
        pass only.
        """
        if not _CREWAI_AVAILABLE:
            raise OrchestrationError(
                "CrewAI is not installed. Install with 'pip install crewai' "
                "or use run_analysis()."
            )
        if not settings.LLM_API_KEY:
            raise LLMNotConfiguredError("LLM orchestration requires CREW_LLM_API_KEY.")

        # Run deterministic inference/RAG exactly once. This is fast and
        # gives the LLM trusted structured inputs instead of asking it to
        # call tools and repeat work.
        base = self.run_analysis()

        try:
            from crewai import Crew, Process, Task

            from .agents import _agent_llm, create_report_agent
        except Exception as error:
            raise OrchestrationError(f"CrewAI import failed: {error}") from error

        evidence_text = []
        for item in base.evidence[: settings.RAG_TOP_K]:
            evidence_text.append(
                f"[{item.document_id}] source={item.source!r} "
                f"score={item.score:.4f}\n{item.text}"
            )

        prediction = base.prediction.model_dump() if base.prediction else None
        risk = base.risk.model_dump() if base.risk else None
        prompt = {
            "patient": self.patient.model_dump(),
            "input_type": self.input_type,
            "prediction": prediction,
            "risk": risk,
            "evidence": evidence_text,
            "verified_recommendations": base.recommendations,
            "verified_context": base.context,
        }

        task_description = (
            "Create concise narrative enrichment for the verified clinical "
            "analysis below. Do NOT recalculate prediction or risk. Do NOT "
            "change any numeric value, diagnosis, evidence source, or "
            "monitoring schedule. Do NOT invent citations. Return ONLY "
            "valid JSON with exactly these keys: patient_summary, "
            "context, recommendations, limitations, doctor_notice. "
            "recommendations must be an array of strings. Keep the "
            "response under 500 words. The report is decision support "
            "only and must be reviewed by a licensed physician.\n\n"
            f"VERIFIED ANALYSIS:\n{json.dumps(prompt, ensure_ascii=False, default=str)}"
        )

        agent = create_report_agent(llm=_agent_llm())
        task = Task(
            description=task_description,
            expected_output=(
                'JSON only: {"patient_summary":"", "context":"", "recommendations":[], '
                '"limitations":"", "doctor_notice":""}'
            ),
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=settings.CREW_VERBOSE,
            memory=False,
            planning=False,
        )

        # One kickoff attempt. Provider-level retry is capped at one retry
        # in config.py, so transient errors cannot turn into multi-minute waits.
        result, failure = self._kickoff_with_retries(
            crew, base, max(1, settings.LLM_KICKOFF_MAX_ATTEMPTS)
        )
        if failure is not None:
            self._mark_llm_path(base, failure)
            return base

        enrichment = self._parse_enrichment(result)
        if enrichment is None:
            logger.warning(
                "Could not parse LLM enrichment; returning deterministic report."
            )
            self._mark_llm_path(base, "fallback:parse")
            return base

        merged = base.model_copy()
        merged.patient_summary = (
            enrichment.get("patient_summary") or base.patient_summary
        )
        merged.context = enrichment.get("context") or base.context
        llm_recs = enrichment.get("recommendations")
        if isinstance(llm_recs, list) and llm_recs:
            # Keep deterministic recommendations authoritative; only append
            # genuinely new narrative items instead of allowing the LLM to
            # replace verified clinical playbook output.
            existing = set(merged.recommendations)
            merged.recommendations = merged.recommendations + [
                str(x) for x in llm_recs if str(x) not in existing
            ]
        merged.limitations = enrichment.get("limitations") or base.limitations
        merged.doctor_notice = enrichment.get("doctor_notice") or base.doctor_notice
        merged.agent_metrics = {
            **(merged.agent_metrics or {}),
            "llm_path": "single_call_enriched",
            "llm_agents": 1,
            "llm_tasks": 1,
        }
        logger.info(
            "Single-call LLM enrichment complete for patient %s", self.patient.id
        )
        return merged

    @staticmethod
    def _mark_llm_path(report: ClinicalReport, status: str) -> None:
        """Record how the report was produced for downstream consumers."""
        report.agent_metrics = {**(report.agent_metrics or {}), "llm_path": status}

    def _kickoff_with_retries(
        self, crew: object, base: ClinicalReport, max_attempts: int
    ) -> tuple[object | None, str | None]:
        """
        Run ``crew.kickoff`` with backoff, falling back gracefully.

        Parameters
        ----------
        crew : object
            Configured CrewAI crew.
        base : ClinicalReport
            Deterministic report used for kickoff inputs.
        max_attempts : int
            Total kickoff tries before giving up (>= 1).

        Returns
        -------
        tuple[object | None, str | None]
            ``(result, None)`` on success, else ``(None,
            "fallback:kickoff:<ErrorName>")``.
        """

        inputs = {
            "base_report": base.to_dict(),
            **(
                {"disease_context": self._disease_context}
                if self._disease_context
                else {}
            ),
        }
        attempts = max(1, max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                return crew.kickoff(inputs=inputs), None
            except Exception as error:  # noqa: BLE001
                last = attempt >= attempts
                if not _retryable_llm_error(error) or last:
                    logger.error(
                        "Crew kickoff failed (attempt %d/%d): %s",
                        attempt,
                        attempts,
                        error,
                    )
                    return None, f"fallback:kickoff:{type(error).__name__}"
                delay = settings.LLM_RETRY_BACKOFF_S * (2 ** (attempt - 1))
                delay += random.uniform(0, min(delay, 10.0))
                logger.warning(
                    "Crew kickoff failed (attempt %d/%d, retrying in %.0fs): %s",
                    attempt,
                    attempts,
                    delay,
                    error,
                )
                time.sleep(delay)
        return None, "fallback:kickoff:exhausted"  # unreachable; guards the loop

    def _build_query(self, prediction) -> str:
        """Build a disease-anchored RAG query including elevated markers."""
        return build_disease_query(prediction, markers=self._markers)

    def _patient_analyst(self) -> str:
        """Summarize patient features for downstream agents."""
        return summarize_patient(
            self.patient,
            features=self._features,
            markers=self._markers,
            input_type=self.input_type,
        )

    @staticmethod
    def _repair_json(text: str) -> str:
        """Apply bounded fixes for common small-model JSON mistakes.

        Strips markdown code fences, removes trailing commas, replaces
        control characters inside the payload, and normalizes curly
        quotes. Deliberately conservative: no structural rewriting.
        """
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        cleaned = cleaned.replace("“", '"').replace("”", '"')
        cleaned = " ".join(cleaned.splitlines())
        return cleaned

    def _parse_enrichment(self, result: object) -> dict[str, object] | None:
        """Parse the small JSON object returned by the single writer agent."""
        text = str(result)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = text[start : end + 1]
        try:
            payload = json.loads(self._repair_json(candidate))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            logger.warning(
                "Enrichment JSON parse failed: %s; raw head: %.500s",
                error,
                text,
            )
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _parse_report(self, result: object) -> ClinicalReport | None:
        text = str(result)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = text[start : end + 1]
        for attempt_text in (candidate, ClinicalCrew._repair_json(candidate)):
            try:
                payload = json.loads(attempt_text)
                return ClinicalReport.model_validate(payload)
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                last_error = error
        logger.warning("Report parse failed: %s", last_error)
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
