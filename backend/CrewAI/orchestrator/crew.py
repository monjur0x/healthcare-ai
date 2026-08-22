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

import importlib.util
import json
import os

from collections.abc import Mapping

import numpy as np

from preprocessing.logger import get_logger

from .config import settings
from .exceptions import LLMNotConfiguredError, OrchestrationError
from .crew_logging import wrap_crew_for_logging
from .crew_logging import wrap_crew_for_logging
from .metrics import compute_agent_metrics
from .schemas import (
    ClinicalReport,
    PatientInfo,
)
from .services import (
    assemble_clinical_report,
    assess_risk,
    retrieve_evidence,
    run_image_prediction,
    run_prediction,
)

logger = get_logger(__name__)

# CrewAI is optional - check availability
_CREWAI_AVAILABLE = importlib.util.find_spec("crewai") is not None


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
    preprocessed : bool
        True when ``features`` were already transformed by the training
        pipeline (CSV path); False applies the model's persisted scaler
        to raw feature values.
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
        self._markers = dict(markers or [])
        self._recommendations = list(recommendations or [])

    def run_analysis(self):
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
            prediction = run_prediction(
                self._model, self._features, preprocessed=self._preprocessed
            )
            risk = assess_risk(prediction, self._markers)

        evidence = []
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

    def run_llm(self):
        """
        Run the CrewAI-orchestrated pipeline (requires an LLM key).

        Enriches the deterministic analysis with agent reasoning. The LLM
        is the native provider/model pair, or a custom OpenAI-compatible
        endpoint (e.g. NVIDIA NIM) when ``CREW_LLM_BASE_URL`` is set. If
        the crew's JSON output cannot be parsed, the deterministic report
        is returned instead.

        Returns
        -------
        ClinicalReport
            The assembled structured report.

        Raises
        ------
        LLMNotConfiguredError
            If ``CREW_LLM_API_KEY`` is not configured.
        OrchestrationError
            If the crew cannot be built or CrewAI is not available.
        """

        if not _CREWAI_AVAILABLE:
            raise OrchestrationError(
                "CrewAI is not installed. Install with 'pip install crewai' "
                "to enable LLM orchestration, or use run_analysis() for the "
                "offline deterministic path."
            )

        if not settings.LLM_API_KEY:
            raise LLMNotConfiguredError(
                "LLM orchestration requires CREW_LLM_API_KEY; use "
                "run_analysis() for the offline deterministic path."
            )

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
        crew = wrap_crew_for_logging(crew)
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
        report = self._merge_llm_over_base(base, parsed)

        # §12 Agent metrics: task completion / collaboration from the real
        # crew task outputs, decision consistency from the deterministic
        # prediction vs the crew-merged prediction (single observation).
        try:
            task_outputs = [task.output for task in tasks.values() if task.output]
            predicted = (
                str(report.prediction.predicted_class) if report.prediction else ""
            )
            report.agent_metrics = compute_agent_metrics(
                task_outputs, [predicted]
            ).to_dict()
        except Exception as error:  # noqa: BLE001 - metrics never block care
            logger.warning("Agent metrics computation failed: %s", error)

        logger.info("LLM analysis complete for patient %s", self.patient.id)
        return report

    def run(self):
        """
        Run the analysis, preferring LLM orchestration when configured.

        Returns
        -------
        ClinicalReport
            The assembled structured report.
        """

        if settings.LLM_API_KEY and _CREWAI_AVAILABLE:
            return self.run_llm()
        return self.run_analysis()

    def _build_query(self, prediction) -> str:
        """Build an evidence query from the prediction (or a generic one)."""
        if prediction is None:
            return "clinical management and monitoring recommendations"
        return (
            f"clinical evidence and management for {prediction.predicted_class} "
            f"at {prediction.confidence:.0%} confidence"
        )

    @staticmethod
    def _parse_report(result: object):
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

    @staticmethod
    def _merge_llm_over_base(base, llm):
        """Merge an LLM report over the deterministic base report."""
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

# Add detailed logging for agent execution
import time
import json
from functools import wraps

def _log_agent_execution(agent, task, inputs, output, execution_time, status, error=None):
    """Log detailed agent execution information."""
    log_data = {
        "agent": getattr(agent, 'role', 'Unknown'),
        "task": getattr(task, 'description', '')[:100] if task else 'Unknown',
        "execution_time_seconds": round(execution_time, 3),
        "status": status,
        "input_keys": list(inputs.keys()) if inputs else [],
        "output_preview": str(output)[:500] if output else None,
        "error": str(error) if error else None
    }
    
    if error:
        logger.error(f"[AGENT FAILED] {json.dumps(log_data, default=str)[:500]}")
    else:
        logger.info(f"[AGENT SUCCESS] {json.dumps(log_data, default=str)[:500]}")

def _wrap_crew_kickoff(crew):
    """Wrap crew.kickoff to add detailed logging."""
    original_kickoff = crew.kickoff
    
    def logged_kickoff(inputs):
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}")
        start_time = time.time()
        
        try:
            result = crew.kickoff(inputs)
            logger.info(f"[CREW COMPLETE] Time: {time.time() - start_time:.3f}s")
            return result
        except Exception as error:
            logger.error(f"[CREW ERROR] {type(error).__name__}: {error}")
            raise
    
    crew.kickoff = crew.kickoff.__class__(crew.kickoff.__func__, crew)
    crew.kickoff = crew.kickoff.__class__.__get__(lambda inputs: None, crew)
    crew.kickoff = lambda inputs: None  # placeholder
    
    # Actually wrap properly
    original_kickoff = crew.kickoff
    
    def logged_kickoff(inputs):
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}")
        start_time = time.time()
        try:
            result = crew.kickoff(inputs)
            logger.info(f"[CREW COMPLETE] Time: {time.time() - start_time:.3f}s")
            return result
        except Exception as error:
            logger.error(f"[CREW ERROR] {type(error).__name__}: {error}")
            raise
    
    crew.kickoff = lambda inputs: (
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}"),
        setattr(crew, '_kickoff_start', time.time()),
        crew.kickoff(inputs)
    )[-1] if False else None
    
    # Simple approach: monkey patch
    original_kickoff = crew.kickoff
    def logged_kickoff(inputs):
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}")
        start = time.time()
        try:
            result = original_kickoff(inputs)
            logger.info(f"[CREW COMPLETE] Time: {time.time() - start:.3f}s")
            return result
        except Exception as e:
            logger.error(f"[CREW ERROR] {type(e).__name__}: {e}")
            raise
    crew.kickoff = logged_kickoff
    return crew

def wrap_crew_for_logging(crew):
    """Wrap crew's kickoff method with detailed logging."""
    original_kickoff = crew.kickoff
    
    def logged_kickoff(inputs):
        import time
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}")
        start_time = time.time()
        try:
            result = crew.kickoff(inputs)
            logger.info(f"[CREW COMPLETE] Time: {time.time() - start_time:.3f}s")
            return result
        except Exception as error:
            logger.error(f"[CREW ERROR] {type(error).__name__}: {error}")
            raise
    
    crew.kickoff = logged_kickoff
    return crew


def wrap_task_execution(agent, task):
    """Wrap a task's execute method for logging."""
    original_execute = task.execute
    
    def logged_execute(*args, **kwargs):
        start_time = time.time()
        logger.info(f"[TASK START] Agent: {agent.role if hasattr(agent, 'role') else 'Unknown'} | Task: {task.description[:100]}")
        try:
            result = original_execute(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"[TASK END] Agent: {task.agent.role if hasattr(task, 'agent') else 'Unknown'} | Task: {task.description[:100]} | Time: {time.time() - start_time:.3f}s | Status: SUCCESS")
            return result
        except Exception as e:
            logger.error(f"[TASK ERROR] Task: {task.description[:100]} | Time: {time.time() - start_time:.3f}s | Error: {e}")
            raise
    
    task.execute = wrapped_execute
    return task


def wrap_crew_tasks(crew):
    """Wrap all tasks in the crew for detailed logging."""
    for task in crew.tasks:
        wrap_task_execution(task.agent, task)
    return crew


# Enhanced logging for CrewAI execution
import time
import json
from functools import wraps

def _wrap_crew_for_logging(crew):
    """Wrap crew's kickoff method with detailed logging."""
    original_kickoff = crew.kickoff
    
    def logged_kickoff(inputs):
        import time
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}")
        start_time = time.time()
        try:
            result = crew.kickoff(inputs)
            logger.info(f"[CREW COMPLETE] Time: {time.time() - start_time:.3f}s")
            return result
        except Exception as error:
            logger.error(f"[CREW ERROR] {type(error).__name__}: {error}")
            raise
    
    crew.kickoff = crew.kickoff.__class__(crew.kickoff.__func__, crew)
    crew.kickoff = crew.kickoff.__class__.__get__(lambda inputs: None, crew)
    
    # Simpler approach - just replace the method
    original_kickoff = crew.kickoff
    def logged_kickoff(inputs):
        import time
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}")
        start_time = time.time()
        try:
            result = original_kickoff(inputs)
            logger.info(f"[CREW COMPLETE] Time: {time.time() - start_time:.3f}s")
            return result
        except Exception as error:
            logger.error(f"[CREW ERROR] {type(error).__name__}: {error}")
            raise
    
    crew.kickoff = logged_kickoff
    return crew


def _wrap_task_execution(agent, task):
    """Wrap a task's execute method for logging."""
    original_execute = task.execute
    
    def logged_execute(*args, **kwargs):
        import time
        logger.info(f"[TASK START] Agent: {agent.role if hasattr(agent, 'role') else 'Unknown'} | Task: {task.description[:100]}")
        start_time = time.time()
        try:
            result = original_execute(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"[TASK END] Agent: {task.agent.role if hasattr(task, 'agent') else 'Unknown'} | Task: {task.description[:100]} | Time: {execution_time:.3f}s | Status: SUCCESS")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[TASK ERROR] Task: {task.description[:100]} | Time: {execution_time:.3f}s | Error: {e}")
            raise
    
    task.execute = logged_execute
    return task


def _wrap_crew_tasks(crew):
    """Wrap all tasks in the crew for detailed logging."""
    for task in crew.tasks:
        wrap_task_execution(task.agent, task)
    return crew


def wrap_crew_for_logging(crew):
    """Apply comprehensive logging to crew execution."""
    crew = _wrap_crew_kickoff(crew)
    crew = _wrap_crew_tasks(crew)
    return crew


# Apply logging to crew in run_llm
def _run_llm_with_logging(self):
    """Run the CrewAI-orchestrated pipeline with detailed logging."""
    
    if not _CREWAI_AVAILABLE:
        raise OrchestrationError(
            "CrewAI is not installed. Install with 'pip install crewai' "
            "to enable LLM orchestration, or use run_analysis() for the "
            "offline deterministic path."
        )

        if not settings.LLM_API_KEY:
            raise LLMNotConfiguredError(
                "LLM orchestration requires CREW_LLM_API_KEY; use "
                "run_analysis() for the offline deterministic path."
            )

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
        crew = wrap_crew_for_logging(crew)
        # Apply logging wrappers
        crew = _wrap_crew_for_logging(crew)
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
        report = self._merge_llm_over_base(base, parsed)

        # §12 Agent metrics: task completion / collaboration from the real
        # crew task outputs, decision consistency from the deterministic
        # prediction vs the crew-merged prediction (single observation).
        try:
            task_outputs = [task.output for task in tasks.values() if task.output]
            predicted = (
                str(report.prediction.predicted_class) if report.prediction else ""
            )
            report.agent_metrics = compute_agent_metrics(
                task_outputs, [predicted]
            ).to_dict()
        except Exception as error:  # noqa: BLE001 - metrics never block care
            logger.warning("Agent metrics computation failed: %s", error)

        logger.info("LLM analysis complete for patient %s", self.patient.id)
        return report


# Enhanced logging for CrewAI execution
import time
import json
from functools import wraps

def _wrap_crew_kickoff(crew):
    """Wrap crew's kickoff method with detailed logging."""
    original_kickoff = crew.kickoff
    
    def logged_kickoff(inputs):
        import time
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}")
        start_time = time.time()
        try:
            result = crew.kickoff(inputs)
            logger.info(f"[CREW COMPLETE] Time: {time.time() - start_time:.3f}s")
            return result
        except Exception as error:
            logger.error(f"[CREW ERROR] {type(error).__name__}: {error}")
            raise
    
    crew.kickoff = crew.kickoff.__class__(crew.kickoff.__func__, crew)
    crew.kickoff = crew.kickoff.__class__.__get__(lambda inputs: None, crew)
    
    # Simpler approach - just replace the method
    original_kickoff = crew.kickoff
    def logged_kickoff(inputs):
        import time
        logger.info(f"[CREW START] Inputs: {list(inputs.keys())}")
        start_time = time.time()
        try:
            result = original_kickoff(inputs)
            logger.info(f"[CREW COMPLETE] Time: {time.time() - start_time:.3f}s")
            return result
        except Exception as error:
            logger.error(f"[CREW ERROR] {type(error).__name__}: {error}")
            raise
    
    crew.kickoff = logged_kickoff
    return crew


def _wrap_task_execution(agent, task):
    """Wrap a task's execute method for logging."""
    original_execute = task.execute
    
    def logged_execute(*args, **kwargs):
        import time
        logger.info(f"[TASK START] Agent: {agent.role if hasattr(agent, 'role') else 'Unknown'} | Task: {task.description[:100]}")
        start_time = time.time()
        try:
            result = original_execute(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"[TASK END] Agent: {task.agent.role if hasattr(task, 'agent') else 'Unknown'} | Task: {task.description[:100]} | Time: {execution_time:.3f}s | Status: SUCCESS")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[TASK ERROR] Task: {task.description[:100]} | Time: {execution_time:.3f}s | Error: {e}")
            raise
    
    task.execute = logged_execute
    return task


def _wrap_crew_tasks(crew):
    """Wrap all tasks in the crew for detailed logging."""
    for task in crew.tasks:
        wrap_task_execution(task.agent, task)
    return crew


def wrap_crew_for_logging(crew):
    """Apply comprehensive logging to crew execution."""
    crew = _wrap_crew_kickoff(crew)
    crew = _wrap_crew_tasks(crew)
    return crew

