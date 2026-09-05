"""
Agent definitions for the healthcare crew.

Agents orchestrate reasoning and consume the outputs of the
preprocessing, prediction, and retrieval modules through tools. They
never implement machine learning themselves.

Five lean agents (patient summary and explanation are folded into the
prediction and report tasks, so no dedicated agents are needed).
"""

from __future__ import annotations

from .config import settings


def _agent_llm() -> str | dict[str, object]:
    """Build the LLM identifier for agents.

    Returns a native ``provider/model`` string when ``LLM_BASE_URL`` is
    empty; otherwise returns a CrewAI config dict for a custom
    OpenAI-compatible endpoint (e.g. NVIDIA NIM) with the key and
    temperature already wired.
    """
    if settings.LLM_BASE_URL:
        return {
            "model": settings.LLM_MODEL,
            "custom_openai": True,
            "base_url": settings.LLM_BASE_URL,
            "api_key": settings.LLM_API_KEY,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "timeout": settings.LLM_TIMEOUT_SECONDS,
            # Free-tier shared pools throttle per-minute; retry with
            # backoff instead of failing the whole agent on a 429.
            "max_retries": settings.LLM_MAX_RETRIES,
        }
    return f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}"


def create_report_agent(llm: str | dict[str, object] | None = None) -> object:
    """Build the single LLM agent used for fast narrative enrichment.

    The expensive multi-agent chain is intentionally avoided. Prediction,
    risk assessment, RAG retrieval, and deterministic recommendations are
    completed in Python; this agent only turns those verified values into
    concise human-readable report text.
    """
    from crewai import Agent

    kwargs: dict[str, object] = {
        "role": "Clinical Report Writer",
        "goal": (
            "Turn verified analysis data into a concise, safe clinical "
            "report narrative."
        ),
        "backstory": (
            "You are a clinical report editor. All predictions, "
            "probabilities, risk scores, and evidence are supplied by "
            "deterministic software. Never change those values, invent "
            "evidence, invent diagnoses, or prescribe beyond the supplied "
            "recommendations."
        ),
        "verbose": settings.CREW_VERBOSE,
        "allow_delegation": False,
        "max_iter": 1,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "max_execution_time": settings.LLM_TIMEOUT_SECONDS,
        "max_rpm": settings.LLM_MAX_RPM,
    }
    if llm is not None:
        kwargs["llm"] = llm
    return Agent(**kwargs)


__all__ = ["create_report_agent"]
