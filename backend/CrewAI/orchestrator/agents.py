"""
Agent definitions for the healthcare crew.

Agents orchestrate reasoning and consume the outputs of the
preprocessing, prediction, and retrieval modules through tools. They
never implement machine learning themselves.
"""

from __future__ import annotations

from collections.abc import Mapping

from .config import settings
from .prompts import AGENT_PROFILES


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
        }
    return f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}"


def create_agents(
    tools: Mapping[str, object], llm: str | dict[str, object] | None = None
) -> dict[str, object]:
    """
    Build the seven healthcare agents.

    Parameters
    ----------
    tools : Mapping[str, object]
        Tool instances keyed by name (prediction, evidence retrieval,
        risk assessment, clinical report, ...).
    llm : str | dict[str, object] | None
        Optional LLM identifier (e.g. ``"google/gemini-3.7-flash"``) or
        CrewAI config dict for a custom OpenAI-compatible endpoint.
        When omitted the agent is constructed without an explicit LLM so
        construction stays hermetic; the configured provider/model is
        used by the LLM orchestration path.

    Returns
    -------
    dict[str, object]
        Agents keyed by role name.
    """

    from crewai import Agent

    common: dict[str, object] = {
        "verbose": settings.CREW_VERBOSE,
        "allow_delegation": False,
        "max_iter": settings.LLM_MAX_ITERATIONS,
    }
    if llm is not None:
        common["llm"] = llm
    tool_map = {
        "patient_analyst": ["csv_summary"],
        "disease_predictor": ["disease_prediction", "risk_assessment"],
        "medical_researcher": ["evidence_retrieval"],
        "treatment_planner": ["evidence_retrieval"],
        "explainability_expert": ["disease_prediction"],
        "risk_monitor": ["risk_assessment"],
        "report_writer": ["clinical_report"],
    }

    agents: dict[str, object] = {}
    for name, profile in AGENT_PROFILES.items():
        agent_tools = [
            tools[tool_name] for tool_name in tool_map[name] if tool_name in tools
        ]
        agents[name] = Agent(
            role=profile["role"],
            goal=profile["goal"],
            backstory=profile["backstory"],
            tools=agent_tools,
            **common,
        )
    return agents


__all__ = ["create_agents"]
