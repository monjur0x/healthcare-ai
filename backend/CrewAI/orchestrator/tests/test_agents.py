"""
Tests for agent / task / crew construction (no LLM kickoff).
"""

from __future__ import annotations

import pytest

from CrewAI.orchestrator.agents import _agent_llm, create_agents
from CrewAI.orchestrator.prompts import AGENT_PROFILES, TASK_DESCRIPTIONS
from CrewAI.orchestrator.schemas import PatientInfo
from CrewAI.orchestrator.tasks import create_tasks


@pytest.fixture
def tools() -> dict[str, object]:
    from CrewAI.orchestrator.tools import (
        ClinicalReportTool,
        PredictionTool,
        RAGRetrievalTool,
        RiskAssessmentTool,
    )

    return {
        "disease_prediction": PredictionTool(model=object()),
        "evidence_retrieval": RAGRetrievalTool(pipeline=object()),
        "risk_assessment": RiskAssessmentTool(),
        "clinical_report": ClinicalReportTool(),
    }


def test_create_agents_builds_all_profiles(tools) -> None:
    agents = create_agents(tools)
    assert set(agents) == set(AGENT_PROFILES)
    assert len(agents) == 7


def test_agents_have_bound_tools(tools) -> None:
    agents = create_agents(tools)
    researcher = agents["medical_researcher"]
    assert len(researcher.tools) == 1


def test_create_tasks_chain(tools) -> None:
    agents = create_agents(tools)
    tasks = create_tasks(agents, PatientInfo(name="P", id="p1", age=40))
    assert set(tasks) == set(TASK_DESCRIPTIONS)
    report_task = tasks["report_generation"]
    assert report_task.context


def test_crew_builds_sequential(tools) -> None:
    from crewai import Crew, Process

    agents = create_agents(tools)
    tasks = create_tasks(agents, PatientInfo(id="p1"))
    crew = Crew(
        agents=list(agents.values()),
        tasks=list(tasks.values()),
        process=Process.sequential,
        verbose=False,
        memory=False,
    )
    assert len(crew.agents) == 7
    assert len(crew.tasks) == 7


def test_agent_llm_native_string_when_no_base_url(monkeypatch) -> None:
    from CrewAI.orchestrator.config import settings

    monkeypatch.setattr(settings, "LLM_BASE_URL", "")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "google")
    monkeypatch.setattr(settings, "LLM_MODEL", "gemini-3.7-flash")
    assert _agent_llm() == "google/gemini-3.7-flash"


def test_agent_llm_custom_openai_when_base_url(monkeypatch) -> None:
    from CrewAI.orchestrator.config import settings

    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(settings, "LLM_MODEL", "meta/llama-3.3-70b-instruct")
    monkeypatch.setattr(settings, "LLM_API_KEY", "nvapi-test")
    monkeypatch.setattr(settings, "LLM_TEMPERATURE", 0.3)

    cfg = _agent_llm()
    assert isinstance(cfg, dict)
    assert cfg["model"] == "meta/llama-3.3-70b-instruct"
    assert cfg["custom_openai"] is True
    assert cfg["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert cfg["api_key"] == "nvapi-test"


def test_agents_accept_custom_openai_llm(tools) -> None:
    from crewai.utilities.llm_utils import create_llm

    cfg = {
        "model": "meta/llama-3.3-70b-instruct",
        "custom_openai": True,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "nvapi-test",
    }
    llm = create_llm(cfg)
    agents = create_agents(tools, llm=llm)
    assert all(agent.llm is llm for agent in agents.values())
