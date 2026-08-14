"""
Tests for agent / task / crew construction (no LLM kickoff).
"""

from __future__ import annotations

import pytest

from CrewAI.orchestrator.agents import create_agents
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
