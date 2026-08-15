"""
Tests for the agent-level metrics (paper Section 12).

The metrics are deterministic and LLM-free, so these tests run without a
CrewAI install or an API key. They exercise the supported input shapes:
raw strings, dicts with ``output``/``result`` keys, and objects exposing
an ``output`` attribute (CrewAI ``Task``-style).
"""

from __future__ import annotations

from types import SimpleNamespace

from CrewAI.orchestrator.metrics import (
    agent_collaboration_score,
    compute_agent_metrics,
    decision_consistency,
    task_completion_rate,
)
from CrewAI.orchestrator.services import assemble_clinical_report
from CrewAI.orchestrator.schemas import PatientInfo


def test_task_completion_rate_all_complete() -> None:
    results = ["done", "also done", "third task output"]
    assert task_completion_rate(results) == 1.0


def test_task_completion_rate_with_failures() -> None:
    results = ["done", "", "", "finished"]
    assert task_completion_rate(results) == 0.5


def test_task_completion_rate_empty_returns_zero() -> None:
    assert task_completion_rate([]) == 0.0


def test_task_completion_rate_accepts_dicts_and_objects() -> None:
    results = [
        {"output": "filled"},
        {"result": "also filled"},
        {"output": ""},
        SimpleNamespace(output="task object output"),
        SimpleNamespace(result="result attribute"),
    ]
    assert task_completion_rate(results) == 0.8


def test_decision_consistency_majority_agreement() -> None:
    assert decision_consistency(["diabetes", "diabetes", "healthy"]) == 2 / 3
    assert decision_consistency(["diabetes", "diabetes", "diabetes"]) == 1.0


def test_decision_consistency_tie_is_stable() -> None:
    score = decision_consistency(["a", "a", "b", "b"])
    assert score == 0.5


def test_decision_consistency_single_and_empty() -> None:
    assert decision_consistency(["diabetes"]) == 1.0
    assert decision_consistency([]) == 1.0


def test_agent_collaboration_score_shared_tokens() -> None:
    results = [
        "patient has elevated glucose levels",
        "glucose elevation suggests diabetes risk",
        "diabetes risk requires monitoring plan",
    ]
    assert agent_collaboration_score(results) == 1.0


def test_agent_collaboration_score_no_shared_tokens() -> None:
    results = [
        "zebra migrations observed",
        "quantum computing lecture notes",
        "ancient pottery restoration techniques",
    ]
    assert agent_collaboration_score(results) == 0.0


def test_agent_collaboration_score_single_result_is_zero() -> None:
    assert agent_collaboration_score(["only one task"]) == 0.0
    assert agent_collaboration_score([]) == 0.0


def test_agent_collaboration_score_partial() -> None:
    results = [
        "diabetes management metformin",
        "unrelated quantum computing notes",
        "metformin reduces diabetes complications",
    ]
    assert agent_collaboration_score(results) == 2 / 3


def test_compute_agent_metrics_aggregates() -> None:
    results = ["done output", "shared output", ""]
    metrics = compute_agent_metrics(results, ["a", "a", "b"])
    assert metrics.task_completion_rate == 2 / 3
    assert metrics.decision_consistency == 2 / 3
    assert metrics.agent_collaboration_score == 2 / 3
    payload = metrics.to_dict()
    assert set(payload) == {
        "task_completion_rate",
        "decision_consistency",
        "agent_collaboration_score",
    }


def test_report_carries_optional_agent_metrics() -> None:
    metrics = compute_agent_metrics(["done"], ["diabetes"]).to_dict()
    report = assemble_clinical_report(
        patient=PatientInfo(id="p1"), agent_metrics=metrics
    )
    assert report.agent_metrics == metrics
    assert report.to_dict()["agent_metrics"] == metrics


def test_report_without_agent_metrics_keeps_none() -> None:
    report = assemble_clinical_report(patient=PatientInfo(id="p1"))
    assert report.agent_metrics is None