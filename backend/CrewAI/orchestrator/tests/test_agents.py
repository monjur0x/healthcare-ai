"""
Tests for agent LLM configuration (no LLM kickoff).

Only the LLM-configuration helpers are tested here. The multi-agent
fabrication API (``create_agents`` / ``create_tasks``) was removed in
the single-agent narrative-enrichment refactor (see ``crew.py``), so the
older construction tests no longer apply.
"""

from __future__ import annotations

from CrewAI.orchestrator.agents import _agent_llm


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
