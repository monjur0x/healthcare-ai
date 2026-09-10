import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _hermetic_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests deterministic: never invoke the CrewAI LLM path."""
    from CrewAI.orchestrator.config import settings

    monkeypatch.setattr(settings, "LLM_API_KEY", "")
