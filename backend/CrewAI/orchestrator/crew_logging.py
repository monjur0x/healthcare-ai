"""Optional runtime logging wrappers for a CrewAI ``Crew``.

The module is deliberately self-contained: it never imports ``crewai``
at module load time, so it works whether or not the optional extra is
installed. :func:`wrap_crew_for_logging` patches ``crew.kickoff`` and
every task's ``execute`` with logging-aware versions that keep the
original return values and re-raise errors unchanged.

Note: ``ClinicalCrew`` produces its own structured ``CrewTrace`` via
``agent_tracing``; these wrappers are for debugging raw CrewAI objects
outside that path.
"""

from __future__ import annotations

import time

from collections.abc import Callable
from typing import Any

from preprocessing.logger import get_logger

logger = get_logger(__name__)


def _wrap_crew_kickoff(crew: Any) -> Any:
    """Wrap ``crew.kickoff`` with start / complete / error logging.

    The original bound method is captured before reassignment so the
    wrapper calls it directly — calling ``crew.kickoff`` from inside the
    wrapper would recurse infinitely after the attribute is patched.
    """
    original_kickoff: Callable = crew.kickoff

    def logged_kickoff(inputs: Any) -> Any:
        logger.info("[CREW START] Inputs: %s", list(inputs.keys()))
        start_time = time.perf_counter()
        try:
            result = original_kickoff(inputs)
            logger.info("[CREW COMPLETE] Time: %.3fs", time.perf_counter() - start_time)
            return result
        except Exception as error:
            logger.error("[CREW ERROR] %s: %s", type(error).__name__, error)
            raise

    crew.kickoff = logged_kickoff
    return crew


def _wrap_task_execution(agent: Any, task: Any) -> Any:
    """Wrap a task's ``execute`` method with per-task logging."""
    original_execute: Callable = task.execute
    agent_role = getattr(agent, "role", "Unknown")

    def logged_execute(*args: Any, **kwargs: Any) -> Any:
        logger.info(
            "[TASK START] Agent: %s | %s", agent_role, str(task.description)[:100]
        )
        start_time = time.perf_counter()
        try:
            result = original_execute(*args, **kwargs)
            logger.info(
                "[TASK END] Agent: %s | Time: %.3fs | Status: SUCCESS",
                agent_role,
                time.perf_counter() - start_time,
            )
            return result
        except Exception as error:
            logger.error(
                "[TASK ERROR] Agent: %s | Time: %.3fs | Error: %s",
                agent_role,
                time.perf_counter() - start_time,
                error,
            )
            raise

    task.execute = logged_execute
    return task


def _wrap_crew_tasks(crew: Any) -> Any:
    """Wrap every task in the crew for detailed logging."""
    for task in crew.tasks:
        _wrap_task_execution(task.agent, task)
    return crew


def wrap_crew_for_logging(crew: Any) -> Any:
    """Apply kickoff + per-task logging wrappers to a CrewAI crew."""
    return _wrap_crew_tasks(_wrap_crew_kickoff(crew))


__all__ = ["wrap_crew_for_logging"]
