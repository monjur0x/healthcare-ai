from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional
from functools import wraps

from preprocessing.logger import get_logger

logger = get_logger(__name__)

# CrewAI is optional - check availability
_CREWAI_AVAILABLE = False
try:
    from crewai import Task, Agent
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False


logger = get_logger(__name__)


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
            result = crew.kickoff(inputs)
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


def wrap_crew_for_logging(crew):
    """Apply comprehensive logging to crew execution."""
    crew = _wrap_crew_kickoff(crew)
    crew = _wrap_crew_tasks(crew)
    return crew