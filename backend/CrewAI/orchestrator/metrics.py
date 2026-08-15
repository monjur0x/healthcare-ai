"""
Agent-level collaboration and reliability metrics (paper Section 12).

Pure, deterministic helpers that score the multi-agent pipeline from its
recorded outputs: how many crew tasks completed, how consistent agent
decisions are across runs, and how much agents built on each other's
output. They deliberately require no LLM so the metrics are reproducible
offline, mirroring the RAG and privacy metric modules.
"""

from __future__ import annotations

import re

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from preprocessing.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _output_of(result: Any) -> str:
    """Extract the output text from a task result of any supported shape."""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, Mapping):
        return str(result.get("output") or result.get("result") or "")
    output = getattr(result, "output", None)
    if output is None:
        output = getattr(result, "result", None)
    return str(output or "").strip()


def _token_set(text: str) -> set[str]:
    """Lowercased alphanumeric tokens (>= 3 chars) of a text."""
    return set(_TOKEN_RE.findall(text.lower()))


def task_completion_rate(crew_results: Sequence[Any]) -> float:
    """
    Fraction of crew tasks that produced a non-empty output.

    Parameters
    ----------
    crew_results : Sequence[Any]
        Task results in execution order. Each result may be a string, a
        dict with ``"output"``/``"result"`` keys, or an object exposing
        an ``output``/``result`` attribute (as CrewAI ``Task`` does).

    Returns
    -------
    float
        Completion rate in ``[0, 1]``; ``0.0`` for an empty sequence.
    """

    if not crew_results:
        return 0.0
    completed = sum(1 for result in crew_results if _output_of(result))
    return float(completed / len(crew_results))


def decision_consistency(predictions: Sequence[str]) -> float:
    """
    Fraction of agent decisions agreeing with the majority decision.

    Parameters
    ----------
    predictions : Sequence[str]
        Predicted classes (or decisions) from repeated runs.

    Returns
    -------
    float
        Consistency in ``[0, 1]``; ``1.0`` for empty or single-item runs
        (nothing to disagree with), ``0.0`` only for a tie-free split
        with no majority (defensive; ties always produce a majority).
    """

    if not predictions:
        return 1.0
    counts = Counter(str(prediction) for prediction in predictions)
    majority = counts.most_common(1)[0][1]
    return float(majority / len(predictions))


def agent_collaboration_score(crew_results: Sequence[Any]) -> float:
    """
    Fraction of crew outputs that share content with another agent's output.

    A lexical proxy for agent collaboration: a task is counted as
    collaborative when its output shares at least one meaningful token
    with the output of any *other* task, indicating that agents built on
    each other's work through the context mechanism.

    Parameters
    ----------
    crew_results : Sequence[Any]
        Task results in execution order (same accepted shapes as
        :func:`task_completion_rate`).

    Returns
    -------
    float
        Collaboration score in ``[0, 1]``; ``0.0`` for fewer than two
        tasks or when no cross-task token sharing occurs.
    """

    token_sets = [_token_set(_output_of(result)) for result in crew_results]
    if len(token_sets) < 2:
        return 0.0
    collaborative = 0
    for index, tokens in enumerate(token_sets):
        others: set[str] = set()
        for other_index, other in enumerate(token_sets):
            if other_index != index:
                others |= other
        if tokens & others:
            collaborative += 1
    return float(collaborative / len(token_sets))


@dataclass(frozen=True)
class AgentMetrics:
    """
    Aggregated agent-level metrics for one crew execution.

    Parameters
    ----------
    task_completion_rate : float
        Fraction of tasks that completed.
    decision_consistency : float
        Fraction of decisions matching the majority.
    agent_collaboration_score : float
        Fraction of outputs sharing content with other agents.
    """

    task_completion_rate: float
    decision_consistency: float
    agent_collaboration_score: float

    def to_dict(self) -> dict[str, float]:
        """
        Serialize the metrics to a JSON-friendly dictionary.

        Returns
        -------
        dict[str, float]
            Metrics keyed by name.
        """

        return {
            "task_completion_rate": self.task_completion_rate,
            "decision_consistency": self.decision_consistency,
            "agent_collaboration_score": self.agent_collaboration_score,
        }


def compute_agent_metrics(
    crew_results: Sequence[Any], predictions: Sequence[str]
) -> AgentMetrics:
    """
    Compute the full agent metrics block for one crew execution.

    Parameters
    ----------
    crew_results : Sequence[Any]
        Task results in execution order.
    predictions : Sequence[str]
        Predicted classes from repeated runs.

    Returns
    -------
    AgentMetrics
        Aggregated agent-level metrics.
    """

    metrics = AgentMetrics(
        task_completion_rate=task_completion_rate(crew_results),
        decision_consistency=decision_consistency(predictions),
        agent_collaboration_score=agent_collaboration_score(crew_results),
    )
    logger.info(
        "Agent metrics: completion=%.3f consistency=%.3f collaboration=%.3f",
        metrics.task_completion_rate,
        metrics.decision_consistency,
        metrics.agent_collaboration_score,
    )
    return metrics


__all__ = [
    "AgentMetrics",
    "agent_collaboration_score",
    "compute_agent_metrics",
    "decision_consistency",
    "task_completion_rate",
]
