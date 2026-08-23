"""Per-agent execution tracing for the clinical crew."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from preprocessing.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentTrace:
    """Execution trace for one agent step."""

    agent_name: str
    role: str
    task_description: str
    input_summary: str = ""
    output_summary: str = ""
    output_data: Any = None
    execution_time_s: float = 0.0
    status: str = "PENDING"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "role": self.role,
            "task": self.task_description,
            "input": self.input_summary,
            "output": self.output_summary,
            "execution_time_s": round(self.execution_time_s, 4),
            "status": self.status,
        }


@dataclass
class CrewTrace:
    """Full execution trace for all agents in one crew run."""

    patient_id: str
    steps: list[AgentTrace] = field(default_factory=list)
    total_time_s: float = 0.0
    all_succeeded: bool = True

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "SUCCESS")

    @property
    def total_agents(self) -> int:
        return len(self.steps)

    def summary(self) -> str:
        header = (
            f"Crew: {self.completed_count}/{self.total_agents} "
            f"agents succeeded in {self.total_time_s:.3f}s"
        )
        lines = [header]
        for step in self.steps:
            icon = {"SUCCESS": "✓", "FAILED": "✗"}.get(step.status, "○")
            lines.append(
                f"  {icon} {step.agent_name} [{step.role}] {step.execution_time_s:.4f}s"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "total_agents": self.total_agents,
            "completed": self.completed_count,
            "total_time_s": round(self.total_time_s, 4),
            "all_succeeded": self.all_succeeded,
            "agents": [s.to_dict() for s in self.steps],
        }
