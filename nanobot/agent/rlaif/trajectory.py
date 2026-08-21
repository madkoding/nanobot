"""Trajectory model for agent self-improvement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnStep:
    """One step in an agent trajectory: an LLM message and optional tool result."""

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    # Provider/model metadata; useful for attributing reward to a policy.
    model: str | None = None
    provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "model": self.model,
            "provider": self.provider,
        }


@dataclass
class Trajectory:
    """A complete agent turn/session capture used to compute AI feedback."""

    task: str
    steps: list[TurnStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TurnStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the trajectory as a chat-messages list for critic prompts."""
        out: list[dict[str, Any]] = []
        for step in self.steps:
            if step.tool_results:
                tool_content = "\n\n".join(
                    _format_tool_result(r) for r in step.tool_results
                )
                out.append({"role": "user", "content": tool_content})
                continue
            msg: dict[str, Any] = {"role": step.role}
            if step.content is not None:
                msg["content"] = step.content
            if step.tool_calls:
                msg["tool_calls"] = step.tool_calls
            out.append(msg)
        return out


def _format_tool_result(result: dict[str, Any]) -> str:
    name = result.get("name", "unknown")
    content = result.get("content", "")
    is_error = result.get("is_error", False)
    prefix = "Error" if is_error else "Result"
    return f"[{prefix} from {name}]:\n{content}"
