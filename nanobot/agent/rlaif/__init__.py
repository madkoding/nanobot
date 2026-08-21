"""RLAIF: Reinforcement Learning from AI Feedback for nanobot self-improvement.

This package provides:
- Trajectory capture for agent turns.
- LLM-as-critic scoring and pairwise preference generation.
- Dataset persistence for future offline RL / DPO / GRPO training.
- Patch-evaluation harness that runs tests and lint on candidate code edits.
- Observer hook that schedules background self-improvement tasks.
"""

from __future__ import annotations

from nanobot.agent.rlaif.critic import RlaifCritic, RlaifCriticResult
from nanobot.agent.rlaif.dataset import RlaifDataset, RlaifPreference
from nanobot.agent.rlaif.harness import PatchHarness, PatchHarnessResult
from nanobot.agent.rlaif.observer import RlaifObserverHook, create_rlaif_observer_hook
from nanobot.agent.rlaif.trajectory import Trajectory, TurnStep

__all__ = [
    "PatchHarness",
    "PatchHarnessResult",
    "RlaifCritic",
    "RlaifCriticResult",
    "RlaifDataset",
    "RlaifObserverHook",
    "RlaifPreference",
    "Trajectory",
    "TurnStep",
    "create_rlaif_observer_hook",
]
