"""Tests for AgentRunner content/reasoning repetition detection.

Covers the two nudges added to the no-tools branch:
1. Goal-conflict nudge: model replied without tools while a sustained goal is active.
2. Repeated content nudge/hard-stop: model repeats the same final content/reasoning.

Content repetition can only be detected when the loop actually re-iterates.
In normal operation the runner breaks on the first no-tools final response,
so repetition requires either a sustained goal (goal_continue injection) or
an external injection_callback that keeps the loop alive. Tests below use both.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.runner import AgentRunner
from nanobot.config.schema import AgentDefaults
from nanobot.providers.base import LLMResponse, ToolCallRequest

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


def _injection_callback(messages: list[str]) -> Callable[..., Awaitable[list[dict[str, Any]]]]:
    """Return an async callback that injects one user message per call, then stops.

    This keeps the runner loop alive for a controlled number of no-tools final
    responses without adding meaningful new signal to the model.
    """
    state: dict[str, Any] = {"idx": 0}

    async def callback(*, limit: int | None = None) -> list[dict[str, Any]]:
        idx = state["idx"]
        if idx < len(messages):
            state["idx"] = idx + 1
            return [{"role": "user", "content": messages[idx]}]
        return []

    return callback


def _goal_conflict_nudges(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        m for m in messages
        if m.get("role") == "user" and "sustained goal is still active" in m.get("content", "")
    ]


def _content_nudges(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        m for m in messages
        if m.get("role") == "user" and "produced the same response" in m.get("content", "")
    ]


@pytest.mark.asyncio
async def test_goal_conflict_nudge_injected_once():
    """When the model replies without tools while a sustained goal is active,
    a one-time actionable nudge is injected before the generic goal_continue."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="I have finished the task.", tool_calls=[]),
        LLMResponse(content="I have finished the task.", tool_calls=[]),
        LLMResponse(content="I have finished the task.", tool_calls=[]),
        LLMResponse(content="I have finished the task.", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = [{"type": "function", "function": {"name": "update_goal"}}]
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=4,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        goal_active_predicate=lambda: True,
        finalize_on_max_iterations=False,
    ))

    assert provider.chat_with_retry.await_count == 4
    assert len(_goal_conflict_nudges(result.messages)) == 1
    # With a sustained goal active, the runner keeps looping until max_iterations.
    assert result.stop_reason == "max_iterations"


@pytest.mark.asyncio
async def test_repeated_content_nudge_with_goal():
    """Two identical no-tools final responses trigger the content nudge when a
    sustained goal keeps the loop alive."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I understand; I need tool X.", tool_calls=[]),
        LLMResponse(content="I understand; I need tool X.", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = [{"type": "function", "function": {"name": "update_goal"}}]
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=4,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        goal_active_predicate=lambda: True,
        finalize_on_max_iterations=False,
    ))

    assert provider.chat_with_retry.await_count == 4
    # First nudge is goal-conflict, second is repeated-content.
    assert len(_goal_conflict_nudges(result.messages)) == 1
    assert len(_content_nudges(result.messages)) == 1
    assert result.stop_reason == "max_iterations"


@pytest.mark.asyncio
async def test_repeated_content_nudge_with_injection_callback():
    """Without a sustained goal, an external injection_callback must keep the
    loop alive for content repetition to be detected."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I understand now.", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=8,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        injection_callback=_injection_callback(["continue"]),
    ))

    # 1st: injection_callback keeps loop alive, 2nd: nudge, 3rd: final answer.
    assert provider.chat_with_retry.await_count == 3
    assert len(_content_nudges(result.messages)) == 1
    assert result.final_content == "I understand now."


@pytest.mark.asyncio
async def test_repeated_content_hard_stop_without_goal():
    """Three identical no-tools final responses hard-stop when the loop is kept
    alive by an external injection and no sustained goal is active."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=8,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        injection_callback=_injection_callback(["continue"]),
    ))

    # Third identical response triggers hard-stop; no fourth LLM call.
    assert provider.chat_with_retry.await_count == 3
    assert result.stop_reason == "repeated_content_loop"
    assert "repeating the same response" in (result.final_content or "")


@pytest.mark.asyncio
async def test_content_repeat_resets_on_change():
    """Changing content resets the repetition counter."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="Maybe I can try X.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="Ok I will stop.", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        injection_callback=_injection_callback(["continue"] * 3),
    ))

    # Pattern is 2,2,1 after reset: never reaches hard-stop limit of 3.
    assert result.stop_reason != "repeated_content_loop"
    # Nudges at the 2nd identical response (iter 2) and again after reset (iter 5).
    assert len(_content_nudges(result.messages)) == 2


@pytest.mark.asyncio
async def test_tool_execution_resets_content_repeat_with_goal():
    """A tool execution in between resets content-repetition tracking, even when
    a sustained goal keeps the loop alive afterwards."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(
            content="Let me check the file.",
            tool_calls=[ToolCallRequest(id="c1", name="read_file", arguments={"path": "x.txt"})],
        ),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
        LLMResponse(content="I cannot proceed.", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="content of x.txt")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        goal_active_predicate=lambda: True,
        finalize_on_max_iterations=False,
    ))

    # After the tool the content count resets: the two repetitions after it
    # trigger exactly one nudge, never a hard-stop (goal is active).
    assert result.stop_reason != "repeated_content_loop"
    assert len(_content_nudges(result.messages)) == 1


@pytest.mark.asyncio
async def test_goal_conflict_nudge_only_with_tools_available():
    """Goal-conflict nudge is only injected when the model actually has tools to use."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="I have finished.", tool_calls=[]),
        LLMResponse(content="ok", tool_calls=[]),
        LLMResponse(content="ok", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        goal_active_predicate=lambda: True,
        finalize_on_max_iterations=False,
    ))

    # No tools available -> no goal-conflict nudge. The generic goal_continue
    # still fires but without tools the model just answers "ok".
    assert len(_goal_conflict_nudges(result.messages)) == 0
    assert result.stop_reason == "max_iterations"


@pytest.mark.asyncio
async def test_cross_turn_repeat_seeded_from_history():
    """A final response repeated from the previous turn is caught on the first
    iteration instead of giving the model 3 free repeats per turn.

    The detector is seeded with the last assistant final response in history, so
    the first identical response this turn counts as repeat #2 (nudge) and the
    second as #3 (hard-stop) — two LLM calls instead of three.
    """
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="https://github.com/SlimeVR/SlimeVR-Tracker-NRF", tool_calls=[]),
        LLMResponse(content="https://github.com/SlimeVR/SlimeVR-Tracker-NRF", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
            # Previous turn's final response — the seed for cross-turn detection.
            {"role": "assistant", "content": "https://github.com/SlimeVR/SlimeVR-Tracker-NRF"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=8,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        injection_callback=_injection_callback(["continue"]),
    ))

    # Seeded count=1 -> first repeat is #2 (nudge), second is #3 (hard-stop).
    assert provider.chat_with_retry.await_count == 2
    assert result.stop_reason == "repeated_content_loop"
    assert len(_content_nudges(result.messages)) == 1


@pytest.mark.asyncio
async def test_cross_turn_seed_ignores_tool_calls_and_blank():
    """Seeding only uses the last assistant *final* response: tool-call turns and
    blank assistant messages are skipped, so a fresh approach is not falsely
    flagged as a repeat."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="I understand now.", tool_calls=[]),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
            # Tool-call turn (no final content) must not seed the detector.
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "x"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=8,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    # No seed -> the single "I understand now." is a fresh response, no nudge.
    assert result.stop_reason != "repeated_content_loop"
    assert len(_content_nudges(result.messages)) == 0
    assert result.final_content == "I understand now."


@pytest.mark.asyncio
async def test_repeated_content_reasoning_counts():
    """Repeated reasoning with similar visible content is detected too."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="Blocked.",
            tool_calls=[],
            reasoning_content="I see no way forward.",
        ),
        LLMResponse(
            content="Blocked.",
            tool_calls=[],
            reasoning_content="I see no way forward.",
        ),
        LLMResponse(
            content="Blocked.",
            tool_calls=[],
            reasoning_content="I see no way forward.",
        ),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=8,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        injection_callback=_injection_callback(["continue"]),
    ))

    assert result.stop_reason == "repeated_content_loop"
    assert provider.chat_with_retry.await_count == 3


def _tool_repeat_nudges(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        m for m in messages
        if m.get("role") == "user" and "repeating the same tool call" in m.get("content", "")
    ]


def _alternating_nudges(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        m for m in messages
        if m.get("role") == "user" and "alternating between the same two tool" in m.get("content", "")
    ]


@pytest.mark.asyncio
async def test_repeated_tool_call_same_result_triggers_hard_stop():
    """The same tool call with the SAME result is a stuck loop."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="reading",
            tool_calls=[ToolCallRequest(id=f"c{i}", name="read_file", arguments={"path": "x.txt"})],
        )
        for i in range(5)
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="same content")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.stop_reason == "repeated_tool_loop"
    assert provider.chat_with_retry.await_count == 5
    assert len(_tool_repeat_nudges(result.messages)) == 1


@pytest.mark.asyncio
async def test_repeated_tool_call_different_result_is_progress():
    """The same tool call with DIFFERENT results is legitimate progress;
    repetition detection must reset and not hard-stop."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="reading",
            tool_calls=[ToolCallRequest(id=f"c{i}", name="read_file", arguments={"path": "x.txt"})],
        )
        for i in range(5)
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=[
        "content 1",
        "content 2",
        "content 3",
        "content 4",
        "content 5",
    ])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        finalize_on_max_iterations=False,
    ))

    # The model kept calling the same file but the result changed every time,
    # so there is real progress. No nudge, no hard-stop.
    assert result.stop_reason != "repeated_tool_loop"
    assert len(_tool_repeat_nudges(result.messages)) == 0
    assert provider.chat_with_retry.await_count == 5


@pytest.mark.asyncio
async def test_repeated_tool_call_mixed_result_resets_counter():
    """Different results reset the counter; identical results after that count
    from the reset value."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="reading",
            tool_calls=[ToolCallRequest(id=f"c{i}", name="read_file", arguments={"path": "x.txt"})],
        )
        for i in range(7)
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(side_effect=[
        "content 1",  # count=1
        "content 2",  # same sig, different result -> reset to 1
        "content 3",  # count=1
        "content 3",  # count=2
        "content 3",  # count=3 -> nudge
        "content 3",  # count=4
        "content 3",  # count=5 -> hard-stop
    ])

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.stop_reason == "repeated_tool_loop"
    assert provider.chat_with_retry.await_count == 7
    assert len(_tool_repeat_nudges(result.messages)) == 1


@pytest.mark.asyncio
async def test_alternating_tool_pattern_nudge_then_hard_stop():
    """A->B->A->B->A->B nudges; A->B->A->B->A->B->A->B hard-stops."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="step A",
            tool_calls=[ToolCallRequest(id=f"c{i}", name="read_file", arguments={"path": "a.txt"})],
        )
        if i % 2 == 0 else
        LLMResponse(
            content="step B",
            tool_calls=[ToolCallRequest(id=f"c{i}", name="read_file", arguments={"path": "b.txt"})],
        )
        for i in range(8)
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="same")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=12,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.stop_reason == "alternating_tool_loop"
    assert provider.chat_with_retry.await_count == 8
    assert len(_alternating_nudges(result.messages)) == 1


@pytest.mark.asyncio
async def test_non_alternating_tool_pattern_does_not_trigger():
    """A->B->A->C is not a stable period-2 pattern; no nudge/hard-stop."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="step A",
            tool_calls=[ToolCallRequest(id="c1", name="read_file", arguments={"path": "a.txt"})],
        ),
        LLMResponse(
            content="step B",
            tool_calls=[ToolCallRequest(id="c2", name="read_file", arguments={"path": "b.txt"})],
        ),
        LLMResponse(
            content="step A",
            tool_calls=[ToolCallRequest(id="c3", name="read_file", arguments={"path": "a.txt"})],
        ),
        LLMResponse(
            content="step C",
            tool_calls=[ToolCallRequest(id="c4", name="read_file", arguments={"path": "c.txt"})],
        ),
        LLMResponse(
            content="step A",
            tool_calls=[ToolCallRequest(id="c5", name="read_file", arguments={"path": "a.txt"})],
        ),
        LLMResponse(
            content="step B",
            tool_calls=[ToolCallRequest(id="c6", name="read_file", arguments={"path": "b.txt"})],
        ),
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="same")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=6,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.stop_reason != "alternating_tool_loop"
    assert len(_alternating_nudges(result.messages)) == 0


@pytest.mark.asyncio
async def test_configurable_repeat_thresholds():
    """Custom thresholds from AgentRunSpec are respected."""
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="reading",
            tool_calls=[ToolCallRequest(id=f"c{i}", name="read_file", arguments={"path": "x.txt"})],
        )
        for i in range(4)
    ])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="same")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "do task"},
        ],
        tools=tools,
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        tool_repeat_nudge_after=2,
        tool_repeat_hard_stop_after=4,
    ))

    assert result.stop_reason == "repeated_tool_loop"
    assert provider.chat_with_retry.await_count == 4
    assert len(_tool_repeat_nudges(result.messages)) == 1
