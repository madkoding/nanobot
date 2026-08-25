"""Tests for RLAIF critic scoring utilities."""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.agent.rlaif.critic import RlaifCritic
from nanobot.providers.base import LLMResponse, ToolCallRequest


class FakeProvider:
    def __init__(self, *, tool_args: dict[str, Any] | None = None) -> None:
        self.tool_args = tool_args or {
            "score": 4,
            "reason": "looks fine",
            "issues": ["minor"],
        }
        self.last_messages: list[dict[str, Any]] | None = None

    async def chat_with_retry(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> LLMResponse:
        self.last_messages = messages
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="rate_solution",
                    arguments=self.tool_args,
                )
            ],
        )


class TestRlaifCriticScore:
    @pytest.mark.asyncio
    async def test_score_returns_value(self) -> None:
        provider = FakeProvider()
        critic = RlaifCritic(provider, model="fake")
        result = await critic.score(
            task="add two numbers",
            candidate={"patch": "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n", "summary": "fix"},
        )
        assert result.score == 4.0
        assert result.reason == "looks fine"
        assert provider.last_messages is not None

    @pytest.mark.asyncio
    async def test_score_handles_no_tool_call(self) -> None:
        provider = FakeProvider(tool_args={})
        # LLMResponse with empty tool_calls requires a special path; the critic
        # parser guards against it, but FakeProvider cannot easily return no
        # tool_calls because the constructor expects a dict.  We instead monkeypatch
        # the method to return an empty response.
        async def empty(*args, **kwargs):
            return LLMResponse(content="no tools", model="fake")

        provider.chat_with_retry = empty  # type: ignore[method-assign]
        critic = RlaifCritic(provider, model="fake")
        result = await critic.score(
            task="task",
            candidate={"patch": "", "summary": "x"},
        )
        assert result.score == 1.0
