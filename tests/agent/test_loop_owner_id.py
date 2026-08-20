from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.runtime_context import RUNTIME_CONTEXT_END, RUNTIME_CONTEXT_TAG


@pytest.fixture
def _loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        owner_id="15551234567",
    )


def _fake_tool(name: str) -> Tool:
    tool = MagicMock(spec=Tool)
    tool.name = name
    tool.to_schema.return_value = {"name": name}
    return tool


@pytest.mark.asyncio
async def test_non_owner_sender_gets_untrusted_block(_loop: AgentLoop) -> None:
    request = RequestContext(
        channel="whatsapp",
        chat_id="group-1",
        sender_id="stranger-1",
        metadata={"message_id": "m1"},
    )
    blocks = await _loop._resolve_runtime_context_for_request(request, _loop.tools)

    trust_blocks = [b for b in blocks if b.source == "sender_trust"]
    assert len(trust_blocks) == 1
    content = trust_blocks[0].content
    assert RUNTIME_CONTEXT_TAG in content
    assert RUNTIME_CONTEXT_END in content
    assert "stranger-1" in content
    assert "15551234567" in content
    assert "untrusted data" in content


@pytest.mark.asyncio
async def test_owner_sender_gets_no_untrusted_block(_loop: AgentLoop) -> None:
    request = RequestContext(
        channel="whatsapp",
        chat_id="group-1",
        sender_id="15551234567",
        metadata={"message_id": "m2"},
    )
    blocks = await _loop._resolve_runtime_context_for_request(request, _loop.tools)

    trust_blocks = [b for b in blocks if b.source == "sender_trust"]
    assert len(trust_blocks) == 0


@pytest.mark.asyncio
async def test_owner_whatsapp_jid_gets_no_untrusted_block(_loop: AgentLoop) -> None:
    request = RequestContext(
        channel="whatsapp",
        chat_id="group-1",
        sender_id="15551234567@s.whatsapp.net",
        metadata={"message_id": "m2"},
    )
    blocks = await _loop._resolve_runtime_context_for_request(request, _loop.tools)

    trust_blocks = [b for b in blocks if b.source == "sender_trust"]
    assert len(trust_blocks) == 0


@pytest.mark.asyncio
async def test_owner_id_none_is_inactive(tmp_path: Path) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )

    request = RequestContext(
        channel="whatsapp",
        chat_id="group-1",
        sender_id="stranger-1",
        metadata={"message_id": "m3"},
    )
    blocks = await loop._resolve_runtime_context_for_request(request, loop.tools)

    trust_blocks = [b for b in blocks if b.source == "sender_trust"]
    assert len(trust_blocks) == 0


def _capture_run_spec(loop: AgentLoop) -> dict[str, object]:
    captured: dict[str, object] = {}

    async def patched_run(spec):
        captured["tools"] = spec.tools
        return AgentRunResult(
            final_content="done",
            tools_used=[],
            messages=[],
            stop_reason="completed",
            had_injections=False,
            usage={},
        )

    loop.runner.run = AsyncMock(side_effect=patched_run)
    return captured


@pytest.mark.asyncio
async def test_owner_sees_all_tools(_loop: AgentLoop) -> None:
    registry = ToolRegistry()
    for name in ["read_file", "write_file", "exec", "create_goal", "todos", "mcp_server_x"]:
        registry.register(_fake_tool(name))
    _loop.tools = registry

    captured = _capture_run_spec(_loop)
    await _loop._run_agent_loop(
        [],
        runtime=_loop.llm_runtime(),
        sender_id="15551234567",
    )

    tools = captured["tools"]
    assert set(tools.tool_names) == {
        "read_file", "write_file", "exec", "create_goal", "todos", "mcp_server_x",
    }


@pytest.mark.asyncio
async def test_non_owner_sees_read_only_tools(_loop: AgentLoop) -> None:
    registry = ToolRegistry()
    for name in ["read_file", "write_file", "exec", "create_goal", "todos", "mcp_server_x"]:
        registry.register(_fake_tool(name))
    _loop.tools = registry

    captured = _capture_run_spec(_loop)
    await _loop._run_agent_loop(
        [],
        runtime=_loop.llm_runtime(),
        sender_id="stranger-1",
    )

    tools = captured["tools"]
    assert set(tools.tool_names) == {"read_file"}


@pytest.mark.asyncio
async def test_no_owner_configured_skips_tool_filter(_loop: AgentLoop) -> None:
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=_loop.workspace,
        model="test-model",
    )
    registry = ToolRegistry()
    for name in ["read_file", "write_file", "todos"]:
        registry.register(_fake_tool(name))
    loop.tools = registry

    captured = _capture_run_spec(loop)
    await loop._run_agent_loop(
        [],
        runtime=loop.llm_runtime(),
        sender_id="stranger-1",
    )

    assert set(captured["tools"].tool_names) == {"read_file", "write_file", "todos"}
