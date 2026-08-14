from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import RequestContext
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
        owner_id="operator-1",
    )


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
    assert "operator-1" in content
    assert "untrusted data" in content


@pytest.mark.asyncio
async def test_owner_sender_gets_no_untrusted_block(_loop: AgentLoop) -> None:
    request = RequestContext(
        channel="whatsapp",
        chat_id="group-1",
        sender_id="operator-1",
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
