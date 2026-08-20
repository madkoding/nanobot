import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.command.builtin import cmd_goal, cmd_pairing, cmd_restart
from nanobot.command.router import CommandContext


@pytest.fixture
def owner_ctx():
    msg = InboundMessage(
        channel="discord", sender_id="owner", chat_id="c1", content="/restart"
    )
    return CommandContext(msg=msg, session=None, key="k", raw="/restart", is_owner=True)


@pytest.fixture
def stranger_ctx():
    msg = InboundMessage(
        channel="discord", sender_id="stranger", chat_id="c1", content="/restart"
    )
    return CommandContext(msg=msg, session=None, key="k", raw="/restart", is_owner=False)


@pytest.mark.asyncio
async def test_restart_requires_owner(stranger_ctx, owner_ctx, monkeypatch) -> None:
    monkeypatch.setattr(
        "nanobot.command.builtin.set_restart_notice_to_env", lambda **kw: None
    )
    refusal = await cmd_restart(stranger_ctx)
    assert isinstance(refusal, OutboundMessage)
    assert "Only the operator" in refusal.content
    owner_result = await cmd_restart(owner_ctx)
    assert isinstance(owner_result, OutboundMessage)
    assert "Restarting" in owner_result.content


@pytest.mark.asyncio
async def test_pairing_requires_owner(stranger_ctx, monkeypatch) -> None:
    monkeypatch.setattr(
        "nanobot.pairing.store.handle_pairing_command",
        lambda ch, args: "approved",
    )
    refusal = await cmd_pairing(stranger_ctx)
    assert isinstance(refusal, OutboundMessage)
    assert "Only the operator" in refusal.content


@pytest.mark.asyncio
async def test_goal_requires_owner(stranger_ctx) -> None:
    stranger_ctx.raw = "/goal test"
    refusal = await cmd_goal(stranger_ctx)
    assert isinstance(refusal, OutboundMessage)
    assert "Only the operator" in refusal.content
