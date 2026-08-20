import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_httpx import HTTPXMock

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel, BoundedSet, TypingIndicator, reconnect_loop


class _DummyChannel(BaseChannel):
    name = "dummy"
    _sent: list[OutboundMessage]

    def __init__(self, config, bus):
        super().__init__(config, bus)
        self._sent = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, msg: OutboundMessage) -> None:
        self._sent.append(msg)


def test_bounded_set_adds_and_evicts_oldest() -> None:
    cache = BoundedSet(3)
    cache.add("a")
    cache.add("b")
    cache.add("c")
    assert "a" in cache
    assert list(cache) == ["a", "b", "c"]
    cache.add("d")
    assert "a" not in cache
    assert "d" in cache
    assert len(cache) == 3


def test_bounded_set_setitem_compat() -> None:
    cache = BoundedSet(2)
    cache["x"] = None
    cache["y"] = None
    assert "x" in cache and "y" in cache


@pytest.mark.asyncio
async def test_typing_indicator_sends_periodically() -> None:
    calls: list[str] = []

    async def _send() -> None:
        calls.append("tick")

    indicator = TypingIndicator(interval=0.05)
    indicator.start("room1", _send)
    await asyncio.sleep(0.12)
    indicator.stop("room1")

    assert len(calls) >= 2


@pytest.mark.asyncio
async def test_typing_indicator_stop_cancels_task() -> None:
    calls: list[str] = []

    async def _send() -> None:
        calls.append("tick")

    indicator = TypingIndicator(interval=0.05)
    indicator.start("room1", _send)
    # Wait until at least one tick has had time to fire, then stop.
    await asyncio.sleep(0.12)
    before = len(calls)
    indicator.stop("room1")
    await asyncio.sleep(0.1)

    # After stop, no new ticks should arrive.
    assert len(calls) == before


@pytest.mark.asyncio
async def test_reconnect_loop_retries_with_backoff() -> None:
    attempts: list[int] = []
    run_flag = {"run": True}

    async def _connect() -> None:
        attempts.append(len(attempts))
        if len(attempts) >= 3:
            run_flag["run"] = False
        raise RuntimeError("boom")

    task = asyncio.create_task(
        reconnect_loop(
            _connect,
            lambda: run_flag["run"],
            base_delay=0.01,
            max_delay=0.05,
        )
    )
    await asyncio.wait_for(task, timeout=0.3)

    assert len(attempts) >= 3


@pytest.mark.asyncio
async def test_reconnect_loop_resets_delay_after_success() -> None:
    attempts: list[str] = []
    run_flag = {"run": True}

    async def _connect() -> None:
        attempts.append("ok")
        if len(attempts) == 1:
            raise RuntimeError("boom")
        run_flag["run"] = False

    task = asyncio.create_task(
        reconnect_loop(
            _connect,
            lambda: run_flag["run"],
            base_delay=0.01,
            max_delay=0.05,
        )
    )
    await asyncio.wait_for(task, timeout=0.15)

    # First attempt fails, second succeeds, then loop exits because should_run is False.
    assert attempts == ["ok", "ok"]


def test_is_allowed_requires_exact_match() -> None:
    channel = _DummyChannel(SimpleNamespace(allow_from=["allow@email.com"]), MessageBus())

    assert channel.is_allowed("allow@email.com") is True
    assert channel.is_allowed("attacker|allow@email.com") is False


def test_is_allowed_supports_dict_allow_from_alias() -> None:
    channel = _DummyChannel({"allowFrom": ["alice"]}, MessageBus())

    assert channel.is_allowed("alice") is True


def test_is_allowed_denies_empty_dict_allow_from() -> None:
    channel = _DummyChannel({"allow_from": []}, MessageBus())

    assert channel.is_allowed("alice") is False


def test_is_allowed_handles_none_allow_from() -> None:
    channel = _DummyChannel({"allow_from": None}, MessageBus())
    assert channel.is_allowed("alice") is False

    channel2 = _DummyChannel({"allowFrom": None}, MessageBus())
    assert channel2.is_allowed("alice") is False


def test_is_allowed_star_allows_all() -> None:
    channel = _DummyChannel({"allowFrom": ["*"]}, MessageBus())
    assert channel.is_allowed("anyone") is True


def test_is_allowed_pairing_fallback(monkeypatch) -> None:
    channel = _DummyChannel({"allowFrom": []}, MessageBus())
    monkeypatch.setattr(
        "nanobot.channels.base.is_approved", lambda _ch, sid: sid == "paired"
    )
    assert channel.is_allowed("paired") is True
    assert channel.is_allowed("unknown") is False


@pytest.mark.asyncio
async def test_handle_message_dm_sends_pairing_code(monkeypatch) -> None:
    channel = _DummyChannel({"allowFrom": []}, MessageBus())
    monkeypatch.setattr(
        "nanobot.channels.base.generate_code", lambda _ch, sid: "ABCD-EFGH"
    )

    await channel._handle_message(
        sender_id="stranger", chat_id="chat1", content="hello", is_dm=True
    )

    assert len(channel._sent) == 1
    msg = channel._sent[0]
    assert "ABCD-EFGH" in msg.content
    assert msg.metadata.get("_pairing_code") == "ABCD-EFGH"


@pytest.mark.asyncio
async def test_handle_message_group_ignores_unknown() -> None:
    channel = _DummyChannel({"allowFrom": []}, MessageBus())

    await channel._handle_message(
        sender_id="stranger", chat_id="chat1", content="hello", is_dm=False
    )

    assert channel._sent == []


@pytest.mark.asyncio
async def test_download_to_media_dir_success(tmp_path: Path, httpx_mock: HTTPXMock, monkeypatch) -> None:
    monkeypatch.setattr(
        "nanobot.config.paths.get_media_dir", lambda _name: tmp_path
    )
    channel = _DummyChannel({}, MessageBus())
    httpx_mock.add_response(url="https://example.com/a.png", content=b"pngdata")

    path, marker = await channel._download_to_media_dir(
        "https://example.com/a.png", "file_a.png", marker_type="image"
    )

    assert path is not None
    assert path.read_bytes() == b"pngdata"
    assert marker == "[image: file_a.png]"


@pytest.mark.asyncio
async def test_download_to_media_dir_failure_returns_marker(
    tmp_path: Path, httpx_mock: HTTPXMock, monkeypatch
) -> None:
    monkeypatch.setattr(
        "nanobot.config.paths.get_media_dir", lambda _name: tmp_path
    )
    channel = _DummyChannel({}, MessageBus())
    httpx_mock.add_response(url="https://example.com/b.png", status_code=404)

    path, marker = await channel._download_to_media_dir(
        "https://example.com/b.png", "file_b.png", marker_type="image"
    )

    assert path is None
    assert "download failed" in marker


@pytest.mark.asyncio
async def test_handle_message_uses_authorization_id_without_changing_sender() -> None:
    bus = MessageBus()
    channel = _DummyChannel({"allowFrom": ["group@g.us"]}, bus)

    await channel._handle_message(
        sender_id="member-lid",
        authorization_id="group@g.us",
        chat_id="group@g.us",
        content="hello",
    )

    msg = await bus.consume_inbound()
    assert msg.sender_id == "member-lid"
    assert msg.chat_id == "group@g.us"


@pytest.mark.asyncio
async def test_handle_message_rejects_when_authorization_id_is_not_allowed() -> None:
    bus = MessageBus()
    channel = _DummyChannel({"allowFrom": ["member-lid"]}, bus)

    await channel._handle_message(
        sender_id="member-lid",
        authorization_id="other-group@g.us",
        chat_id="other-group@g.us",
        content="hello",
    )

    assert bus.inbound_size == 0

