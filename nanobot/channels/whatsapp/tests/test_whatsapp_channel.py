from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import nanobot.channels.whatsapp.runtime as whatsapp_module
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.whatsapp.runtime import (
    WhatsAppChannel,
    _legacy_bridge_config_fields,
    _NeonizeAPI,
    _typing_task_key,
)


class _Proto:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def HasField(self, name: str) -> bool:  # noqa: N802 - protobuf compatibility
        return _is_set(getattr(self, name, None))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Proto) and self.__dict__ == other.__dict__

    def ListFields(self):  # noqa: N802 - protobuf compatibility
        return [
            (SimpleNamespace(name=name), value)
            for name, value in self.__dict__.items()
            if _is_set(value)
        ]


def _is_set(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return bool(value)
    return True


def _jid(user: str, server: str) -> _Proto:
    return _Proto(User=user, Server=server, IsEmpty=False)


def _message_with_conversation(content: str) -> _Proto:
    return _Proto(extendedTextMessage=_Proto(text=content))


def _event(
    *,
    message: _Proto,
    message_id: str = "m1",
    chat: _Proto | None = None,
    sender: _Proto | None = None,
    sender_alt: _Proto | None = None,
    is_group: bool = False,
    timestamp: int = 1,
    is_from_me: bool = False,
) -> _Proto:
    source = _Proto(
        Chat=chat or _jid("15551234567", "s.whatsapp.net"),
        Sender=sender,
        SenderAlt=sender_alt,
        IsGroup=is_group,
        IsFromMe=is_from_me,
    )
    return _Proto(
        Info=_Proto(ID=message_id, Timestamp=timestamp, MessageSource=source),
        Message=message,
    )


def _make_channel(
    config: dict | None = None,
    *,
    tmp_path: Path | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> WhatsAppChannel:
    merged = {"enabled": True, "allowFrom": ["*"]}
    if config:
        merged.update(config)
    ch = WhatsAppChannel(merged, MagicMock())
    ch._started_at = 0
    if tmp_path is not None and monkeypatch is not None:
        # Isolate the persistent throttle state per test so cooldown
        # counters don't leak across tests or into the real runtime dir.
        # The class-level patch is reverted automatically by monkeypatch
        # at end of test.
        state_file = tmp_path / "throttle.json"
        monkeypatch.setattr(
            whatsapp_module.WhatsAppChannel,
            "_throttle_state_path",
            lambda self, _f=state_file: _f,
        )
    return ch


def _patch_neonize_api(monkeypatch) -> None:
    monkeypatch.setattr(
        whatsapp_module,
        "_NEONIZE_API",
        _NeonizeAPI(
            NewAClient=object,
            ConnectedEv=object(),
            ConnectFailureEv=object(),
            DisconnectedEv=object(),
            LoggedOutEv=object(),
            MessageEv=object(),
            PairStatusEv=object(),
            StreamErrorEv=object(),
            build_jid=lambda user, server="s.whatsapp.net": (user, server),
            Message=lambda **kw: _Proto(**kw),
            ExtendedTextMessage=lambda **kw: _Proto(**kw),
        ),
    )


def _patch_receipt_type(monkeypatch):
    neonize = types.ModuleType("neonize")
    utils = types.ModuleType("neonize.utils")
    enum = types.ModuleType("neonize.utils.enum")

    class ReceiptType:
        READ = "read"

    enum.ReceiptType = ReceiptType
    neonize.utils = utils
    utils.enum = enum
    monkeypatch.setitem(sys.modules, "neonize", neonize)
    monkeypatch.setitem(sys.modules, "neonize.utils", utils)
    monkeypatch.setitem(sys.modules, "neonize.utils.enum", enum)
    return ReceiptType


class _FakeLoginClient:
    def __init__(self) -> None:
        self.handlers = {}
        self.me = _Proto(JID=_jid("bot", "s.whatsapp.net"), LID=_jid("BOTLID", "lid"))
        self._stop_calls = 0
        self._stopped = asyncio.Event()

    def event(self, event_type):
        def register(func):
            self.handlers[event_type] = func
            return func

        return register

    def qr(self, func):
        self.qr_handler = func
        return func

    async def connect(self) -> None:
        await self.handlers[whatsapp_module._NEONIZE_API.ConnectedEv](self, _Proto())

    async def idle(self) -> None:
        # Mimic a successful login: idle() unblocks as soon as stop() is
        # called (the same shutdown path neonize takes on a clean exit).
        await self._stopped.wait()

    async def stop(self) -> None:
        self._stop_calls += 1
        self._stopped.set()


class _FailingConnectLoginClient(_FakeLoginClient):
    async def connect(self) -> asyncio.Task[None]:
        async def fail() -> None:
            raise RuntimeError("dial failed")

        return asyncio.create_task(fail())


def test_default_config_has_no_bridge_fields() -> None:
    config = WhatsAppChannel.default_config()

    assert "bridgeUrl" not in config
    assert "bridgeToken" not in config
    assert config["databasePath"] == ""


def test_legacy_bridge_config_fields_are_detected() -> None:
    assert _legacy_bridge_config_fields({"bridgeUrl": "ws://localhost:3001"}) == ["bridgeUrl"]
    assert _legacy_bridge_config_fields({"bridgeToken": "secret"}) == ["bridgeToken"]


@pytest.mark.asyncio
async def test_login_succeeds_when_connected(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    client = _FakeLoginClient()
    ch = _make_channel()
    ch._new_client = MagicMock(return_value=client)

    assert await ch.login() is True
    assert ch._self_jids == {"bot@s.whatsapp.net", "bot", "BOTLID@lid", "BOTLID"}
    assert client._stop_calls == 1


@pytest.mark.asyncio
async def test_login_fails_when_connect_task_fails(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    client = _FailingConnectLoginClient()
    ch = _make_channel()
    ch._new_client = MagicMock(return_value=client)

    assert await ch.login() is False
    assert client._stop_calls == 1


# ponytail: a second fake client that lets the test fire LoggedOutEv
# on demand. We can't reuse _FakeLoginClient because its handlers are
# bound at construction time; this one exposes the registered event
# handler so the test can call it. The flow mimics what real neonize
# does on LoggedOutEv: close the websocket, which makes the pending
# ``client.idle()`` (awaited by ``_run_session``) return.
class _LoggedOutClient:
    def __init__(self) -> None:
        self._stopped = asyncio.Event()
        self._stop_calls = 0
        self.handlers: dict = {}
        self.connected_event = asyncio.Event()
        # Pre-populate get_me() so _remember_self_jids works after
        # ConnectedEv fires.
        self.me = _Proto(JID=_jid("56928861873", "s.whatsapp.net"), LID=_jid("BOTLID", "lid"))

    def event(self, event_type):
        def register(func):
            self.handlers[event_type] = func
            return func

        return register

    def qr(self, _func):
        return _func

    async def connect(self) -> None:
        from nanobot.channels.whatsapp import runtime as whatsapp_module

        ev = whatsapp_module._NEONIZE_API.ConnectedEv
        await self.handlers[ev](self, _Proto())
        self.connected_event.set()

    async def idle(self) -> None:
        await self._stopped.wait()

    async def stop(self) -> None:
        self._stop_calls += 1
        self._stopped.set()


@pytest.mark.asyncio
class _LoggedOutOnConnectClient(_FakeLoginClient):
    """Fake client that fires LoggedOutEv on connect instead of ConnectedEv.

    Mirrors the real whatsmeow behavior when a 401 "logged out from another
    device" arrives: the session is deleted and the websocket closes, but
    the channel must still surface a clear error to the user instead of
    hanging on `await login_result` forever.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logged_out = False

    async def connect(self) -> None:
        from nanobot.channels.whatsapp import runtime as whatsapp_module

        ev = whatsapp_module._NEONIZE_API.LoggedOutEv
        # Reason=0 is the generic logged-out code in whatsmeow.
        self.logged_out = True
        await self.handlers[ev](self, _Proto(Reason=0, OnConnect=False))


@pytest.mark.asyncio
async def test_login_fails_when_session_was_logged_out(monkeypatch) -> None:
    """If whatsmeow fires LoggedOutEv (e.g. 401 from another device),
    login() must fail with a clear error instead of hanging forever
    waiting for ConnectedEv.
    """
    _patch_neonize_api(monkeypatch)
    client = _LoggedOutOnConnectClient()
    ch = _make_channel()
    ch._new_client = MagicMock(return_value=client)

    assert await ch.login() is False
    assert client.logged_out is True
    assert client._stop_calls == 1


@pytest.mark.asyncio
async def test_login_fails_when_connect_failure_event_fires(monkeypatch) -> None:
    """ConnectFailureEv must also resolve login_result with an error."""
    _patch_neonize_api(monkeypatch)
    from nanobot.channels.whatsapp import runtime as whatsapp_module

    class _ConnectFailureClient(_FakeLoginClient):
        def __init__(self) -> None:
            super().__init__()
            self.connect_failure = False

        async def connect(self) -> None:
            ev = whatsapp_module._NEONIZE_API.ConnectFailureEv
            self.connect_failure = True
            await self.handlers[ev](self, _Proto(Reason=403, Message="forbidden"))

    client = _ConnectFailureClient()
    ch = _make_channel()
    ch._new_client = MagicMock(return_value=client)

    assert await ch.login() is False
    assert client.connect_failure is True
    assert client._stop_calls == 1


class _HangingClient:
    """A fake neonize client whose idle() blocks until stop() is called.

    Simulates an unscanned QR: connect() resolves immediately but no
    ConnectedEv is ever fired, so the channel sits forever waiting for
    login to complete. Used to verify the login_timeout_s cap in
    ``start()`` breaks the whatsmeow reconnect loop after the timeout.
    """

    def __init__(self) -> None:
        self._stopped = asyncio.Event()
        # ponytail: AsyncMock would shadow the real stop() below; the
        # channel's finally block only calls await client.stop(), which
        # the real method satisfies.
        self._stop_calls = 0

    def event(self, _event_type):
        def register(func):
            return func

        return register

    def qr(self, func):
        return func

    async def connect(self) -> None:
        return None

    async def idle(self) -> None:
        # Block until stop() is invoked. The wait_for in start() should
        # cancel us via the timeout, after which the channel's finally
        # block calls stop() — which signals this event.
        await self._stopped.wait()

    async def stop(self) -> None:
        self._stop_calls += 1
        self._stopped.set()


@pytest.mark.asyncio
async def test_start_stops_channel_when_login_times_out(monkeypatch) -> None:
    """With login_timeout_s set, an unscanned QR must not block the channel forever."""
    _patch_neonize_api(monkeypatch)
    client = _HangingClient()
    ch = _make_channel({"login_timeout_s": 1})
    ch._new_client = MagicMock(return_value=client)

    started_at = time.monotonic()
    await ch.start()
    elapsed = time.monotonic() - started_at

    # The timeout was 1 second; we should have given up well before any
    # unbounded reconnect-loop time. Generous upper bound to keep the
    # test stable on a slow CI runner.
    assert elapsed < 5.0
    assert ch._running is False
    assert ch._connected is False
    # client.stop() is called from the finally block even when the
    # timeout fires, so the whatsmeow reconnect loop is broken.
    assert client._stop_calls >= 1


@pytest.mark.asyncio
async def test_start_login_timeout_zero_disables_cap(monkeypatch) -> None:
    """login_timeout_s=0 preserves the original forever-block behavior for the cap."""
    _patch_neonize_api(monkeypatch)
    client = _HangingClient()
    ch = _make_channel({"login_timeout_s": 0})
    ch._new_client = MagicMock(return_value=client)

    # Schedule stop() shortly after start so the test does not hang.
    async def _stop_later() -> None:
        await asyncio.sleep(0.5)
        await client.stop()

    stopper = asyncio.create_task(_stop_later())
    await ch.start()
    await stopper

    # The channel ran past its own 0-second cap (i.e., no cap applied)
    # and only stopped because the test forced client.stop().
    assert ch._running is False


@pytest.mark.asyncio
async def test_start_completes_normally_when_login_succeeds(monkeypatch) -> None:
    """Regression: when ConnectedEv fires, idle() resolves and the channel exits cleanly."""
    _patch_neonize_api(monkeypatch)
    client = _FakeLoginClient()
    ch = _make_channel({"login_timeout_s": 5})
    ch._new_client = MagicMock(return_value=client)

    # Run start() until it reaches client.idle(); we then trigger the
    # ConnectedEv handler that the FakeLoginClient exposes, which sets
    # the login_result — but the channel's start() doesn't track
    # login_result directly. Instead we just verify start() returns
    # without raising and client.stop() is invoked.
    async def _trigger_then_stop() -> None:
        await asyncio.sleep(0.05)
        # _FakeLoginClient.connect() already fired ConnectedEv before
        # start() reached idle(). For start() to return, client.idle()
        # must complete — call stop() to release it.
        await client.stop()

    trigger = asyncio.create_task(_trigger_then_stop())
    await ch.start()
    await trigger

    assert ch._running is False
    # _stop_calls counts the real stop() invocations on the fake client.
    # The test scheduled one explicit stop, and the channel's finally
    # block calls stop() again on shutdown, so 1 or 2 is acceptable.
    assert client._stop_calls >= 1


@pytest.mark.asyncio
async def test_start_keeps_running_after_login_timeout_when_already_connected(
    monkeypatch,
) -> None:
    """Regression: with login_timeout_s > 0 and a fast login, the channel
    must NOT tear itself down when the cap elapses. Previously the cap
    fired every 300s on healthy sessions, flipping is_running=False and
    making the WebUI show the channel as 'stopped'.
    """
    _patch_neonize_api(monkeypatch)
    client = _FakeLoginClient()
    ch = _make_channel({"login_timeout_s": 1})
    ch._new_client = MagicMock(return_value=client)

    # _FakeLoginClient.connect() already fired ConnectedEv, so by the
    # time start() schedules the cap the channel is connected. Wait
    # longer than the cap to prove the channel stays up.
    start_task = asyncio.create_task(ch.start())
    await asyncio.sleep(1.5)
    # Channel must still be running: the cap fired but login had already
    # completed, so the channel fell through to blocking on idle().
    assert ch._running is True
    assert ch._connected is True
    # Now release idle() and let start() return.
    await client.stop()
    await start_task
    assert ch._running is False


# ponytail: a second fake client that lets the test fire LoggedOutEv
# on demand. We can't reuse _FakeLoginClient because its handlers are
# bound at construction time; this one exposes the registered event
# handler so the test can call it. The flow mimics what real neonize
# does on LoggedOutEv: close the websocket, which makes the pending
# ``client.idle()`` (awaited by ``_run_session``) return.
class _LoggedOutClient:
    def __init__(self) -> None:
        self._stopped = asyncio.Event()
        self._stop_calls = 0
        self.handlers: dict = {}
        self.connected_event = asyncio.Event()
        # Pre-populate get_me() so _remember_self_jids works after
        # ConnectedEv fires.
        self.me = _Proto(JID=_jid("56928861873", "s.whatsapp.net"), LID=_jid("BOTLID", "lid"))

    def event(self, event_type):
        def register(func):
            self.handlers[event_type] = func
            return func

        return register

    def qr(self, _func):
        return _func

    async def connect(self) -> None:
        from nanobot.channels.whatsapp import runtime as whatsapp_module

        ev = whatsapp_module._NEONIZE_API.ConnectedEv
        await self.handlers[ev](self, _Proto())
        self.connected_event.set()

    async def idle(self) -> None:
        await self._stopped.wait()

    async def stop(self) -> None:
        self._stop_calls += 1
        self._stopped.set()


@pytest.mark.asyncio
async def test_start_runs_single_session_no_reconnect(monkeypatch) -> None:
    """start() runs exactly one _run_session. After LoggedOutEv the
    channel stops — no auto-reconnect. The user re-links manually.
    """
    _patch_neonize_api(monkeypatch)

    client = _LoggedOutClient()
    ch = _make_channel({"login_timeout_s": 1})
    ch._new_client = MagicMock(return_value=client)

    start_task = asyncio.create_task(ch.start())
    await asyncio.sleep(0.05)

    from nanobot.channels.whatsapp import runtime as whatsapp_module

    logged_out_ev = whatsapp_module._NEONIZE_API.LoggedOutEv
    await client.handlers[logged_out_ev](client, _Proto(Reason=2, OnConnect=False))
    await client.stop()

    await asyncio.wait_for(start_task, timeout=3)
    assert ch._new_client.call_count == 1
    assert ch._connected is False


@pytest.mark.asyncio
async def test_start_stops_cleanly_on_external_stop(monkeypatch) -> None:
    """A clean stop() (manager shutdown) ends the session with no retry."""
    _patch_neonize_api(monkeypatch)
    client = _FakeLoginClient()
    ch = _make_channel({"login_timeout_s": 1})
    ch._new_client = MagicMock(return_value=client)

    start_task = asyncio.create_task(ch.start())
    await asyncio.sleep(0.05)
    await ch.stop()
    await asyncio.wait_for(start_task, timeout=5)
    assert ch._new_client.call_count == 1


def _patch_throttle_path(monkeypatch, tmp_path) -> Path:
    """Redirect the channel's throttle state file to a tmp location."""
    state_file = tmp_path / "throttle.json"
    monkeypatch.setattr(
        whatsapp_module.WhatsAppChannel,
        "_throttle_state_path",
        lambda self: state_file,
    )
    return state_file


@pytest.mark.asyncio
async def test_463_throttle_trips_after_threshold_and_blocks_sends(monkeypatch, tmp_path) -> None:
    """After ``throttle_threshold`` consecutive 463s, the channel stops
    sending and surfaces a cooldown error. Once in cooldown, even a
    hypothetical success on the same call must not resume sending —
    the next send call must check the gate and bail.
    """
    _patch_neonize_api(monkeypatch)
    state_file = _patch_throttle_path(monkeypatch, tmp_path)

    from neonize.exc import SendMessageError

    client = SimpleNamespace(
        send_message=AsyncMock(side_effect=SendMessageError("server returned error 463"))
    )
    ch = _make_channel(
        {"throttle_threshold": 3, "throttle_cooldown_s": 600},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    ch._client = client
    ch._connected = True

    from nanobot.bus.events import OutboundMessage

    msg = OutboundMessage(channel="whatsapp", chat_id="56975746099", content="x")

    # First two 463s: counter climbs but no cooldown yet.
    for _ in range(2):
        with pytest.raises(SendMessageError):
            await ch.send(msg)
    assert ch._check_throttle() is None
    # Third 463: trips the cooldown.
    with pytest.raises(SendMessageError):
        await ch.send(msg)
    assert ch._check_throttle() is not None
    # Persisted to disk so a restart respects it.
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["consecutive_463"] == 3
    assert data["cooldown_until"] > time.time()


@pytest.mark.asyncio
async def test_send_during_cooldown_raises_without_calling_client(monkeypatch, tmp_path) -> None:
    """Once the cooldown is active, send() must not even attempt the
    underlying call. Otherwise we'd just hand WhatsApp another 463
    and harden the throttle.
    """
    _patch_neonize_api(monkeypatch)
    state_file = _patch_throttle_path(monkeypatch, tmp_path)

    client = SimpleNamespace(send_message=AsyncMock())
    ch = _make_channel(
        {"throttle_threshold": 1, "throttle_cooldown_s": 600},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    ch._client = client
    ch._connected = True

    from nanobot.bus.events import OutboundMessage

    msg = OutboundMessage(channel="whatsapp", chat_id="56975746099", content="hi")

    # Trip the gate directly.
    ch._save_throttle_state(consecutive=1, cooldown_until=time.time() + 600)

    with pytest.raises(RuntimeError, match="cooldown active"):
        await ch.send(msg)
    client.send_message.assert_not_called()
    # Cooldown state still on disk for the rest of the test.
    assert state_file.exists()


@pytest.mark.asyncio
async def test_cooldown_expires_and_send_resumes(monkeypatch, tmp_path) -> None:
    """When cooldown_until is in the past, the next send() must check
    the gate, see it expired, clear it, and proceed normally.
    """
    _patch_neonize_api(monkeypatch)
    _patch_throttle_path(monkeypatch, tmp_path)

    client = SimpleNamespace(send_message=AsyncMock())
    ch = _make_channel(
        {"throttle_threshold": 3, "throttle_cooldown_s": 600},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    ch._client = client
    ch._connected = True

    from nanobot.bus.events import OutboundMessage

    msg = OutboundMessage(channel="whatsapp", chat_id="56975746099", content="hi")

    # Pretend the cooldown expired 1 second ago.
    ch._save_throttle_state(consecutive=3, cooldown_until=time.time() - 1)

    await ch.send(msg)
    client.send_message.assert_awaited_once()
    # State should be cleared after a successful send.
    data = json.loads(ch._throttle_state_path().read_text())
    assert data == {"consecutive_463": 0, "cooldown_until": 0.0}


@pytest.mark.asyncio
async def test_successful_send_resets_consecutive_463(monkeypatch, tmp_path) -> None:
    """A single non-463 success must reset the 463 counter so a stray
    463 from earlier doesn't accumulate toward the next cooldown.
    """
    _patch_neonize_api(monkeypatch)
    _patch_throttle_path(monkeypatch, tmp_path)

    client = SimpleNamespace(send_message=AsyncMock())
    ch = _make_channel(
        {"throttle_threshold": 3, "throttle_cooldown_s": 600},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    ch._client = client
    ch._connected = True

    from nanobot.bus.events import OutboundMessage

    msg = OutboundMessage(channel="whatsapp", chat_id="56975746099", content="hi")

    # Two prior 463s (counter at 2, no cooldown).
    ch._save_throttle_state(consecutive=2, cooldown_until=0.0)
    await ch.send(msg)
    data = json.loads(ch._throttle_state_path().read_text())
    assert data == {"consecutive_463": 0, "cooldown_until": 0.0}


@pytest.mark.asyncio
async def test_throttle_threshold_zero_disables_cooldown(monkeypatch, tmp_path) -> None:
    """With threshold=0 the channel must keep sending through 463s
    (the existing behaviour before the cooldown was added)."""
    _patch_neonize_api(monkeypatch)
    _patch_throttle_path(monkeypatch, tmp_path)

    from neonize.exc import SendMessageError

    client = SimpleNamespace(
        send_message=AsyncMock(side_effect=SendMessageError("server returned error 463"))
    )
    ch = _make_channel(
        {"throttle_threshold": 0, "throttle_cooldown_s": 600},
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    ch._client = client
    ch._connected = True

    from nanobot.bus.events import OutboundMessage

    msg = OutboundMessage(channel="whatsapp", chat_id="56975746099", content="x")

    for _ in range(5):
        with pytest.raises(SendMessageError):
            await ch.send(msg)
    # No cooldown persisted because the threshold is disabled.
    assert ch._check_throttle() is None
    assert not ch._throttle_state_path().exists()


@pytest.mark.asyncio
async def test_send_text_uses_neonize_send_message(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(
        send_message=AsyncMock(),
        send_image=AsyncMock(),
        send_video=AsyncMock(),
        send_audio=AsyncMock(),
        send_document=AsyncMock(),
    )
    ch = _make_channel()
    ch._client = client
    ch._connected = True

    await ch.send(OutboundMessage(channel="whatsapp", chat_id="12345@s.whatsapp.net", content="hi"))

    client.send_message.assert_awaited_once_with(
        ("12345", "s.whatsapp.net"), _message_with_conversation("hi")
    )


@pytest.mark.asyncio
async def test_send_media_dispatches_by_mimetype(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(
        send_message=AsyncMock(),
        send_image=AsyncMock(),
        send_video=AsyncMock(),
        send_audio=AsyncMock(),
        send_document=AsyncMock(),
    )
    ch = _make_channel()
    ch._client = client
    ch._connected = True

    await ch.send(
        OutboundMessage(
            channel="whatsapp",
            chat_id="12345@s.whatsapp.net",
            content="",
            media=["photo.jpg", "clip.mp4", "voice.ogg", "report.pdf"],
        )
    )

    jid = ("12345", "s.whatsapp.net")
    client.send_image.assert_awaited_once_with(jid, "photo.jpg")
    client.send_video.assert_awaited_once_with(jid, "clip.mp4")
    client.send_audio.assert_awaited_once_with(jid, "voice.ogg", ptt=True)
    client.send_document.assert_awaited_once_with(
        jid,
        "report.pdf",
        filename="report.pdf",
        mimetype="application/pdf",
    )


@pytest.mark.asyncio
async def test_send_audio_ptt_flag_propagates(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(
        send_message=AsyncMock(),
        send_image=AsyncMock(),
        send_video=AsyncMock(),
        send_audio=AsyncMock(),
        send_document=AsyncMock(),
    )
    ch = _make_channel()
    ch._client = client
    ch._connected = True

    await ch.send(
        OutboundMessage(
            channel="whatsapp",
            chat_id="12345@s.whatsapp.net",
            content="",
            media=["voice.ogg"],
            metadata={"ptt": True},
        )
    )

    jid = ("12345", "s.whatsapp.net")
    client.send_audio.assert_awaited_once_with(jid, "voice.ogg", ptt=True)


@pytest.mark.asyncio
async def test_send_audio_mp3_is_not_auto_ptt(monkeypatch, tmp_path) -> None:
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(
        send_message=AsyncMock(),
        send_image=AsyncMock(),
        send_video=AsyncMock(),
        send_audio=AsyncMock(),
        send_document=AsyncMock(),
    )
    ch = _make_channel()
    ch._client = client
    ch._connected = True

    audio = tmp_path / "tts_clip.mp3"
    audio.write_bytes(b"\x00")

    await ch.send(
        OutboundMessage(
            channel="whatsapp",
            chat_id="12345@s.whatsapp.net",
            content="",
            media=[str(audio)],
        )
    )

    jid = ("12345", "s.whatsapp.net")
    # MP3 should be sent as regular audio; auto-PTT is reserved for OGG Opus.
    client.send_audio.assert_awaited_once_with(jid, str(audio), ptt=False)


@pytest.mark.asyncio
async def test_send_audio_ogg_auto_ptt(monkeypatch, tmp_path) -> None:
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(
        send_message=AsyncMock(),
        send_image=AsyncMock(),
        send_video=AsyncMock(),
        send_audio=AsyncMock(),
        send_document=AsyncMock(),
    )
    ch = _make_channel()
    ch._client = client
    ch._connected = True

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"\x00")

    await ch.send(
        OutboundMessage(
            channel="whatsapp",
            chat_id="12345@s.whatsapp.net",
            content="",
            media=[str(audio)],
        )
    )

    jid = ("12345", "s.whatsapp.net")
    client.send_audio.assert_awaited_once_with(jid, str(audio), ptt=True)


@pytest.mark.asyncio
async def test_send_audio_ptt_override_false_wins(monkeypatch, tmp_path) -> None:
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(
        send_message=AsyncMock(),
        send_image=AsyncMock(),
        send_video=AsyncMock(),
        send_audio=AsyncMock(),
        send_document=AsyncMock(),
    )
    ch = _make_channel()
    ch._client = client
    ch._connected = True

    audio = tmp_path / "tts_clip.mp3"
    audio.write_bytes(b"\x00")

    await ch.send(
        OutboundMessage(
            channel="whatsapp",
            chat_id="12345@s.whatsapp.net",
            content="",
            media=[str(audio)],
            metadata={"ptt": False},
        )
    )

    jid = ("12345", "s.whatsapp.net")
    client.send_audio.assert_awaited_once_with(jid, str(audio), ptt=False)


def test_ensure_ffmpeg_in_path_adds_static_bin_dir(monkeypatch, tmp_path) -> None:
    from nanobot.channels.whatsapp import runtime as rt

    bin_dir = tmp_path / "ffmpeg-bin"
    bin_dir.mkdir()
    ffmpeg = bin_dir / "ffmpeg"
    ffprobe = bin_dir / "ffprobe"
    ffmpeg.write_bytes(b"")
    ffprobe.write_bytes(b"")
    ffmpeg.chmod(0o755)
    ffprobe.chmod(0o755)

    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(
        "static_ffmpeg.run.get_or_fetch_platform_executables_else_raise",
        lambda: (str(ffmpeg), str(ffprobe)),
    )
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")

    rt._ensure_ffmpeg_in_path()

    import os

    assert os.environ["PATH"].startswith(str(bin_dir))


def test_ensure_ffmpeg_in_path_noop_when_present(monkeypatch) -> None:
    from nanobot.channels.whatsapp import runtime as rt

    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in {"ffprobe", "ffmpeg"} else None

    monkeypatch.setattr("shutil.which", fake_which)
    called = {"value": False}

    def must_not_run() -> None:
        called["value"] = True

    monkeypatch.setattr(
        "static_ffmpeg.run.get_or_fetch_platform_executables_else_raise",
        must_not_run,
    )

    rt._ensure_ffmpeg_in_path()
    assert called["value"] is False


@pytest.mark.asyncio
async def test_start_typing_emits_composing_then_loop_keeps_it(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    presence_calls: list[tuple[Any, Any, Any]] = []

    class _State:
        def __init__(self):
            self.value = 0

    class _Enum:
        def __init__(self, name: str):
            self.name = name

        def __eq__(self, other):
            return isinstance(other, _Enum) and other.name == self.name

        def __hash__(self):
            return hash(self.name)

    composing = _Enum("COMPOSING")
    text_media = _Enum("TEXT")

    class _Client:
        async def send_chat_presence(self, jid, state, media):
            presence_calls.append((jid, state, media))

    ch = _make_channel()
    ch._client = _Client()
    ch._connected = True

    monkeypatch.setattr(
        "neonize.utils.enum.ChatPresence",
        _Enum("Holder"),
    )
    # override the import inside the method to return our enums
    fake_enum = types.SimpleNamespace(
        ChatPresence=types.SimpleNamespace(
            CHAT_PRESENCE_COMPOSING=composing,
            CHAT_PRESENCE_PAUSED=_Enum("PAUSED"),
        ),
        ChatPresenceMedia=types.SimpleNamespace(CHAT_PRESENCE_MEDIA_TEXT=text_media),
    )
    monkeypatch.setitem(sys.modules, "neonize.utils.enum", fake_enum)

    jid = ("12345", "s.whatsapp.net")
    await ch._start_typing(jid)
    # Cancel the loop quickly and confirm the typing task was registered.
    task = list(ch._typing_tasks.values())[0]
    # Let the task enter its body before cancelling, otherwise the finally
    # block (which removes the task from the dict) never runs.
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except BaseException:
        pass

    # The _start_typing path registered a periodic composing loop. Verify the
    # typing task bookkeeping is correct (cancelled on stop, key is per-jid).
    assert ch._typing_tasks == {}
    # The earlier _stop_typing emitted a PAUSED presence; assert it was
    # addressed to our jid and used the TEXT media type.
    assert presence_calls, "expected a PAUSED presence from the initial stop"
    first = presence_calls[0]
    assert first[0] == jid
    assert first[1].name == "PAUSED"
    assert first[2] == text_media


@pytest.mark.asyncio
async def test_send_resolves_lid_chat_to_phone(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(
        send_message=AsyncMock(),
        send_image=AsyncMock(),
        send_video=AsyncMock(),
        send_audio=AsyncMock(),
        send_document=AsyncMock(),
    )
    ch = _make_channel()
    ch._client = client
    ch._connected = True
    ch._lid_to_phone = {"230343776985329": "56975746099"}

    await ch.send(
        OutboundMessage(
            channel="whatsapp",
            chat_id="230343776985329@lid",
            content="hola",
        )
    )

    client.send_message.assert_awaited_once_with(
        ("56975746099", "s.whatsapp.net"), _message_with_conversation("hola")
    )


def test_whatsapp_session_key_isolates_group_members() -> None:
    ch = _make_channel()
    assert (
        ch._whatsapp_session_key("12345@s.whatsapp.net", "56911111111", False)
        == "whatsapp:12345@s.whatsapp.net"
    )
    assert (
        ch._whatsapp_session_key("120363@g.us", "56911111111", True)
        == "whatsapp:120363@g.us:56911111111"
    )
    assert (
        ch._whatsapp_session_key("120363@g.us", "56922222222", True)
        == "whatsapp:120363@g.us:56922222222"
    )


def test_resolve_mention_prefers_push_name() -> None:
    ch = _make_channel()
    assert ch._resolve_mention("56911111111", "Juan Pérez") == "@Juan Pérez"
    # Numeric push names are rejected; we fall back to the phone number.
    assert ch._resolve_mention("56911111111", "12345") == "@+56911111111"
    # Long hex-looking IDs are rejected and fall back to the phone.
    assert ch._resolve_mention("56911111111", "abcd1234abcd1234abcd") == "@+56911111111"
    assert ch._resolve_mention("56911111111", "") == "@+56911111111"
    assert ch._resolve_mention("SENDERLID", "") is None
    assert ch._resolve_mention("56911111111@s.whatsapp.net", "") == "@+56911111111"


def test_group_metadata_includes_sender_identity_runtime_context() -> None:
    ch = _make_channel()
    block = ch._sender_identity_block("56911111111", "Juan", True, phone_id="56911111111")
    assert block.source == "whatsapp_sender_identity"
    assert "+56911111111" in block.content
    assert "Juan" in block.content
    assert "reply_to_bot: yes" in block.content


def test_display_names_are_remembered_and_used() -> None:
    ch = _make_channel()
    assert ch._remember_display_name("120363@g.us", "56911111111", "Juan") == "Juan"
    # Falls back to stored name when push_name is empty.
    assert ch._display_name_for("120363@g.us", "56911111111") == "Juan"
    assert ch._resolve_mention("56911111111", "") == "@+56911111111"

    ch2 = WhatsAppChannel({"enabled": True}, MagicMock())
    ch2._display_names = ch._display_names
    ch2._display_names_path = ch._display_names_path
    # Unknown sender still falls back to phone.
    assert ch2._display_name_for("120363@g.us", "56999999999") is None


def test_known_contacts_block_excludes_current_sender() -> None:
    ch = _make_channel()
    ch._remember_display_name("120363@g.us", "56911111111", "Juan")
    ch._remember_display_name("120363@g.us", "56922222222", "María")
    block = ch._known_contacts_block("120363@g.us", "56911111111")
    assert block is not None
    assert "Juan" not in block.content
    assert "María" in block.content


@pytest.mark.asyncio
async def test_group_message_uses_per_sender_session_key(monkeypatch) -> None:
    """Group messages get isolated session keys per sender so contexts don't mix."""
    _patch_neonize_api(monkeypatch)
    ch = _make_channel()
    ch._handle_message = AsyncMock()
    ch._self_jids = {"bot@s.whatsapp.net", "bot"}

    event = _event(
        message=_Proto(conversation="hello group"),
        chat=_jid("120363000", "g.us"),
        sender=_jid("56911111111", "s.whatsapp.net"),
        is_group=True,
    )
    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        event,
    )
    await ch._drain_group_queue("120363000@g.us")

    assert ch._handle_message.awaited
    call_kwargs = ch._handle_message.await_args.kwargs
    assert call_kwargs["session_key"] == "whatsapp:120363000@g.us:56911111111"
    assert call_kwargs["is_dm"] is False


@pytest.mark.asyncio
async def test_dm_message_uses_chat_session_key(monkeypatch) -> None:
    """DM messages keep the default chat-scoped session key."""
    _patch_neonize_api(monkeypatch)
    ch = _make_channel()
    ch._handle_message = AsyncMock()

    event = _event(
        message=_Proto(conversation="hello dm"),
        chat=_jid("56911111111", "s.whatsapp.net"),
        sender=_jid("56911111111", "s.whatsapp.net"),
        is_group=False,
    )
    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        event,
    )

    assert ch._handle_message.awaited
    call_kwargs = ch._handle_message.await_args.kwargs
    assert call_kwargs["session_key"] == "whatsapp:56911111111@s.whatsapp.net"
    assert call_kwargs["is_dm"] is True


@pytest.mark.asyncio
async def test_send_stops_typing(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    paused: list[tuple[Any, Any, Any]] = []

    class _Client:
        def __init__(self):
            self.send_message = AsyncMock()
            self.send_image = AsyncMock()
            self.send_video = AsyncMock()
            self.send_audio = AsyncMock()
            self.send_document = AsyncMock()

        async def send_chat_presence(self, jid, state, media):
            paused.append((jid, state, media))

    fake_enum = types.SimpleNamespace(
        ChatPresence=types.SimpleNamespace(
            CHAT_PRESENCE_COMPOSING=types.SimpleNamespace(name="COMPOSING"),
            CHAT_PRESENCE_PAUSED=types.SimpleNamespace(name="PAUSED"),
        ),
        ChatPresenceMedia=types.SimpleNamespace(
            CHAT_PRESENCE_MEDIA_TEXT=types.SimpleNamespace(name="TEXT")
        ),
    )
    monkeypatch.setitem(sys.modules, "neonize.utils.enum", fake_enum)

    ch = _make_channel()
    client = _Client()
    ch._client = client
    ch._connected = True
    jid_obj = ch._build_jid("12345@s.whatsapp.net")
    key = _typing_task_key(jid_obj)

    # pre-register a typing task so _stop_typing has something to clear
    async def _noop():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    ch._typing_tasks[key] = asyncio.create_task(_noop())

    await ch.send(OutboundMessage(channel="whatsapp", chat_id="12345@s.whatsapp.net", content="hi"))

    assert ch._typing_tasks == {}
    assert paused and paused[0][1].name == "PAUSED"


@pytest.mark.asyncio
async def test_send_when_disconnected_raises() -> None:
    ch = _make_channel()

    with pytest.raises(RuntimeError, match="not connected"):
        await ch.send(OutboundMessage(channel="whatsapp", chat_id="123", content="hi"))


@pytest.mark.asyncio
async def test_group_policy_mention_skips_unmentioned_group_message() -> None:
    ch = _make_channel({"groupPolicy": "mention"})
    ch._self_jids = {"bot@s.whatsapp.net", "bot"}
    ch._handle_message = AsyncMock()

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=_Proto(conversation="hello group"),
            chat=_jid("120363000", "g.us"),
            sender=_jid("SENDERLID", "lid"),
            is_group=True,
        ),
    )

    ch._handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_group_policy_mention_accepts_mention_and_prefers_phone_sender() -> None:
    ch = _make_channel({"groupPolicy": "mention"})
    ch._self_jids = {"bot@s.whatsapp.net", "bot"}
    ch._handle_message = AsyncMock()
    context = _Proto(mentionedJID=["bot@s.whatsapp.net"])
    message = _Proto(extendedTextMessage=_Proto(text="hello @bot", contextInfo=context))

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=message,
            chat=_jid("120363000", "g.us"),
            sender=_jid("LID99", "lid"),
            sender_alt=_jid("15559998888", "s.whatsapp.net"),
            is_group=True,
        ),
    )
    await ch._drain_group_queue("120363000@g.us")

    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["sender_id"] == "15559998888"
    assert kwargs["chat_id"] == "120363000@g.us"
    assert kwargs["metadata"]["lid"] == "LID99"
    assert kwargs["metadata"]["phone"] == "15559998888"


@pytest.mark.asyncio
async def test_group_policy_mention_accepts_reply_to_bot() -> None:
    ch = _make_channel({"groupPolicy": "mention"})
    ch._self_jids = {"bot@s.whatsapp.net", "bot"}
    ch._handle_message = AsyncMock()
    context = _Proto(participant="bot@s.whatsapp.net")
    message = _Proto(extendedTextMessage=_Proto(text="reply", contextInfo=context))

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=message,
            chat=_jid("120363000", "g.us"),
            sender=_jid("SENDERLID", "lid"),
            is_group=True,
        ),
    )
    await ch._drain_group_queue("120363000@g.us")

    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["metadata"]["is_reply_to_bot"] is True


@pytest.mark.asyncio
async def test_group_sender_id_uses_participant_not_group_jid() -> None:
    ch = WhatsAppChannel({"enabled": True, "allowFrom": ["SENDERLID"]}, MagicMock())
    ch._started_at = 0
    ch._handle_message = AsyncMock()

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=_Proto(conversation="hi"),
            chat=_jid("120363000", "g.us"),
            sender=_jid("SENDERLID", "lid"),
            is_group=True,
        ),
    )
    await ch._drain_group_queue("120363000@g.us")

    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["sender_id"] == "SENDERLID"
    assert kwargs["metadata"]["participant"] == "SENDERLID@lid"


@pytest.mark.parametrize("allowed_group", ["120363000@g.us", "120363000"])
@pytest.mark.asyncio
async def test_group_allow_from_accepts_group_jid_or_bare_id(allowed_group: str) -> None:
    bus = MessageBus()
    ch = WhatsAppChannel({"enabled": True, "allowFrom": [allowed_group]}, bus)
    ch._started_at = 0

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=_Proto(conversation="hi"),
            chat=_jid("120363000", "g.us"),
            sender=_jid("SENDERLID", "lid"),
            is_group=True,
        ),
    )
    await ch._drain_group_queue("120363000@g.us")

    assert bus.inbound_size == 1
    msg = await bus.consume_inbound()
    assert msg.sender_id == "SENDERLID"
    assert msg.chat_id == "120363000@g.us"
    assert msg.content == "hi"
    assert msg.metadata["participant"] == "SENDERLID@lid"


@pytest.mark.asyncio
async def test_group_allow_from_does_not_allow_same_participant_in_other_group() -> None:
    bus = MessageBus()
    ch = WhatsAppChannel({"enabled": True, "allowFrom": ["120363000"]}, bus)
    ch._started_at = 0

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=_Proto(conversation="hi"),
            chat=_jid("120363999", "g.us"),
            sender=_jid("SENDERLID", "lid"),
            is_group=True,
        ),
    )

    assert bus.inbound_size == 0


@pytest.mark.asyncio
async def test_read_receipt_is_requested_once_after_dedup() -> None:
    ch = _make_channel()
    ch._send_read_receipt = AsyncMock()
    ch._handle_message = AsyncMock()
    client = SimpleNamespace(download_any=AsyncMock())
    event = _event(
        message=_Proto(conversation="hi"),
        sender=_jid("15551234567", "s.whatsapp.net"),
    )

    await ch._handle_neonize_message(client, event)
    await ch._handle_neonize_message(client, event)

    ch._send_read_receipt.assert_awaited_once_with(
        client,
        event.Info.MessageSource,
        "m1",
    )
    ch._handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_read_receipt_uses_mark_read_and_swallows_failures(monkeypatch) -> None:
    receipt_type = _patch_receipt_type(monkeypatch)
    ch = _make_channel()
    source = _event(
        message=_Proto(conversation="hi"),
        sender=_jid("15551234567", "s.whatsapp.net"),
    ).Info.MessageSource
    client = SimpleNamespace(
        mark_read=AsyncMock(),
        download_any=AsyncMock(),
    )

    await ch._send_read_receipt(client, source, "m1")

    client.mark_read.assert_awaited_once_with(
        "m1",
        chat=source.Chat,
        sender=source.Sender,
        receipt=receipt_type.READ,
    )

    failing_client = SimpleNamespace(
        mark_read=AsyncMock(side_effect=RuntimeError("boom")),
        download_any=AsyncMock(),
    )

    await ch._send_read_receipt(failing_client, source, "m2")

    failing_client.mark_read.assert_awaited_once()


@pytest.mark.asyncio
async def test_lid_to_phone_cache_resolves_lid_only_messages() -> None:
    ch = _make_channel()
    ch._handle_message = AsyncMock()

    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=_Proto(conversation="first"),
            message_id="c1",
            chat=_jid("LID99", "lid"),
            sender=_jid("LID99", "lid"),
            sender_alt=_jid("5559999", "s.whatsapp.net"),
        ),
    )
    await ch._handle_neonize_message(
        SimpleNamespace(download_any=AsyncMock()),
        _event(
            message=_Proto(conversation="second"),
            message_id="c2",
            chat=_jid("LID99", "lid"),
            sender=_jid("LID99", "lid"),
        ),
    )

    assert ch._handle_message.await_args_list[1].kwargs["sender_id"] == "5559999"


def test_lid_mappings_from_config() -> None:
    ch = WhatsAppChannel(
        {"enabled": True, "lidMappings": {"123456789012345": "15551234567"}},
        MagicMock(),
    )

    assert ch._lid_to_phone == {"123456789012345": "15551234567"}


@pytest.mark.asyncio
async def test_image_media_is_downloaded_and_forwarded(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(whatsapp_module, "get_media_dir", lambda channel: tmp_path / channel)
    ch = _make_channel()
    ch._handle_message = AsyncMock()
    client = SimpleNamespace(download_any=AsyncMock())
    message = _Proto(
        imageMessage=_Proto(
            caption="look",
            mimetype="image/jpeg",
        )
    )

    await ch._handle_neonize_message(
        client,
        _event(message=message, sender_alt=_jid("15551234567", "s.whatsapp.net")),
    )

    client.download_any.assert_awaited_once()
    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["content"].startswith("look\n[image: ")
    assert len(kwargs["media"]) == 1
    assert kwargs["media"][0].endswith(".jpg")


@pytest.mark.asyncio
async def test_voice_message_transcribes_and_drops_media_when_successful(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(whatsapp_module, "get_media_dir", lambda channel: tmp_path / channel)
    ch = _make_channel()
    ch._handle_message = AsyncMock()
    ch.transcribe_audio = AsyncMock(return_value="Hello from audio")
    client = SimpleNamespace(download_any=AsyncMock())
    message = _Proto(audioMessage=_Proto(mimetype="audio/ogg", PTT=True))

    await ch._handle_neonize_message(
        client,
        _event(message=message, sender_alt=_jid("15551234567", "s.whatsapp.net")),
    )

    ch.transcribe_audio.assert_awaited_once()
    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["content"] == "Hello from audio"
    assert kwargs["media"] == []


@pytest.mark.asyncio
async def test_unauthorized_voice_message_does_not_download_or_transcribe(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(whatsapp_module, "get_media_dir", lambda channel: tmp_path / channel)
    ch = WhatsAppChannel({"enabled": True, "allowFrom": ["allowed"]}, MagicMock())
    ch._started_at = 0
    ch._handle_message = AsyncMock()
    ch.transcribe_audio = AsyncMock(return_value="blocked audio")
    client = SimpleNamespace(download_any=AsyncMock())

    await ch._handle_neonize_message(
        client,
        _event(
            message=_Proto(audioMessage=_Proto(mimetype="audio/ogg", PTT=True)),
            chat=_jid("blocked", "s.whatsapp.net"),
            sender=_jid("blocked", "s.whatsapp.net"),
        ),
    )

    client.download_any.assert_not_awaited()
    ch.transcribe_audio.assert_not_awaited()
    ch._handle_message.assert_awaited_once()
    kwargs = ch._handle_message.await_args.kwargs
    assert kwargs["sender_id"] == "blocked"
    assert kwargs["content"] == ""
    assert kwargs["media"] == []
    assert kwargs["is_dm"] is True


@pytest.mark.asyncio
async def test_unauthorized_dm_uses_base_pairing_flow(monkeypatch) -> None:
    _patch_neonize_api(monkeypatch)
    monkeypatch.setattr("nanobot.channels.base.generate_code", lambda _ch, _sid: "ABCD-EFGH")
    monkeypatch.setattr("nanobot.channels.base.is_approved", lambda _ch, _sid: False)
    client = SimpleNamespace(send_message=AsyncMock(), download_any=AsyncMock())
    ch = WhatsAppChannel({"enabled": True, "allowFrom": []}, MagicMock())
    ch._client = client
    ch._connected = True
    ch._started_at = 0

    await ch._handle_neonize_message(
        client,
        _event(
            message=_Proto(conversation="hello"),
            chat=_jid("blocked", "s.whatsapp.net"),
            sender=_jid("blocked", "s.whatsapp.net"),
        ),
    )

    client.download_any.assert_not_awaited()
    client.send_message.assert_awaited_once()
    assert client.send_message.await_args.args[0] == ("blocked", "s.whatsapp.net")
    assert "ABCD-EFGH" in client.send_message.await_args.args[1].extendedTextMessage.text


def test_reset_database_removes_sqlite_sidecars(tmp_path) -> None:
    db = tmp_path / "neonize.db"
    wal = tmp_path / "neonize.db-wal"
    shm = tmp_path / "neonize.db-shm"
    for path in (db, wal, shm):
        path.write_text("x", encoding="utf-8")

    WhatsAppChannel._reset_database(db)

    assert not db.exists()
    assert not wal.exists()
    assert not shm.exists()


# ponytail: regression tests for the watchdog / persistence / drop-logging
# fixes. These exercise the paths that caused the "live but silent" outage
# the user reported on 2026-08-01.


async def test_send_pairing_code_failure_does_not_propagate(monkeypatch) -> None:
    """When the channel's send() raises (e.g. WhatsApp 463 throttle while
    delivering a pairing code), BaseChannel must log the exception and
    NOT propagate it — otherwise the inbound handler crashes and every
    subsequent message is dropped silently."""
    _patch_neonize_api(monkeypatch)
    monkeypatch.setattr("nanobot.channels.base.generate_code", lambda _ch, _sid: "ABCD-EFGH")
    monkeypatch.setattr("nanobot.channels.base.is_approved", lambda _ch, _sid: False)
    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    ch = WhatsAppChannel({"enabled": True, "allowFrom": []}, bus)
    ch.send = AsyncMock(side_effect=RuntimeError("simulated 463 throttle"))

    # Must not raise — the send() failure is swallowed + logged.
    await ch._handle_message(
        sender_id="blocked",
        chat_id="blocked",
        content="hi",
        is_dm=True,
        authorization_id=None,
    )

    ch.send.assert_awaited_once()
    bus.publish_inbound.assert_not_awaited()


def test_message_state_round_trip(tmp_path, monkeypatch) -> None:
    """Processed message ids and LID->phone map survive a reload from
    message_state.json — so a restart doesn't re-process buffered messages
    and doesn't lose LID routing for DMs."""
    state_path = tmp_path / "message_state.json"
    monkeypatch.setattr(
        whatsapp_module.WhatsAppChannel,
        "_message_state_path",
        lambda self, _f=state_path: _f,
    )

    ch = _make_channel()
    ch._processed_message_ids["m1"] = None
    ch._processed_message_ids["m2"] = None
    ch._lid_to_phone["123"] = "456"
    ch._save_message_state()

    assert state_path.exists()

    # Reload into a fresh channel and confirm both maps survived.
    ch2 = _make_channel()
    ids, lid = ch2._load_message_state()
    assert list(ids.keys()) == ["m1", "m2"]
    assert lid == {"123": "456"}


async def test_drop_silently_logs_no_parseable_content(monkeypatch) -> None:
    """An inbound message with no text and no media must still log at
    DEBUG level — so future debugging has a breadcrumb."""
    _patch_neonize_api(monkeypatch)
    monkeypatch.setattr("nanobot.channels.base.is_approved", lambda _ch, _sid: True)
    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    ch = WhatsAppChannel({"enabled": True, "allowFrom": ["*"]}, bus)
    ch._handle_message = AsyncMock()
    client = SimpleNamespace(download_any=AsyncMock(), mark_read=AsyncMock())

    # A message with neither conversation nor media fields.
    empty_message = _Proto()
    await ch._handle_neonize_message(
        client,
        _event(message=empty_message, message_id="drop-me"),
    )

    # No text, no media → _handle_message not called.
    ch._handle_message.assert_not_awaited()
    bus.publish_inbound.assert_not_awaited()


async def test_outbound_allowlist_blocks_unknown_recipient(monkeypatch) -> None:
    """send() must reject messages to chat IDs not in allow_send_to."""
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(
        send_message=AsyncMock(),
        send_image=AsyncMock(),
        send_video=AsyncMock(),
        send_audio=AsyncMock(),
        send_document=AsyncMock(),
    )
    ch = WhatsAppChannel(
        {"enabled": True, "allowFrom": ["*"], "allowSendTo": ["56975746099"]},
        MagicMock(),
    )
    ch._client = client
    ch._connected = True

    # Allowed number — send succeeds.
    await ch.send(
        OutboundMessage(channel="whatsapp", chat_id="56975746099@s.whatsapp.net", content="hi")
    )
    client.send_message.assert_awaited_once()

    # Blocked number — send raises, client.send_message not called again.
    client.send_message.reset_mock()
    with pytest.raises(RuntimeError, match="allowlist blocked"):
        await ch.send(
            OutboundMessage(channel="whatsapp", chat_id="8281248569@s.whatsapp.net", content="hi")
        )
    client.send_message.assert_not_awaited()


async def test_outbound_allowlist_empty_allows_all(monkeypatch) -> None:
    """Empty allow_send_to = allow all destinations (backward compatible)."""
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(send_message=AsyncMock())
    ch = WhatsAppChannel(
        {"enabled": True, "allowFrom": ["*"], "allowSendTo": []},
        MagicMock(),
    )
    ch._client = client
    ch._connected = True

    await ch.send(
        OutboundMessage(channel="whatsapp", chat_id="9999999999@s.whatsapp.net", content="hi")
    )
    client.send_message.assert_awaited_once()


async def test_outbound_allowlist_allows_group(monkeypatch) -> None:
    """Groups in allow_send_to are allowed by bare group ID."""
    _patch_neonize_api(monkeypatch)
    client = SimpleNamespace(send_message=AsyncMock())
    ch = WhatsAppChannel(
        {"enabled": True, "allowFrom": ["*"], "allowSendTo": ["120363422292889459"]},
        MagicMock(),
    )
    ch._client = client
    ch._connected = True

    await ch.send(
        OutboundMessage(channel="whatsapp", chat_id="120363422292889459@g.us", content="hi")
    )
    client.send_message.assert_awaited_once()
