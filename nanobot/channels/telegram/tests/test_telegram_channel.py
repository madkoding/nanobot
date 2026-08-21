import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# Check optional Telegram dependencies before running tests
try:
    import telegram  # noqa: F401
except ImportError:
    pytest.skip("Telegram dependencies not installed (python-telegram-bot)", allow_module_level=True)

from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram.runtime import (
    TELEGRAM_MAX_MESSAGE_LEN,
    TELEGRAM_REPLY_CONTEXT_MAX_LEN,
    TelegramChannel,
    TelegramConfig,
    _markdown_to_telegram_html,
    _split_telegram_markdown,
    _StreamBuf,
)


class _FakeHTTPXRequest:
    instances: list["_FakeHTTPXRequest"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    @classmethod
    def clear(cls) -> None:
        cls.instances.clear()


class _FakeUpdater:
    def __init__(self, on_start_polling) -> None:
        self._on_start_polling = on_start_polling
        self.start_polling_kwargs = None
        self.start_webhook_kwargs = None

    async def start_polling(self, **kwargs) -> None:
        self.start_polling_kwargs = kwargs
        self._on_start_polling()

    async def start_webhook(self, **kwargs) -> None:
        self.start_webhook_kwargs = kwargs
        self._on_start_polling()

    async def stop(self) -> None:
        pass


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.sent_media: list[dict] = []
        self.get_me_calls = 0

    async def get_me(self):
        self.get_me_calls += 1
        return SimpleNamespace(id=999, username="nanobot_test")

    async def set_my_commands(self, commands) -> None:
        self.commands = commands

    async def send_message(self, **kwargs):
        self.sent_messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.sent_messages))

    async def send_photo(self, **kwargs) -> None:
        self.sent_media.append({"kind": "photo", **kwargs})

    async def send_video(self, **kwargs) -> None:
        self.sent_media.append({"kind": "video", **kwargs})

    async def send_voice(self, **kwargs) -> None:
        self.sent_media.append({"kind": "voice", **kwargs})

    async def send_audio(self, **kwargs) -> None:
        self.sent_media.append({"kind": "audio", **kwargs})

    async def send_document(self, **kwargs) -> None:
        self.sent_media.append({"kind": "document", **kwargs})

    async def send_chat_action(self, **kwargs) -> None:
        pass

    async def send_poll(self, **kwargs):
        self.sent_polls = getattr(self, "sent_polls", [])
        self.sent_polls.append(kwargs)
        return SimpleNamespace(
            poll=SimpleNamespace(id=f"poll_{len(self.sent_polls)}"),
            message_id=len(self.sent_polls),
        )

    async def edit_message_text(self, **kwargs):
        self.edited_messages = getattr(self, "edited_messages", [])
        self.edited_messages.append(kwargs.get("message_id"))
        return SimpleNamespace(message_id=kwargs.get("message_id"))

    async def do_api_request(self, method: str, **kwargs):
        """Fake Bot API 10.x method (sendRichMessage, editMessageText rich, etc.)."""
        self.api_calls = getattr(self, "api_calls", [])
        self.api_calls.append({"method": method, **kwargs})
        if method == "sendRichMessage":
            return SimpleNamespace(message_id=len(self.api_calls))
        return SimpleNamespace(message_id=len(self.api_calls))

    async def get_file(self, file_id: str):
        """Return a fake file that 'downloads' to a path (for reply-to-media tests)."""
        async def _fake_download(path) -> None:
            pass
        return SimpleNamespace(download_to_drive=_fake_download)


class _FakeApp:
    def __init__(self, on_start_polling) -> None:
        self.bot = _FakeBot()
        self.updater = _FakeUpdater(on_start_polling)
        self.handlers = []
        self.error_handlers = []

    def add_error_handler(self, handler) -> None:
        self.error_handlers.append(handler)

    def add_handler(self, handler) -> None:
        self.handlers.append(handler)

    async def initialize(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _FakeBuilder:
    def __init__(self, app: _FakeApp) -> None:
        self.app = app
        self.token_value = None
        self.request_value = None
        self.get_updates_request_value = None

    def token(self, token: str):
        self.token_value = token
        return self

    def request(self, request):
        self.request_value = request
        return self

    def get_updates_request(self, request):
        self.get_updates_request_value = request
        return self

    def proxy(self, _proxy):
        raise AssertionError("builder.proxy should not be called when request is set")

    def get_updates_proxy(self, _proxy):
        raise AssertionError("builder.get_updates_proxy should not be called when request is set")

    def build(self):
        return self.app


def _make_telegram_update(
    *,
    chat_type: str = "group",
    text: str | None = None,
    caption: str | None = None,
    entities=None,
    caption_entities=None,
    reply_to_message=None,
    location=None,
):
    user = SimpleNamespace(id=12345, username="alice", first_name="Alice")
    message = SimpleNamespace(
        chat=SimpleNamespace(type=chat_type, is_forum=False),
        chat_id=-100123,
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        reply_to_message=reply_to_message,
        photo=None,
        voice=None,
        audio=None,
        document=None,
        location=location,
        media_group_id=None,
        message_thread_id=None,
        message_id=1,
    )
    return SimpleNamespace(message=message, effective_user=user)


def _assert_code_blocks_render_balanced(chunks: list[str]) -> None:
    for chunk in chunks:
        html = _markdown_to_telegram_html(chunk)
        assert html.count("<pre><code>") == html.count("</code></pre>")


def test_split_telegram_markdown_inside_code_block_moves_before_fence() -> None:
    content = "Intro paragraph.\n```python\nprint('a')\nprint('b')\n```\nDone"

    chunks = _split_telegram_markdown(content, max_len=35)

    assert chunks[0] == "Intro paragraph.\n"
    assert chunks[1].startswith("```python\nprint('a')")
    _assert_code_blocks_render_balanced(chunks)


def test_split_telegram_markdown_long_code_block_closes_and_reopens() -> None:
    content = "```python\n" + ("print('line one')\n" * 6) + "```\nDone"

    chunks = _split_telegram_markdown(content, max_len=60)

    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert chunks[0].startswith("```python\n")
    assert chunks[0].endswith("\n```")
    assert chunks[1].startswith("```python\n")
    _assert_code_blocks_render_balanced(chunks)


def test_split_telegram_markdown_multiple_code_blocks() -> None:
    content = (
        "First\n"
        "```js\n"
        "one();\n"
        "```\n"
        "Middle paragraph here\n"
        "```py\n"
        "two()\n"
        "three()\n"
        "```\n"
        "End"
    )

    chunks = _split_telegram_markdown(content, max_len=55)

    assert chunks[0].endswith("Middle paragraph here\n")
    assert chunks[1].startswith("```py\n")
    _assert_code_blocks_render_balanced(chunks)


def test_split_telegram_markdown_leading_whitespace_before_fence() -> None:
    content = "\n```python\n" + ("print('line one')\n" * 6) + "```\nDone"

    chunks = _split_telegram_markdown(content, max_len=60)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)
    assert chunks[0].startswith("```python\n")
    _assert_code_blocks_render_balanced(chunks)


def test_split_telegram_markdown_long_single_line_code_body() -> None:
    """Long fence bodies with no interior newlines must still advance."""
    body = "a" * 4500
    content = f"```\n{body}\n```"

    chunks = _split_telegram_markdown(content, TELEGRAM_MAX_MESSAGE_LEN)

    assert len(chunks) > 1
    assert all(len(chunk) <= TELEGRAM_MAX_MESSAGE_LEN for chunk in chunks)
    assert chunks[0].startswith("```\n")
    assert chunks[0].endswith("\n```")
    assert chunks[1].startswith("```\n")
    reassembled = []
    for chunk in chunks:
        part = chunk.split("\n", 1)[1]
        if part.endswith("\n```"):
            part = part[:-4]
        elif part.endswith("```"):
            part = part[:-3]
        reassembled.append(part)
    assert "".join(reassembled) == body
    _assert_code_blocks_render_balanced(chunks)


def test_split_telegram_markdown_tiny_limit_hard_cuts_fence_prefix() -> None:
    """Adaptive HTML limits can shrink max_len to the fence+closer size."""
    body = "a" * 100
    content = f"```\n{body}"

    chunks = _split_telegram_markdown(content, max_len=8)

    assert chunks
    assert all(len(chunk) <= 8 for chunk in chunks)
    assert "".join(chunks).replace("```", "").replace("\n", "") == body


def test_split_telegram_markdown_tiny_limit_with_early_body_newline() -> None:
    body = "a" * 100
    content = f"```\na\n{body}"

    chunks = _split_telegram_markdown(content, max_len=8)

    assert chunks
    assert all(len(chunk) <= 8 for chunk in chunks)
    plain = "".join(chunks).replace("```", "")
    assert "a" in plain
    assert plain.count("a") >= 100


def test_split_telegram_markdown_leading_space_in_fence_body() -> None:
    body = "a" * 4500
    content = f"```\n {body}"

    chunks = _split_telegram_markdown(content, TELEGRAM_MAX_MESSAGE_LEN)

    assert chunks
    assert all(len(chunk) <= TELEGRAM_MAX_MESSAGE_LEN for chunk in chunks)
    plain = "".join(chunks).replace("```", "").replace("\n", "")
    assert plain.count("a") == 4500




@pytest.mark.asyncio
async def test_start_creates_separate_pools_with_proxy(monkeypatch) -> None:
    _FakeHTTPXRequest.clear()
    config = TelegramConfig(
        enabled=True,
        token="123:abc",
        allow_from=["*"],
        proxy="http://127.0.0.1:7890",
    )
    bus = MessageBus()
    channel = TelegramChannel(config, bus)
    app = _FakeApp(lambda: setattr(channel, "_running", False))
    builder = _FakeBuilder(app)

    monkeypatch.setattr("nanobot.channels.telegram.runtime.HTTPXRequest", _FakeHTTPXRequest)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.Application",
        SimpleNamespace(builder=lambda: builder),
    )

    await channel.start()

    assert len(_FakeHTTPXRequest.instances) == 2
    api_req, poll_req = _FakeHTTPXRequest.instances
    assert api_req.kwargs["proxy"] == config.proxy
    assert poll_req.kwargs["proxy"] == config.proxy
    assert api_req.kwargs["connection_pool_size"] == 32
    assert poll_req.kwargs["connection_pool_size"] == 4
    assert builder.request_value is api_req
    assert builder.get_updates_request_value is poll_req
    assert callable(app.updater.start_polling_kwargs["error_callback"])
    assert any(cmd.command == "status" for cmd in app.bot.commands)
    assert any(cmd.command == "history" for cmd in app.bot.commands)
    assert any(cmd.command == "skill" for cmd in app.bot.commands)
    assert any(cmd.command == "dream" for cmd in app.bot.commands)
    assert any(cmd.command == "dream_log" for cmd in app.bot.commands)
    assert any(cmd.command == "dream_restore" for cmd in app.bot.commands)
    assert any(cmd.command == "dream_prompt" for cmd in app.bot.commands)


@pytest.mark.asyncio
async def test_start_respects_custom_pool_config(monkeypatch) -> None:
    _FakeHTTPXRequest.clear()
    config = TelegramConfig(
        enabled=True,
        token="123:abc",
        allow_from=["*"],
        connection_pool_size=32,
        pool_timeout=10.0,
    )
    bus = MessageBus()
    channel = TelegramChannel(config, bus)
    app = _FakeApp(lambda: setattr(channel, "_running", False))
    builder = _FakeBuilder(app)

    monkeypatch.setattr("nanobot.channels.telegram.runtime.HTTPXRequest", _FakeHTTPXRequest)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.Application",
        SimpleNamespace(builder=lambda: builder),
    )

    await channel.start()

    api_req = _FakeHTTPXRequest.instances[0]
    poll_req = _FakeHTTPXRequest.instances[1]
    assert api_req.kwargs["connection_pool_size"] == 32
    assert api_req.kwargs["pool_timeout"] == 10.0
    assert poll_req.kwargs["pool_timeout"] == 10.0


def test_webhook_config_requires_https_url_and_secret() -> None:
    with pytest.raises(ValueError, match="webhook_url is required"):
        TelegramConfig(enabled=True, token="123:abc", mode="webhook")

    with pytest.raises(ValueError, match="public HTTPS URL"):
        TelegramConfig(
            enabled=True,
            token="123:abc",
            mode="webhook",
            webhook_url="http://example.com/telegram",
            webhook_secret_token="secret",
        )

    with pytest.raises(ValueError, match="webhook_secret_token is required"):
        TelegramConfig(
            enabled=True,
            token="123:abc",
            mode="webhook",
            webhook_url="https://example.com/telegram",
        )


@pytest.mark.asyncio
async def test_start_webhook_mode(monkeypatch) -> None:
    _FakeHTTPXRequest.clear()
    config = TelegramConfig(
        enabled=True,
        token="123:abc",
        allow_from=["*"],
        mode="webhook",
        webhook_url="https://example.com/telegram",
        webhook_listen_host="127.0.0.1",
        webhook_listen_port=8081,
        webhook_path="/telegram",
        webhook_secret_token="secret-token",
        webhook_max_connections=1,
    )
    bus = MessageBus()
    channel = TelegramChannel(config, bus)
    app = _FakeApp(lambda: setattr(channel, "_running", False))
    builder = _FakeBuilder(app)

    monkeypatch.setattr("nanobot.channels.telegram.runtime.HTTPXRequest", _FakeHTTPXRequest)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.Application",
        SimpleNamespace(builder=lambda: builder),
    )

    await channel.start()

    assert app.updater.start_polling_kwargs is None
    assert app.updater.start_webhook_kwargs == {
        "listen": "127.0.0.1",
        "port": 8081,
        "url_path": "telegram",
        "webhook_url": "https://example.com/telegram",
        "allowed_updates": ["message", "poll_answer"],
        "drop_pending_updates": False,
        "secret_token": "secret-token",
        "max_connections": 1,
    }


@pytest.mark.asyncio
async def test_running_message_handler_reorders_same_session_updates() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    seen: list[int] = []

    async def fake_process(update, context) -> None:
        seen.append(update.message.message_id)

    channel._process_message_update = fake_process
    channel._running = True

    first = _make_telegram_update(text="first")
    first.update_id = 100
    first.message.message_id = 1
    second = _make_telegram_update(text="second")
    second.update_id = 101
    second.message.message_id = 2

    await channel._on_message(second, None)
    await channel._on_message(first, None)
    await asyncio.sleep(0.3)
    channel._running = False

    assert seen == [1, 2]


@pytest.mark.asyncio
async def test_send_text_retries_on_timeout() -> None:
    """_send_text retries on TimedOut before succeeding."""
    from telegram.error import TimedOut

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    call_count = 0
    original_send = channel._app.bot.send_message

    async def flaky_send(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise TimedOut()
        return await original_send(**kwargs)

    channel._app.bot.send_message = flaky_send

    import nanobot.channels.telegram.runtime as tg_mod
    orig_delay = tg_mod._SEND_RETRY_BASE_DELAY
    tg_mod._SEND_RETRY_BASE_DELAY = 0.01
    try:
        await channel._send_text(123, "hello", None, {})
    finally:
        tg_mod._SEND_RETRY_BASE_DELAY = orig_delay

    assert call_count == 3
    assert len(channel._app.bot.sent_messages) == 1


@pytest.mark.asyncio
async def test_send_text_gives_up_after_max_retries() -> None:
    """_send_text raises TimedOut after exhausting all retries."""
    from telegram.error import TimedOut

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    async def always_timeout(**kwargs):
        raise TimedOut()

    channel._app.bot.send_message = always_timeout

    import nanobot.channels.telegram.runtime as tg_mod
    orig_delay = tg_mod._SEND_RETRY_BASE_DELAY
    tg_mod._SEND_RETRY_BASE_DELAY = 0.01
    try:
        with pytest.raises(TimedOut):
            await channel._send_text(123, "hello", None, {})
    finally:
        tg_mod._SEND_RETRY_BASE_DELAY = orig_delay

    assert channel._app.bot.sent_messages == []


@pytest.mark.asyncio
async def test_send_rich_capability_error_latches_and_falls_back() -> None:
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], rich_messages=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock(side_effect=BadRequest("Method not found"))

    await channel.send(OutboundMessage(channel="telegram", chat_id="123", content="**hello**"))

    assert channel._rich_send_disabled is True
    channel._app.bot.do_api_request.assert_awaited_once()
    assert len(channel._app.bot.sent_messages) == 1
    assert channel._app.bot.sent_messages[0]["text"]


@pytest.mark.asyncio
async def test_send_rich_bad_request_does_not_latch_capability() -> None:
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], rich_messages=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock(
        side_effect=BadRequest("Bad Request: message to reply not found")
    )

    await channel.send(OutboundMessage(channel="telegram", chat_id="123", content="**hello**"))

    assert channel._rich_send_disabled is False
    channel._app.bot.do_api_request.assert_awaited_once()
    assert len(channel._app.bot.sent_messages) == 1


@pytest.mark.asyncio
async def test_rich_messages_default_skips_send_rich_message() -> None:
    """By default, sendRichMessage should not be called."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send(OutboundMessage(channel="telegram", chat_id="123", content="**hello**"))

    channel._app.bot.do_api_request.assert_not_called()
    assert len(channel._app.bot.sent_messages) == 1
    assert channel._app.bot.sent_messages[0]["text"]


@pytest.mark.asyncio
async def test_on_error_logs_network_issues_as_warning(monkeypatch) -> None:
    from telegram.error import NetworkError

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    recorded: list[tuple[str, str]] = []

    monkeypatch.setattr(
        channel.logger,
        "warning",
        lambda message, error: recorded.append(("warning", message.format(error))),
    )
    monkeypatch.setattr(
        channel.logger,
        "error",
        lambda message, error: recorded.append(("error", message.format(error))),
    )

    await channel._on_error(object(), SimpleNamespace(error=NetworkError("proxy disconnected")))

    assert recorded == [("warning", "network issue: proxy disconnected")]


@pytest.mark.asyncio
async def test_on_error_summarizes_empty_network_error(monkeypatch) -> None:
    from telegram.error import NetworkError

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    recorded: list[tuple[str, str]] = []

    monkeypatch.setattr(
        channel.logger,
        "warning",
        lambda message, error: recorded.append(("warning", message.format(error))),
    )

    await channel._on_error(object(), SimpleNamespace(error=NetworkError("")))

    assert recorded == [("warning", "network issue: NetworkError")]


@pytest.mark.asyncio
async def test_on_error_keeps_non_network_exceptions_as_error(monkeypatch) -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    recorded: list[tuple[str, str]] = []

    monkeypatch.setattr(
        channel.logger,
        "warning",
        lambda message, error: recorded.append(("warning", message.format(error))),
    )
    monkeypatch.setattr(
        channel.logger,
        "error",
        lambda message, error: recorded.append(("error", message.format(error))),
    )

    await channel._on_error(object(), SimpleNamespace(error=RuntimeError("boom")))

    assert recorded == [("error", "error: boom")]


@pytest.mark.asyncio
async def test_send_delta_stream_end_raises_and_keeps_buffer_on_failure() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock(side_effect=RuntimeError("boom"))
    channel._stream_bufs["123"] = _StreamBuf(text="hello", message_id=7, last_edit=0.0)

    with pytest.raises(RuntimeError, match="boom"):
        await channel.send_delta("123", "", stream_end=True)

    assert "123" in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_stream_end_treats_not_modified_as_success() -> None:
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock(side_effect=BadRequest("Message is not modified"))
    channel._stream_bufs["123"] = _StreamBuf(text="hello", message_id=7, last_edit=0.0, stream_id="s:0")

    await channel.send_delta("123", "", stream_id="s:0", stream_end=True)

    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_stream_end_does_not_fallback_on_network_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TimedOut during HTML edit should propagate, never fall back to plain text."""
    from telegram.error import TimedOut

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    monkeypatch.setattr("nanobot.channels.telegram.runtime._SEND_RETRY_BASE_DELAY", 0)
    # _call_with_retry retries TimedOut up to 3 times, so the mock will be called
    # multiple times – but all calls must be with parse_mode="HTML" (no plain fallback).
    channel._app.bot.edit_message_text = AsyncMock(side_effect=TimedOut("network timeout"))
    channel._stream_bufs["123"] = _StreamBuf(text="hello", message_id=7, last_edit=0.0)

    with pytest.raises(TimedOut, match="network timeout"):
        await channel.send_delta("123", "", stream_end=True)

    # Every call to edit_message_text must have used parse_mode="HTML" —
    # no plain-text fallback call should have been made.
    for call in channel._app.bot.edit_message_text.call_args_list:
        assert call.kwargs.get("parse_mode") == "HTML"
    assert channel._app.bot.edit_message_text.await_count == 3
    # Buffer should still be present (not cleaned up on error)
    assert "123" in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_stream_end_does_not_fallback_on_network_error() -> None:
    """NetworkError during HTML edit should propagate, never fall back to plain text."""
    from telegram.error import NetworkError

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock(side_effect=NetworkError("connection reset"))
    channel._stream_bufs["123"] = _StreamBuf(text="hello", message_id=7, last_edit=0.0)

    with pytest.raises(NetworkError, match="connection reset"):
        await channel.send_delta("123", "", stream_end=True)

    # Every call to edit_message_text must have used parse_mode="HTML" —
    # no plain-text fallback call should have been made.
    for call in channel._app.bot.edit_message_text.call_args_list:
        assert call.kwargs.get("parse_mode") == "HTML"
    # Buffer should still be present (not cleaned up on error)
    assert "123" in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_stream_end_falls_back_on_bad_request() -> None:
    """BadRequest (HTML parse error) should still trigger plain-text fallback."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    # First call (HTML) raises BadRequest, second call (plain) succeeds
    channel._app.bot.edit_message_text = AsyncMock(
        side_effect=[BadRequest("Can't parse entities"), None]
    )
    channel._stream_bufs["123"] = _StreamBuf(text="hello <bad>", message_id=7, last_edit=0.0)

    await channel.send_delta("123", "", stream_end=True)

    # edit_message_text should have been called twice: once for HTML, once for plain fallback
    assert channel._app.bot.edit_message_text.call_count == 2
    # Second call should not use parse_mode="HTML"
    second_call_kwargs = channel._app.bot.edit_message_text.call_args_list[1].kwargs
    assert "parse_mode" not in second_call_kwargs or second_call_kwargs.get("parse_mode") is None
    # Buffer should be cleaned up on success
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_stream_end_splits_oversized_reply() -> None:
    """Final streamed reply exceeding Telegram limit is split into chunks.

    The fix converts markdown to HTML first, then splits by 4096 (actual Telegram
    limit), ensuring the edited message always fits within Telegram's constraint.
    Previously, the code split by 4000 (TELEGRAM_MAX_MESSAGE_LEN) before HTML
    conversion, which could still overflow when HTML tags were added.
    """
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock()
    channel._app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=99))

    oversized = "x" * (4000 + 500)
    channel._stream_bufs["123"] = _StreamBuf(text=oversized, message_id=7, last_edit=0.0)

    await channel.send_delta("123", "", stream_end=True)

    channel._app.bot.edit_message_text.assert_called_once()
    edit_text = channel._app.bot.edit_message_text.call_args.kwargs.get("text", "")
    assert len(edit_text) <= 4096, f"edit_text length {len(edit_text)} exceeds Telegram's 4096 limit"

    channel._app.bot.send_message.assert_called_once()
    send_text = channel._app.bot.send_message.call_args.kwargs.get("text", "")
    assert len(send_text) <= 4096
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_stream_end_html_expansion_does_not_overflow() -> None:
    """Markdown that expands when converted to HTML is still split correctly.

    This is the actual bug from issue #3315: markdown like **bold** expands to
    <b>bold</b>, adding ~33% characters. A 3600-char message with heavy markdown
    could become 4800+ chars after HTML conversion, exceeding 4096 limit.
    The fix converts to HTML first, THEN splits by 4096.
    """
    from nanobot.channels.telegram.runtime import _markdown_to_telegram_html

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock()
    channel._app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=99))

    markdown_text = "**bold** " * 400  # 3600 chars raw, expands ~33% to 4800 HTML
    raw_len = len(markdown_text)
    html_len = len(_markdown_to_telegram_html(markdown_text))
    assert html_len > 4096, f"Test precondition failed: HTML should exceed 4096 (was {html_len})"

    channel._stream_bufs["123"] = _StreamBuf(text=markdown_text, message_id=7, last_edit=0.0)

    await channel.send_delta("123", "", stream_end=True)

    channel._app.bot.edit_message_text.assert_called_once()
    edit_text = channel._app.bot.edit_message_text.call_args.kwargs.get("text", "")
    assert len(edit_text) <= 4096, (
        f"HTML text length {len(edit_text)} exceeds Telegram's 4096 limit. "
        f"Raw was {raw_len}, HTML was {html_len}."
    )

    channel._app.bot.send_message.assert_called_once()
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_stream_end_splits_long_code_block_before_html_rendering() -> None:
    """Final streamed replies must not split Telegram HTML inside <pre><code>."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock()
    channel._app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=99))

    raw_text = "```python\n" + ("print(\"line\")\n" * 450) + "```\nDone"
    channel._stream_bufs["123"] = _StreamBuf(text=raw_text, message_id=7, last_edit=0.0)

    await channel.send_delta("123", "", stream_end=True)

    html_chunks = [
        channel._app.bot.edit_message_text.call_args.kwargs.get("text", ""),
        *[
            call.kwargs.get("text", "")
            for call in channel._app.bot.send_message.call_args_list
        ],
    ]
    assert len(html_chunks) > 1
    for html in html_chunks:
        assert len(html) <= 4096
        assert html.count("<pre><code>") == html.count("</code></pre>")
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_new_stream_id_replaces_stale_buffer() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._stream_bufs["123"] = _StreamBuf(
        text="hello",
        message_id=7,
        last_edit=0.0,
        stream_id="old:0",
    )

    await channel.send_delta("123", "world", stream_id="new:0")

    buf = channel._stream_bufs["123"]
    assert buf.text == "world"
    assert buf.stream_id == "new:0"
    assert buf.message_id == 1


@pytest.mark.asyncio
async def test_send_delta_incremental_edit_treats_not_modified_as_success() -> None:
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._stream_bufs["123"] = _StreamBuf(text="hello", message_id=7, last_edit=0.0, stream_id="s:0")
    channel._app.bot.edit_message_text = AsyncMock(side_effect=BadRequest("Message is not modified"))

    await channel.send_delta("123", "", stream_id="s:0")

    assert channel._stream_bufs["123"].last_edit > 0.0


@pytest.mark.asyncio
async def test_send_delta_incremental_edit_splits_oversized_buffer() -> None:
    """Mid-stream overflow: once buf.text exceeds Telegram's limit, split into
    chunks, edit the current message with the first chunk, and re-anchor the
    buffer to a new message for the tail so further deltas keep streaming."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock()
    channel._app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=99))

    first_chunk = f"**{'x' * 3900}**\n"
    tail = "**tail** " * 50
    oversized = first_chunk + tail
    channel._stream_bufs["123"] = _StreamBuf(
        text=oversized, message_id=7, last_edit=0.0, stream_id="s:0"
    )

    await channel.send_delta("123", "y", stream_id="s:0")

    channel._app.bot.edit_message_text.assert_called_once()
    edit_kwargs = channel._app.bot.edit_message_text.call_args.kwargs
    assert edit_kwargs["text"] == _markdown_to_telegram_html(first_chunk.rstrip())
    assert edit_kwargs["parse_mode"] == "HTML"

    channel._app.bot.send_message.assert_called_once()
    send_kwargs = channel._app.bot.send_message.call_args.kwargs
    assert send_kwargs["parse_mode"] == "HTML"
    buf = channel._stream_bufs["123"]
    assert buf.message_id == 99
    assert buf.text == tail + "y"
    assert send_kwargs["text"] == _markdown_to_telegram_html(buf.text)
    assert buf.last_edit > 0.0


@pytest.mark.asyncio
async def test_send_delta_incremental_html_expansion_does_not_overflow() -> None:
    """Mid-stream HTML chunks stay within Telegram's rendered payload limit."""
    from nanobot.channels.telegram.runtime import TELEGRAM_HTML_MAX_LEN

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock()
    channel._app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=99))

    oversized = "**bold** " * 501
    assert len(_markdown_to_telegram_html(oversized)) > TELEGRAM_HTML_MAX_LEN
    channel._stream_bufs["123"] = _StreamBuf(
        text=oversized, message_id=7, last_edit=0.0, stream_id="s:0"
    )

    await channel.send_delta("123", "y", stream_id="s:0")

    payloads = [
        channel._app.bot.edit_message_text.call_args.kwargs,
        *[call.kwargs for call in channel._app.bot.send_message.call_args_list],
    ]
    assert len(payloads) > 1
    assert all(payload["parse_mode"] == "HTML" for payload in payloads)
    assert all(len(payload["text"]) <= TELEGRAM_HTML_MAX_LEN for payload in payloads)

    buf = channel._stream_bufs["123"]
    assert payloads[-1]["text"] == _markdown_to_telegram_html(buf.text)
    assert "<b>" not in buf.text


@pytest.mark.asyncio
async def test_send_delta_incremental_html_parse_failure_falls_back_to_plain() -> None:
    """Telegram HTML rejections retry overflow chunks as plain text."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.edit_message_text = AsyncMock(
        side_effect=[BadRequest("Can't parse entities"), None]
    )
    channel._app.bot.send_message = AsyncMock(
        side_effect=[BadRequest("Can't parse entities"), SimpleNamespace(message_id=99)]
    )

    first_chunk = f"**{'x' * 3900}**\n"
    tail = "**tail** " * 50
    channel._stream_bufs["123"] = _StreamBuf(
        text=first_chunk + tail, message_id=7, last_edit=0.0, stream_id="s:0"
    )

    await channel.send_delta("123", "y", stream_id="s:0")

    edit_calls = channel._app.bot.edit_message_text.call_args_list
    assert edit_calls[0].kwargs["parse_mode"] == "HTML"
    assert edit_calls[1].kwargs["text"] == first_chunk.rstrip()
    assert "parse_mode" not in edit_calls[1].kwargs

    send_calls = channel._app.bot.send_message.call_args_list
    assert send_calls[0].kwargs["parse_mode"] == "HTML"
    assert send_calls[1].kwargs["text"] == tail + "y"
    assert "parse_mode" not in send_calls[1].kwargs
    assert channel._stream_bufs["123"].text == tail + "y"


@pytest.mark.asyncio
async def test_send_delta_initial_send_keeps_message_in_thread() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    await channel.send_delta(
        "123",
        "hello",
        {"message_thread_id": 42},
        stream_id="s:0",
    )

    assert channel._app.bot.sent_messages[0]["message_thread_id"] == 42


def test_derive_topic_session_key_uses_thread_id() -> None:
    message = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        chat_id=-100123,
        message_thread_id=42,
    )

    assert TelegramChannel._derive_topic_session_key(message) == "telegram:-100123:topic:42"


def test_derive_topic_session_key_private_dm_thread() -> None:
    """Private DM threads (Telegram Threaded Mode) must get their own session key."""
    message = SimpleNamespace(
        chat=SimpleNamespace(type="private"),
        chat_id=999,
        message_thread_id=7,
    )
    assert TelegramChannel._derive_topic_session_key(message) == "telegram:999:topic:7"


def test_derive_topic_session_key_none_without_thread() -> None:
    """No thread id → no topic session key, regardless of chat type."""
    for chat_type in ("private", "supergroup", "group"):
        message = SimpleNamespace(
            chat=SimpleNamespace(type=chat_type),
            chat_id=123,
            message_thread_id=None,
        )
        assert TelegramChannel._derive_topic_session_key(message) is None


def test_get_extension_falls_back_to_original_filename() -> None:
    channel = TelegramChannel(TelegramConfig(), MessageBus())

    assert channel._get_extension("file", None, "report.pdf") == ".pdf"
    assert channel._get_extension("file", None, "archive.tar.gz") == ".tar.gz"


def test_telegram_group_policy_defaults_to_mention() -> None:
    assert TelegramConfig().group_policy == "mention"


def test_is_allowed_accepts_legacy_telegram_id_username_formats() -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["12345", "alice", "67890|bob"]), MessageBus())

    assert channel.is_allowed("12345|carol") is True
    assert channel.is_allowed("99999|alice") is True
    assert channel.is_allowed("67890|bob") is True


def test_is_allowed_rejects_invalid_legacy_telegram_sender_shapes() -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["alice"]), MessageBus())

    assert channel.is_allowed("attacker|alice|extra") is False
    assert channel.is_allowed("not-a-number|alice") is False


@pytest.mark.asyncio
async def test_send_progress_keeps_message_in_topic() -> None:
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="hello",
            event=ProgressEvent(content="hello"),
            metadata={"message_thread_id": 42},
        )
    )

    assert channel._app.bot.sent_messages[0]["message_thread_id"] == 42


@pytest.mark.asyncio
async def test_send_reply_infers_topic_from_message_id_cache() -> None:
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], reply_to_message=True)
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp(lambda: None)
    channel._message_threads[("123", 10)] = 42

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="hello",
            metadata={"message_id": 10},
        )
    )

    assert channel._app.bot.sent_messages[0]["message_thread_id"] == 42
    assert channel._app.bot.sent_messages[0]["reply_parameters"].message_id == 10


@pytest.mark.asyncio
async def test_reply_anchor_first_send_quotes_second_does_not() -> None:
    """T-002: el primer send de un chat quotea; los siguientes no (default reply_to_message=false)."""
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="first",
            metadata={"message_id": 1},
        )
    )
    assert channel._app.bot.sent_messages[0]["reply_parameters"].message_id == 1

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="second",
            metadata={"message_id": 2},
        )
    )
    assert channel._app.bot.sent_messages[1].get("reply_parameters") is None


@pytest.mark.asyncio
async def test_reply_opt_in_always_quotes() -> None:
    """T-002: reply_to_message=True conserva el comportamiento legacy (siempre quote)."""
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], reply_to_message=True)
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="456",
            content="first",
            metadata={"message_id": 1},
        )
    )
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="456",
            content="second",
            metadata={"message_id": 2},
        )
    )

    assert channel._app.bot.sent_messages[0]["reply_parameters"].message_id == 1
    assert channel._app.bot.sent_messages[1]["reply_parameters"].message_id == 2


@pytest.mark.asyncio
async def test_reply_no_message_id_no_quote() -> None:
    """T-002: sin message_id no hay quote y el ancla no se consume."""
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="789",
            content="no message_id",
        )
    )
    assert channel._app.bot.sent_messages[0].get("reply_parameters") is None

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="789",
            content="with message_id",
            metadata={"message_id": 3},
        )
    )
    assert channel._app.bot.sent_messages[1]["reply_parameters"].message_id == 3


@pytest.mark.asyncio
async def test_reply_reset_on_new_command() -> None:
    """T-006: /new reinicia el ancla; el siguiente mensaje vuelve a quotea."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle

    # Primer mensaje quotea (ancla consumida); el segundo no.
    await channel.send(
        OutboundMessage(
            channel="telegram", chat_id="-100123",
            content="first", metadata={"message_id": 1},
        )
    )
    await channel.send(
        OutboundMessage(
            channel="telegram", chat_id="-100123",
            content="second", metadata={"message_id": 2},
        )
    )
    assert channel._app.bot.sent_messages[0]["reply_parameters"].message_id == 1
    assert channel._app.bot.sent_messages[1].get("reply_parameters") is None

    # /new resetea el ancla del chat.
    await channel._process_forward_command(_make_telegram_update(text="/new"), None)
    assert handled[0]["content"] == "/new"

    # El siguiente mensaje vuelve a quotea.
    await channel.send(
        OutboundMessage(
            channel="telegram", chat_id="-100123",
            content="third", metadata={"message_id": 3},
        )
    )
    assert channel._app.bot.sent_messages[2]["reply_parameters"].message_id == 3


@pytest.mark.asyncio
async def test_reply_reset_on_start_command() -> None:
    """T-006: /start reinicia el ancla; el siguiente mensaje vuelve a quotea."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    # Primer mensaje quotea (ancla consumida); el segundo no.
    await channel.send(
        OutboundMessage(
            channel="telegram", chat_id="-100123",
            content="first", metadata={"message_id": 1},
        )
    )
    await channel.send(
        OutboundMessage(
            channel="telegram", chat_id="-100123",
            content="second", metadata={"message_id": 2},
        )
    )
    assert channel._app.bot.sent_messages[0]["reply_parameters"].message_id == 1
    assert channel._app.bot.sent_messages[1].get("reply_parameters") is None

    # /start resetea el ancla del chat.
    update = _make_telegram_update(text="/start", chat_type="private")
    update.message.reply_text = AsyncMock()
    await channel._on_start(update, None)
    update.message.reply_text.assert_awaited_once()

    # El siguiente mensaje vuelve a quotea.
    await channel.send(
        OutboundMessage(
            channel="telegram", chat_id="-100123",
            content="third", metadata={"message_id": 3},
        )
    )
    assert channel._app.bot.sent_messages[2]["reply_parameters"].message_id == 3


@pytest.mark.asyncio
async def test_send_remote_media_url_after_security_validation(monkeypatch) -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    monkeypatch.setattr("nanobot.channels.telegram.runtime.validate_url_target", lambda url: (True, ""))

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            media=["https://example.com/cat.jpg"],
        )
    )

    assert channel._app.bot.sent_media == [
        {
            "kind": "photo",
            "chat_id": 123,
            "photo": "https://example.com/cat.jpg",
            "reply_parameters": None,
        }
    ]


@pytest.mark.asyncio
async def test_send_local_media_preserves_filename(tmp_path: Path) -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    attachment = tmp_path / "report.final.md"
    attachment.write_bytes(b"# Report\n")

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            media=[str(attachment)],
        )
    )

    assert channel._app.bot.sent_media == [
        {
            "kind": "document",
            "chat_id": 123,
            "document": b"# Report\n",
            "reply_parameters": None,
            "filename": "report.final.md",
        }
    ]


@pytest.mark.asyncio
async def test_send_blocks_unsafe_remote_media_url(monkeypatch) -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.validate_url_target",
        lambda url: (False, "Blocked: example.com resolves to private/internal address 127.0.0.1"),
    )

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            media=["http://example.com/internal.jpg"],
        )
    )

    assert channel._app.bot.sent_media == []
    assert channel._app.bot.sent_messages == [
        {
            "chat_id": 123,
            "text": "[Failed to send: internal.jpg]",
            "reply_parameters": None,
        }
    ]


@pytest.mark.asyncio
async def test_group_policy_mention_ignores_unmentioned_group_message() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="mention"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    await channel._on_message(_make_telegram_update(text="hello everyone"), None)

    assert handled == []
    assert channel._app.bot.get_me_calls == 1


@pytest.mark.asyncio
async def test_group_policy_mention_accepts_text_mention_and_caches_bot_identity() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="mention"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    mention = SimpleNamespace(type="mention", offset=0, length=13)
    await channel._on_message(_make_telegram_update(text="@nanobot_test hi", entities=[mention]), None)
    await channel._on_message(_make_telegram_update(text="@nanobot_test again", entities=[mention]), None)

    assert len(handled) == 2
    assert channel._app.bot.get_me_calls == 1


@pytest.mark.asyncio
async def test_group_policy_mention_accepts_caption_mention() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="mention"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    mention = SimpleNamespace(type="mention", offset=0, length=13)
    await channel._on_message(
        _make_telegram_update(caption="@nanobot_test photo", caption_entities=[mention]),
        None,
    )

    assert len(handled) == 1
    assert handled[0]["content"] == "@nanobot_test photo"


@pytest.mark.asyncio
async def test_group_policy_mention_accepts_reply_to_bot() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="mention"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    reply = SimpleNamespace(from_user=SimpleNamespace(id=999))
    await channel._on_message(_make_telegram_update(text="reply", reply_to_message=reply), None)

    assert len(handled) == 1


@pytest.mark.asyncio
async def test_group_policy_open_accepts_plain_group_message() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    await channel._on_message(_make_telegram_update(text="hello group"), None)

    assert len(handled) == 1
    assert channel._app.bot.get_me_calls == 0


@pytest.mark.asyncio
async def test_extract_reply_context_no_reply() -> None:
    """When there is no reply_to_message, _extract_reply_context returns None."""
    channel = TelegramChannel(TelegramConfig(enabled=True, token="123:abc"), MessageBus())
    message = SimpleNamespace(reply_to_message=None)
    assert await channel._extract_reply_context(message) is None


@pytest.mark.asyncio
async def test_extract_reply_context_with_text() -> None:
    """When reply has text, return prefixed string."""
    channel = TelegramChannel(TelegramConfig(enabled=True, token="123:abc"), MessageBus())
    channel._app = _FakeApp(lambda: None)
    reply = SimpleNamespace(text="Hello world", caption=None, from_user=SimpleNamespace(id=2, username="testuser", first_name="Test"))
    message = SimpleNamespace(reply_to_message=reply)
    assert await channel._extract_reply_context(message) == "[Reply to @testuser: Hello world]"


@pytest.mark.asyncio
async def test_extract_reply_context_with_caption_only() -> None:
    """When reply has only caption (no text), caption is used."""
    channel = TelegramChannel(TelegramConfig(enabled=True, token="123:abc"), MessageBus())
    channel._app = _FakeApp(lambda: None)
    reply = SimpleNamespace(text=None, caption="Photo caption", from_user=SimpleNamespace(id=2, username=None, first_name="Test"))
    message = SimpleNamespace(reply_to_message=reply)
    assert await channel._extract_reply_context(message) == "[Reply to Test: Photo caption]"


@pytest.mark.asyncio
async def test_extract_reply_context_truncation() -> None:
    """Reply text is truncated at TELEGRAM_REPLY_CONTEXT_MAX_LEN."""
    channel = TelegramChannel(TelegramConfig(enabled=True, token="123:abc"), MessageBus())
    channel._app = _FakeApp(lambda: None)
    long_text = "x" * (TELEGRAM_REPLY_CONTEXT_MAX_LEN + 100)
    reply = SimpleNamespace(text=long_text, caption=None, from_user=SimpleNamespace(id=2, username=None, first_name=None))
    message = SimpleNamespace(reply_to_message=reply)
    result = await channel._extract_reply_context(message)
    assert result is not None
    assert result.startswith("[Reply to: ")
    assert result.endswith("...]")
    assert len(result) == len("[Reply to: ]") + TELEGRAM_REPLY_CONTEXT_MAX_LEN + len("...")


@pytest.mark.asyncio
async def test_extract_reply_context_no_text_returns_none() -> None:
    """When reply has no text/caption, _extract_reply_context returns None (media handled separately)."""
    channel = TelegramChannel(TelegramConfig(enabled=True, token="123:abc"), MessageBus())
    reply = SimpleNamespace(text=None, caption=None)
    message = SimpleNamespace(reply_to_message=reply)
    assert await channel._extract_reply_context(message) is None


@pytest.mark.asyncio
async def test_on_message_includes_reply_context() -> None:
    """When user replies to a message, content passed to bus starts with reply context."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    reply = SimpleNamespace(text="Hello", message_id=2, from_user=SimpleNamespace(id=1))
    update = _make_telegram_update(text="translate this", reply_to_message=reply)
    await channel._on_message(update, None)

    assert len(handled) == 1
    assert handled[0]["content"].startswith("[Reply to: Hello]")
    assert "translate this" in handled[0]["content"]


@pytest.mark.asyncio
async def test_download_message_media_returns_path_when_download_succeeds(
    monkeypatch, tmp_path
) -> None:
    """_download_message_media returns (paths, content_parts) when bot.get_file and download succeed."""
    media_dir = tmp_path / "media" / "telegram"
    media_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.get_media_dir",
        lambda channel=None: media_dir if channel else tmp_path / "media",
    )

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.get_file = AsyncMock(
        return_value=SimpleNamespace(download_to_drive=AsyncMock(return_value=None))
    )

    msg = SimpleNamespace(
        photo=[SimpleNamespace(file_id="fid123", mime_type="image/jpeg")],
        voice=None,
        audio=None,
        document=None,
        video=None,
        video_note=None,
        animation=None,
    )
    paths, parts = await channel._download_message_media(msg)
    assert len(paths) == 1
    assert len(parts) == 1
    assert "fid123" in paths[0]
    assert "[image:" in parts[0]


@pytest.mark.asyncio
async def test_download_message_media_uses_file_unique_id_when_available(
    monkeypatch, tmp_path
) -> None:
    media_dir = tmp_path / "media" / "telegram"
    media_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.get_media_dir",
        lambda channel=None: media_dir if channel else tmp_path / "media",
    )

    downloaded: dict[str, str] = {}

    async def _download_to_drive(path: str) -> None:
        downloaded["path"] = path

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    app = _FakeApp(lambda: None)
    app.bot.get_file = AsyncMock(
        return_value=SimpleNamespace(download_to_drive=_download_to_drive)
    )
    channel._app = app

    msg = SimpleNamespace(
        photo=[
            SimpleNamespace(
                file_id="file-id-that-should-not-be-used",
                file_unique_id="stable-unique-id",
                mime_type="image/jpeg",
                file_name=None,
            )
        ],
        voice=None,
        audio=None,
        document=None,
        video=None,
        video_note=None,
        animation=None,
    )

    paths, parts = await channel._download_message_media(msg)

    assert downloaded["path"].endswith("stable-unique-id.jpg")
    assert paths == [str(media_dir / "stable-unique-id.jpg")]
    assert parts == [f"[image: {media_dir / 'stable-unique-id.jpg'}]"]


@pytest.mark.asyncio
async def test_on_message_attaches_reply_to_media_when_available(monkeypatch, tmp_path) -> None:
    """When user replies to a message with media, that media is downloaded and attached to the turn."""
    media_dir = tmp_path / "media" / "telegram"
    media_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.get_media_dir",
        lambda channel=None: media_dir if channel else tmp_path / "media",
    )

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    app = _FakeApp(lambda: None)
    app.bot.get_file = AsyncMock(
        return_value=SimpleNamespace(download_to_drive=AsyncMock(return_value=None))
    )
    channel._app = app
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    reply_with_photo = SimpleNamespace(
        text=None,
        caption=None,
        photo=[SimpleNamespace(file_id="reply_photo_fid", mime_type="image/jpeg")],
        document=None,
        voice=None,
        audio=None,
        video=None,
        video_note=None,
        animation=None,
    )
    update = _make_telegram_update(
        text="what is the image?",
        reply_to_message=reply_with_photo,
    )
    await channel._on_message(update, None)

    assert len(handled) == 1
    assert handled[0]["content"].startswith("[Reply to: [image:")
    assert "what is the image?" in handled[0]["content"]
    assert len(handled[0]["media"]) == 1
    assert "reply_photo_fid" in handled[0]["media"][0]


@pytest.mark.asyncio
async def test_on_message_reply_to_media_fallback_when_download_fails() -> None:
    """When reply has media but download fails, no media attached and no reply tag."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.get_file = None
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    reply_with_photo = SimpleNamespace(
        text=None,
        caption=None,
        photo=[SimpleNamespace(file_id="x", mime_type="image/jpeg")],
        document=None,
        voice=None,
        audio=None,
        video=None,
        video_note=None,
        animation=None,
    )
    update = _make_telegram_update(text="what is this?", reply_to_message=reply_with_photo)
    await channel._on_message(update, None)

    assert len(handled) == 1
    assert "what is this?" in handled[0]["content"]
    assert handled[0]["media"] == []


@pytest.mark.asyncio
async def test_on_message_reply_to_caption_and_media(monkeypatch, tmp_path) -> None:
    """When replying to a message with caption + photo, both text context and media are included."""
    media_dir = tmp_path / "media" / "telegram"
    media_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.get_media_dir",
        lambda channel=None: media_dir if channel else tmp_path / "media",
    )

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    app = _FakeApp(lambda: None)
    app.bot.get_file = AsyncMock(
        return_value=SimpleNamespace(download_to_drive=AsyncMock(return_value=None))
    )
    channel._app = app
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    reply_with_caption_and_photo = SimpleNamespace(
        text=None,
        caption="A cute cat",
        photo=[SimpleNamespace(file_id="cat_fid", mime_type="image/jpeg")],
        document=None,
        voice=None,
        audio=None,
        video=None,
        video_note=None,
        animation=None,
    )
    update = _make_telegram_update(
        text="what breed is this?",
        reply_to_message=reply_with_caption_and_photo,
    )
    await channel._on_message(update, None)

    assert len(handled) == 1
    assert "[Reply to: A cute cat]" in handled[0]["content"]
    assert "what breed is this?" in handled[0]["content"]
    assert len(handled[0]["media"]) == 1
    assert "cat_fid" in handled[0]["media"][0]


@pytest.mark.asyncio
async def test_forward_command_does_not_inject_reply_context() -> None:
    """Slash commands forwarded via _forward_command must not include reply context."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle

    reply = SimpleNamespace(text="some old message", message_id=2, from_user=SimpleNamespace(id=1))
    update = _make_telegram_update(text="/new", reply_to_message=reply)
    await channel._forward_command(update, None)

    assert len(handled) == 1
    assert handled[0]["content"] == "/new"


@pytest.mark.asyncio
async def test_forward_command_pairs_unauthorized_private_user(monkeypatch) -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["999"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    monkeypatch.setattr(
        "nanobot.channels.base.generate_code", lambda _ch, _sid: "ABCD-EFGH"
    )

    await channel._forward_command(_make_telegram_update(text="/new", chat_type="private"), None)

    assert len(channel._app.bot.sent_messages) == 1
    assert "ABCD-EFGH" in channel._app.bot.sent_messages[0]["text"]


@pytest.mark.asyncio
async def test_forward_command_preserves_dream_log_args_and_strips_bot_suffix() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    update = _make_telegram_update(text="/dream-log@nanobot_test deadbeef", reply_to_message=None)

    await channel._forward_command(update, None)

    assert len(handled) == 1
    assert handled[0]["content"] == "/dream-log deadbeef"


@pytest.mark.asyncio
async def test_forward_command_normalizes_telegram_safe_dream_aliases() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle
    update = _make_telegram_update(text="/dream_restore@nanobot_test deadbeef", reply_to_message=None)

    await channel._forward_command(update, None)

    assert len(handled) == 1
    assert handled[0]["content"] == "/dream-restore deadbeef"

    handled.clear()
    update = _make_telegram_update(text="/dream_prompt@nanobot_test init", reply_to_message=None)

    await channel._forward_command(update, None)

    assert len(handled) == 1
    assert handled[0]["content"] == "/dream-prompt init"


def test_telegram_bus_slash_command_regex_matches_agent_loop_commands() -> None:
    """Bus-routed slash commands must match the Telegram handler regex (see builtin router)."""
    pat = TelegramChannel.TELEGRAM_BUS_SLASH_COMMAND_RE
    assert pat.fullmatch("/history")
    assert pat.fullmatch("/history 5")
    assert pat.fullmatch("/goal ship the feature")
    assert pat.fullmatch("/trigger")
    assert pat.fullmatch("/trigger PR review")
    assert pat.fullmatch("/pairing list")
    assert pat.fullmatch("/model fast")
    assert pat.fullmatch("/skill")
    assert pat.fullmatch("/skill@nanobot_bot")
    assert pat.fullmatch("/new@nanobot_bot")
    assert pat.fullmatch("/goal@nanobot_bot refine objective")
    assert pat.fullmatch("/trigger@nanobot_bot CI summary")
    assert pat.fullmatch("/dream-log deadbeef") is None
    assert pat.fullmatch("/dream-restore deadbeef") is None
    assert pat.fullmatch("/dream-prompt init") is None


@pytest.mark.asyncio
async def test_on_help_includes_restart_command() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    update = _make_telegram_update(text="/help", chat_type="private")
    update.message.reply_text = AsyncMock()

    await channel._on_help(update, None)

    update.message.reply_text.assert_awaited_once()
    help_text = update.message.reply_text.await_args.args[0]
    assert "/restart" in help_text
    assert "/status" in help_text
    assert "/skill" in help_text
    assert "/dream" in help_text
    assert "/dream-log" in help_text
    assert "/dream-prompt" in help_text
    assert "/goal" in help_text
    assert "/trigger" in help_text
    assert "/pairing" in help_text
    assert "/model" in help_text
    assert "/dream-restore" in help_text


@pytest.mark.asyncio
async def test_on_start_sends_pairing_code_to_unauthorized_private_user(monkeypatch) -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["999"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    update = _make_telegram_update(text="/start", chat_type="private")
    update.message.reply_text = AsyncMock()
    monkeypatch.setattr(
        "nanobot.channels.base.generate_code", lambda _ch, _sid: "ABCD-EFGH"
    )

    await channel._on_start(update, None)

    update.message.reply_text.assert_not_awaited()
    assert len(channel._app.bot.sent_messages) == 1
    assert "ABCD-EFGH" in channel._app.bot.sent_messages[0]["text"]


@pytest.mark.asyncio
async def test_on_help_sends_pairing_code_to_unauthorized_private_user(monkeypatch) -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["999"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    update = _make_telegram_update(text="/help", chat_type="private")
    update.message.reply_text = AsyncMock()
    monkeypatch.setattr(
        "nanobot.channels.base.generate_code", lambda _ch, _sid: "ABCD-EFGH"
    )

    await channel._on_help(update, None)

    update.message.reply_text.assert_not_awaited()
    assert len(channel._app.bot.sent_messages) == 1
    assert "ABCD-EFGH" in channel._app.bot.sent_messages[0]["text"]


@pytest.mark.asyncio
async def test_on_message_pairs_unauthorized_private_user_before_side_effects(
    monkeypatch,
) -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["999"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    started_typing: list[str] = []
    channel._start_typing = lambda chat_id: started_typing.append(chat_id)
    channel._add_reaction = AsyncMock(return_value=None)
    channel._download_message_media = AsyncMock(return_value=([], []))
    monkeypatch.setattr(
        "nanobot.channels.base.generate_code", lambda _ch, _sid: "ABCD-EFGH"
    )

    await channel._on_message(_make_telegram_update(text="hello", chat_type="private"), None)

    assert started_typing == []
    channel._add_reaction.assert_not_awaited()
    channel._download_message_media.assert_not_awaited()
    assert len(channel._app.bot.sent_messages) == 1
    assert "ABCD-EFGH" in channel._app.bot.sent_messages[0]["text"]


@pytest.mark.asyncio
async def test_on_message_location_content() -> None:
    """Location messages are forwarded as [location: lat, lon] content."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    location = SimpleNamespace(latitude=48.8566, longitude=2.3522)
    update = _make_telegram_update(location=location)
    await channel._on_message(update, None)

    assert len(handled) == 1
    assert handled[0]["content"] == "[location: 48.8566, 2.3522]"


@pytest.mark.asyncio
async def test_on_message_location_with_text() -> None:
    """Location messages with accompanying text include both in content."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], group_policy="open"),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []
    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)
    channel._handle_message = capture_handle
    channel._start_typing = lambda _chat_id: None

    location = SimpleNamespace(latitude=51.5074, longitude=-0.1278)
    update = _make_telegram_update(text="meet me here", location=location)
    await channel._on_message(update, None)

    assert len(handled) == 1
    assert "meet me here" in handled[0]["content"]
    assert "[location: 51.5074, -0.1278]" in handled[0]["content"]


# ---------------------------------------------------------------------------
# Tests for retry amplification fix (issue #3050)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_text_does_not_fallback_on_network_timeout() -> None:
    """TimedOut should propagate immediately, NOT trigger plain-text fallback.

    Before the fix, _send_text caught ALL exceptions (including TimedOut)
    and retried as plain text, doubling connection demand during pool
    exhaustion — see issue #3050.
    """
    from telegram.error import TimedOut

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    call_count = 0

    async def always_timeout(**kwargs):
        nonlocal call_count
        call_count += 1
        raise TimedOut()

    channel._app.bot.send_message = always_timeout

    import nanobot.channels.telegram.runtime as tg_mod
    orig_delay = tg_mod._SEND_RETRY_BASE_DELAY
    tg_mod._SEND_RETRY_BASE_DELAY = 0.01
    try:
        with pytest.raises(TimedOut):
            await channel._send_text(123, "hello", None, {})
    finally:
        tg_mod._SEND_RETRY_BASE_DELAY = orig_delay

    # With the fix: only _call_with_retry's 3 HTML attempts (no plain fallback).
    # Before the fix: 3 HTML + 3 plain = 6 attempts.
    assert call_count == 3, (
        f"Expected 3 calls (HTML retries only), got {call_count} "
        "(plain-text fallback should not trigger on TimedOut)"
    )


@pytest.mark.asyncio
async def test_send_text_does_not_fallback_on_network_error() -> None:
    """NetworkError should propagate immediately, NOT trigger plain-text fallback."""
    from telegram.error import NetworkError

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    call_count = 0

    async def always_network_error(**kwargs):
        nonlocal call_count
        call_count += 1
        raise NetworkError("Connection reset")

    channel._app.bot.send_message = always_network_error

    import nanobot.channels.telegram.runtime as tg_mod
    orig_delay = tg_mod._SEND_RETRY_BASE_DELAY
    tg_mod._SEND_RETRY_BASE_DELAY = 0.01
    try:
        with pytest.raises(NetworkError):
            await channel._send_text(123, "hello", None, {})
    finally:
        tg_mod._SEND_RETRY_BASE_DELAY = orig_delay

    # _call_with_retry does NOT retry NetworkError (only TimedOut/RetryAfter),
    # so it raises after 1 attempt. The fix prevents plain-text fallback.
    # Before the fix: 1 HTML + 1 plain = 2. After the fix: 1 HTML only.
    assert call_count == 1, (
        f"Expected 1 call (HTML only, no plain fallback), got {call_count}"
    )


@pytest.mark.asyncio
async def test_send_text_falls_back_on_bad_request() -> None:
    """BadRequest (HTML parse error) should still trigger plain-text fallback."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    original_send = channel._app.bot.send_message
    html_call_count = 0

    async def html_fails(**kwargs):
        nonlocal html_call_count
        if kwargs.get("parse_mode") == "HTML":
            html_call_count += 1
            raise BadRequest("Can't parse entities")
        return await original_send(**kwargs)

    channel._app.bot.send_message = html_fails

    import nanobot.channels.telegram.runtime as tg_mod
    orig_delay = tg_mod._SEND_RETRY_BASE_DELAY
    tg_mod._SEND_RETRY_BASE_DELAY = 0.01
    try:
        await channel._send_text(123, "hello **world**", None, {})
    finally:
        tg_mod._SEND_RETRY_BASE_DELAY = orig_delay

    # HTML attempt failed with BadRequest → fallback to plain text succeeds.
    assert html_call_count == 1, f"Expected 1 HTML attempt, got {html_call_count}"
    assert len(channel._app.bot.sent_messages) == 1
    # Plain text send should NOT have parse_mode
    assert channel._app.bot.sent_messages[0].get("parse_mode") is None


@pytest.mark.asyncio
async def test_send_text_bad_request_plain_fallback_exhausted() -> None:
    """When both HTML and plain-text fallback fail with BadRequest, the error propagates."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    call_count = 0

    async def always_bad_request(**kwargs):
        nonlocal call_count
        call_count += 1
        raise BadRequest("Bad request")

    channel._app.bot.send_message = always_bad_request

    import nanobot.channels.telegram.runtime as tg_mod
    orig_delay = tg_mod._SEND_RETRY_BASE_DELAY
    tg_mod._SEND_RETRY_BASE_DELAY = 0.01
    try:
        with pytest.raises(BadRequest):
            await channel._send_text(123, "hello", None, {})
    finally:
        tg_mod._SEND_RETRY_BASE_DELAY = orig_delay

    # _call_with_retry does NOT retry BadRequest (only TimedOut/RetryAfter),
    # so HTML fails after 1 attempt → fallback to plain also fails after 1 attempt.
    # Before the fix: 2 total. After the fix: still 2 (BadRequest SHOULD fallback).
    assert call_count == 2, f"Expected 2 calls (1 HTML + 1 plain), got {call_count}"


# ---------------------------------------------------------------------------
# _markdown_to_telegram_html formatting tests
# ---------------------------------------------------------------------------

def test_markdown_to_html_headers_become_bold() -> None:
    from nanobot.channels.telegram.runtime import _markdown_to_telegram_html

    assert _markdown_to_telegram_html("# Title") == "<b>Title</b>"
    assert _markdown_to_telegram_html("## Subtitle") == "<b>Subtitle</b>"
    assert _markdown_to_telegram_html("### Deep") == "<b>Deep</b>"


def test_markdown_to_html_numbered_lists_preserved() -> None:
    from nanobot.channels.telegram.runtime import _markdown_to_telegram_html

    text = "1. First\n2. Second\n3. Third"
    result = _markdown_to_telegram_html(text)
    assert "1. First" in result
    assert "2. Second" in result
    assert "3. Third" in result


def test_markdown_to_html_numbered_list_normalizes_whitespace() -> None:
    from nanobot.channels.telegram.runtime import _markdown_to_telegram_html

    # Extra spaces after dot should be normalized
    text = "1.   Lots of space\n2.  Two spaces"
    result = _markdown_to_telegram_html(text)
    assert "1. Lots of space" in result
    assert "2. Two spaces" in result


def test_markdown_to_html_headers_survive_html_escaping() -> None:
    """Headers containing special HTML chars should still render as bold."""
    from nanobot.channels.telegram.runtime import _markdown_to_telegram_html

    result = _markdown_to_telegram_html("# A < B & C > D")
    assert "<b>A &lt; B &amp; C &gt; D</b>" == result


def test_markdown_to_html_mixed_formatting() -> None:
    """Headers, bullets, numbered lists, and bold coexist correctly."""
    from nanobot.channels.telegram.runtime import _markdown_to_telegram_html

    text = "# Overview\n\n- bullet one\n- bullet two\n\n1. step one\n2. step two\n\n**bold text**"
    result = _markdown_to_telegram_html(text)
    assert "<b>Overview</b>" in result
    assert "\u2022 bullet one" in result
    assert "1. step one" in result
    assert "<b>bold text</b>" in result


# ---------------------------------------------------------------------------
# _strip_md_block tests
# ---------------------------------------------------------------------------

def test_strip_md_block_removes_inline_formatting() -> None:
    from nanobot.channels.telegram.runtime import _strip_md_block

    text = "**bold** and _italic_ and ~~struck~~"
    result = _strip_md_block(text)
    assert result == "bold and italic and struck"


def test_strip_md_block_strips_headers() -> None:
    from nanobot.channels.telegram.runtime import _strip_md_block

    assert _strip_md_block("## Title\nBody") == "Title\nBody"


def test_strip_md_block_converts_bullets_and_numbers() -> None:
    from nanobot.channels.telegram.runtime import _strip_md_block

    text = "- item a\n1. item b\n2. item c"
    result = _strip_md_block(text)
    assert "\u2022 item a" in result
    assert "1. item b" in result
    assert "2. item c" in result


def test_strip_md_block_strips_links() -> None:
    from nanobot.channels.telegram.runtime import _strip_md_block

    assert _strip_md_block("[click here](https://example.com)") == "click here"


# ---------------------------------------------------------------------------
# Streaming mid-edit uses _strip_md_block
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_delta_mid_stream_strips_markdown() -> None:
    """Mid-stream edits should strip markdown so users see clean text."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=42))
    channel._app.bot.edit_message_text = AsyncMock()

    # Initial send with markdown
    await channel.send_delta("999", "**hello** world")
    sent_text = channel._app.bot.send_message.call_args.kwargs.get("text", "")
    # Should NOT contain raw markdown asterisks
    assert "**" not in sent_text
    assert "hello world" in sent_text

    # Mid-stream edit
    import time
    buf = channel._stream_bufs["999"]
    buf.last_edit = time.monotonic() - 10  # force edit interval
    await channel.send_delta("999", "\n### Title\n1. step")
    edited_text = channel._app.bot.edit_message_text.call_args.kwargs.get("text", "")
    assert "###" not in edited_text
    assert "**" not in edited_text
    assert "Title" in edited_text
    assert "1. step" in edited_text


def test_build_keyboard_respects_inline_keyboards_flag() -> None:
    """``_build_keyboard`` returns ``None`` whenever the feature flag is off,
    regardless of whether buttons are provided; returns a proper Markup only
    when the flag is explicitly enabled. Pins the kill-switch so accidentally
    flipping the default doesn't silently expose callback handlers."""
    from telegram import InlineKeyboardMarkup

    off = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", inline_keyboards=False),
        MessageBus(),
    )
    assert off._build_keyboard([["A", "B"]]) is None

    on = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", inline_keyboards=True),
        MessageBus(),
    )
    assert on._build_keyboard([]) is None  # empty still no-op
    markup = on._build_keyboard([["Yes", "No"], ["Cancel"]])
    assert isinstance(markup, InlineKeyboardMarkup)
    rows = markup.inline_keyboard
    assert [[b.text for b in row] for row in rows] == [["Yes", "No"], ["Cancel"]]
    # callback_data mirrors label so _on_callback_query can echo the tap back.
    assert rows[0][0].callback_data == "Yes"


def test_safe_callback_data_truncates_at_utf8_boundary() -> None:
    # Telegram's 64-byte callback_data cap is a hard API limit; silent 400s were the bug.
    short = "Yes"
    assert TelegramChannel._safe_callback_data(short) == short

    long_ascii = "a" * 100
    out = TelegramChannel._safe_callback_data(long_ascii)
    assert len(out.encode("utf-8")) <= 64
    assert long_ascii.startswith(out)

    # Multibyte labels must not split a codepoint mid-byte.
    long_cjk = "同意并继续下一步，我已阅读并同意了服务条款以及隐私政策"
    assert len(long_cjk.encode("utf-8")) > 64
    out = TelegramChannel._safe_callback_data(long_cjk)
    assert len(out.encode("utf-8")) <= 64
    assert long_cjk.startswith(out)
    out.encode("utf-8").decode("utf-8")  # must round-trip cleanly


def test_build_keyboard_uses_safe_callback_data_for_long_labels() -> None:
    # Pins the integration so a long-label payload survives ``send_message`` instead of 400ing.
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", inline_keyboards=True),
        MessageBus(),
    )
    long_label = "Approve and continue to the next step with the updated terms of service"
    assert len(long_label.encode("utf-8")) > 64

    markup = channel._build_keyboard([[long_label]])
    btn = markup.inline_keyboard[0][0]
    assert btn.text == long_label  # display preserved
    assert len(btn.callback_data.encode("utf-8")) <= 64
    assert long_label.startswith(btn.callback_data)


def test_buttons_as_text_format_preserves_rows_and_labels() -> None:
    # Canonical shape: one row per line, labels bracketed. Layout survives the fallback.
    assert TelegramChannel._buttons_as_text([["Yes", "No"], ["Cancel"]]) == "[Yes] [No]\n[Cancel]"
    assert TelegramChannel._buttons_as_text([["Only"]]) == "[Only]"
    assert TelegramChannel._buttons_as_text([[], ["A"]]) == "[A]"  # empty rows skipped


@pytest.mark.asyncio
async def test_send_falls_back_buttons_to_inline_text_when_flag_off() -> None:
    """Buttons are semantic options; with ``inline_keyboards=False`` we must
    splice labels into the text so users still see the choices. Silent-drop
    was the pre-fallback bug — the agent got a success reply while the user
    saw a question with no options."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], inline_keyboards=False),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Proceed?",
            buttons=[["Yes", "No"], ["Cancel"]],
        )
    )

    assert len(channel._app.bot.sent_messages) == 1
    sent = channel._app.bot.sent_messages[0]
    assert sent.get("reply_markup") is None
    assert "Proceed?" in sent["text"]
    assert "[Yes] [No]" in sent["text"]
    assert "[Cancel]" in sent["text"]


@pytest.mark.asyncio
async def test_send_uses_native_keyboard_when_flag_on() -> None:
    """With the flag on, the content stays clean and buttons ride in ``reply_markup``."""
    from telegram import InlineKeyboardMarkup

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], inline_keyboards=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Proceed?",
            buttons=[["Yes", "No"]],
        )
    )

    sent = channel._app.bot.sent_messages[0]
    assert isinstance(sent.get("reply_markup"), InlineKeyboardMarkup)
    assert "[Yes]" not in sent["text"]  # native keyboard owns the rendering


@pytest.mark.asyncio
async def test_callback_query_ignores_unauthorized_user_before_side_effects() -> None:
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["999"], inline_keyboards=True),
        MessageBus(),
    )
    channel._handle_message = AsyncMock()

    query = SimpleNamespace(
        id="cb_1",
        data="Yes",
        answer=AsyncMock(),
        message=SimpleNamespace(
            chat_id=123,
            edit_reply_markup=AsyncMock(),
        ),
    )
    update = SimpleNamespace(
        callback_query=query,
        effective_user=SimpleNamespace(id=12345, username="alice", first_name="Alice"),
    )

    await channel._on_callback_query(update, None)

    query.answer.assert_not_awaited()
    query.message.edit_reply_markup.assert_not_awaited()
    channel._handle_message.assert_not_awaited()

# ---------------------------------------------------------------------------
# T1: Telegram Generative UI — Rich Messages, drafts, reply keyboards,
# comandos dinámicos y ephemeral messages (spec-telegram-generative-ui.md)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_rich_includes_reply_markup_and_ephemeral() -> None:
    """T1.1: sendRichMessage payload carries reply_markup and ephemeral fields."""
    from telegram import InlineKeyboardMarkup

    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, inline_keyboards=True,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="**hola**",
            metadata={"user_id": 12345},
            ephemeral=True,
        )
    )

    # Sin reply_markup en este caso; verificar ephemeral en el payload rich.
    call = channel._app.bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    payload = call.kwargs["api_kwargs"]
    assert payload["rich_message"]["markdown"] == "**hola**"
    assert payload.get("is_ephemeral") is True
    assert payload.get("receiver_user_id") == 12345

    # Con reply_markup explícito (inline keyboard) también viaja en el payload.
    channel._app.bot.do_api_request.reset_mock()
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="**hola**",
            buttons=[["Go"]],
        )
    )
    call = channel._app.bot.do_api_request.call_args
    payload = call.kwargs["api_kwargs"]
    assert isinstance(payload.get("reply_markup"), InlineKeyboardMarkup)

    # Con reply_keyboard (teclado de respuesta) viaja como ReplyKeyboardMarkup
    # en el payload rich (solo en el último chunk).
    from telegram import ReplyKeyboardMarkup

    channel._app.bot.do_api_request.reset_mock()
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="**hola**",
            reply_keyboard=[["Sí", "No"]],
        )
    )
    call = channel._app.bot.do_api_request.call_args
    payload = call.kwargs["api_kwargs"]
    assert isinstance(payload.get("reply_markup"), ReplyKeyboardMarkup)
    labels = [btn.text for row in payload["reply_markup"].keyboard for btn in row]
    assert labels == ["Sí", "No"]


@pytest.mark.asyncio
async def test_send_rich_splits_oversized_content_into_chunks() -> None:
    """T1.1: contenido > 30.000 chars se parte en chunks rich (límite 32.768)."""
    from nanobot.channels.telegram.runtime import TELEGRAM_RICH_MAX_LEN

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], rich_messages=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    big = "# Título\n\n" + ("párrafo **bold** con texto\n\n" * 2000)
    assert len(big) > TELEGRAM_RICH_MAX_LEN

    await channel.send(OutboundMessage(channel="telegram", chat_id="123", content=big))

    calls = [c for c in channel._app.bot.do_api_request.call_args_list if c.args[0] == "sendRichMessage"]
    assert len(calls) >= 2
    for c in calls:
        md = c.kwargs["api_kwargs"]["rich_message"]["markdown"]
        assert len(md) <= TELEGRAM_RICH_MAX_LEN
    # El contenido completo se preserva (sin pérdida de texto).
    joined = "".join(c.kwargs["api_kwargs"]["rich_message"]["markdown"] for c in calls)
    assert "".join(joined.split()) == "".join(big.split())


@pytest.mark.asyncio
async def test_send_delta_rich_finalizes_with_edit_rich_in_place() -> None:
    """T1.2: con rich_messages, el streaming usa un preview legacy
    (send_message + edit_message_text) y en stream_end convierte el preview
    en rich message in-place con editMessageText(rich_message=...) — un solo
    mensaje, sin drafts huérfanos ni duplicados."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_delta("123", "hola ", stream_id="s:0")
    await asyncio.sleep(0.15)
    await channel.send_delta("123", "mundo", stream_id="s:0")

    # Preview legacy: un solo mensaje real, editado durante el stream.
    assert len(channel._app.bot.sent_messages) == 1
    assert channel._stream_bufs["123"].message_id == 1
    # No se usaron drafts rich durante el streaming.
    draft_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "sendRichMessageDraft"
    ]
    assert draft_calls == []

    await channel.send_delta("123", "", stream_id="s:0", stream_end=True)

    # Final: editMessageText rich in-place sobre el preview.
    edit_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "editMessageText"
    ]
    assert len(edit_calls) == 1
    assert edit_calls[0].kwargs["api_kwargs"]["message_id"] == 1
    assert edit_calls[0].kwargs["api_kwargs"]["rich_message"]["markdown"] == "hola mundo"
    # Un solo mensaje en el chat: el preview editado, sin huérfanos.
    assert len(channel._app.bot.sent_messages) == 1
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_stream_legacy_second_send_no_quote() -> None:
    """T-004: en streaming legacy el primer preview quotea; el segundo stream no."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    meta = {"message_id": 10}
    await channel.send_delta("123", "hola ", stream_id="s:0", metadata=meta)
    await asyncio.sleep(0.05)
    await channel.send_delta("123", "mundo", stream_id="s:0", metadata=meta)
    await channel.send_delta("123", "", stream_id="s:0", stream_end=True, metadata=meta)

    assert len(channel._app.bot.sent_messages) == 1
    assert channel._app.bot.sent_messages[0]["reply_parameters"]["message_id"] == 10

    await channel.send_delta("123", "segundo stream", stream_id="s:1", metadata=meta)
    await asyncio.sleep(0.05)
    await channel.send_delta("123", " fin", stream_id="s:1", metadata=meta)
    await channel.send_delta("123", "", stream_id="s:1", stream_end=True, metadata=meta)

    assert len(channel._app.bot.sent_messages) == 2
    assert channel._app.bot.sent_messages[1].get("reply_parameters") is None


@pytest.mark.asyncio
async def test_stream_rich_draft_second_send_no_quote() -> None:
    """T-004: en streaming rich el primer draft quotea; el segundo stream no."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    meta = {"is_group": False, "message_id": 20}

    # Primer stream: el reasoning abre el draft; el final fija con sendRichMessage.
    await channel.send_reasoning_delta("123", "analizo", metadata=meta, stream_id="s:0")
    await channel.send_reasoning_end("123", metadata=meta, stream_id="s:0")
    await channel.send_delta("123", "hola ", stream_id="s:0", metadata=meta)
    await channel.send_delta("123", "", stream_id="s:0", stream_end=True, metadata=meta)

    rich_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "sendRichMessage"
    ]
    assert len(rich_calls) == 1
    assert rich_calls[0].kwargs["api_kwargs"]["reply_parameters"]["message_id"] == 20

    channel._app.bot.do_api_request.reset_mock()

    # Segundo stream: mismo flujo, el chat ya está anclado → sin quote.
    await channel.send_reasoning_delta("123", "otra vez", metadata=meta, stream_id="s:1")
    await channel.send_reasoning_end("123", metadata=meta, stream_id="s:1")
    await channel.send_delta("123", "segundo ", stream_id="s:1", metadata=meta)
    await channel.send_delta("123", "", stream_id="s:1", stream_end=True, metadata=meta)

    rich_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "sendRichMessage"
    ]
    assert len(rich_calls) == 1
    assert rich_calls[0].kwargs["api_kwargs"].get("reply_parameters") is None


@pytest.mark.asyncio
async def test_stream_rich_no_preview_legacy_sends_new_message_with_quote_only_first() -> None:
    """T-004: si no hay preview legacy ni draft, el envío final es un mensaje nuevo
    y respeta el ancla: el primero quotea, los siguientes no."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    # Forzar fallback total: rich y edit rich fallan, no hay preview legacy.
    channel._app.bot.do_api_request = AsyncMock(side_effect=BadRequest("Method not found"))

    meta = {"message_id": 30}
    await channel.send_delta("123", "primera respuesta", stream_id="s:0", metadata=meta)
    await channel.send_delta("123", "", stream_id="s:0", stream_end=True, metadata=meta)

    assert len(channel._app.bot.sent_messages) == 1
    assert channel._app.bot.sent_messages[0]["reply_parameters"]["message_id"] == 30

    await channel.send_delta("123", "segunda respuesta", stream_id="s:1", metadata=meta)
    await channel.send_delta("123", "", stream_id="s:1", stream_end=True, metadata=meta)

    assert len(channel._app.bot.sent_messages) == 2
    assert channel._app.bot.sent_messages[1].get("reply_parameters") is None


@pytest.mark.asyncio
async def test_send_delta_rich_reply_parameters_propagate_to_preview_and_final() -> None:
    """T1.2: cuando el stream responde a un mensaje (metadata.message_id),
    el preview legacy y el edit rich final llevan reply_parameters."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    meta = {"message_id": 10}
    await channel.send_delta("123", "hola ", stream_id="s:0", metadata=meta)
    await asyncio.sleep(0.15)
    await channel.send_delta("123", "mundo", stream_id="s:0", metadata=meta)

    # El preview legacy se envía como reply al mensaje original.
    assert len(channel._app.bot.sent_messages) == 1
    rp = channel._app.bot.sent_messages[0].get("reply_parameters")
    assert rp is not None
    assert rp["message_id"] == 10
    assert rp["allow_sending_without_reply"] is True

    await channel.send_delta("123", "", stream_id="s:0", stream_end=True, metadata=meta)

    edit_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "editMessageText"
    ]
    assert len(edit_calls) == 1
    assert edit_calls[0].kwargs["api_kwargs"]["rich_message"]["markdown"] == "hola mundo"
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_rich_falls_back_to_legacy_edit_when_rich_unsupported() -> None:
    """T1.2: si editMessageText rich no está disponible, se cae al path
    legacy (edit HTML in-place) y se hace latch-off del rich."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], rich_messages=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock(side_effect=BadRequest("Method not found"))

    await channel.send_delta("123", "hola", stream_id="s:0")
    await channel.send_delta("123", "", stream_id="s:0", stream_end=True)

    assert channel._rich_send_disabled is True
    # El preview legacy se editó con HTML (un solo mensaje, sin huérfanos).
    assert len(channel._app.bot.sent_messages) == 1
    assert channel._app.bot.edited_messages == [1]
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_send_delta_without_rich_keeps_legacy_path() -> None:
    """T1.2: sin rich_messages, el streaming sigue usando send + edit."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_delta("123", "hola", stream_id="s:0")
    await channel.send_delta("123", " mundo", stream_id="s:0")

    channel._app.bot.do_api_request.assert_not_called()
    assert len(channel._app.bot.sent_messages) == 1
    assert channel._stream_bufs["123"].message_id == 1


@pytest.mark.asyncio
async def test_send_reply_keyboard_on_final_chunk() -> None:
    """T1.3: reply_keyboard se adjunta como ReplyKeyboardMarkup (one_time +
    placeholder) solo en el último chunk del mensaje."""
    from telegram import ReplyKeyboardMarkup

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    long_text = "línea\n" * 700  # fuerza múltiples chunks (> 4000 chars)
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content=long_text,
            reply_keyboard=[["Sí", "No"], ["Cancelar"]],
        )
    )

    assert len(channel._app.bot.sent_messages) > 1
    for i, sent in enumerate(channel._app.bot.sent_messages):
        if i < len(channel._app.bot.sent_messages) - 1:
            assert sent.get("reply_markup") is None
        else:
            markup = sent.get("reply_markup")
            assert isinstance(markup, ReplyKeyboardMarkup)
            assert markup.one_time_keyboard is True
            assert markup.input_field_placeholder == "Elige una opción…"
            labels = [btn.text for row in markup.keyboard for btn in row]
            assert labels == ["Sí", "No", "Cancelar"]


@pytest.mark.asyncio
async def test_send_reply_keyboard_remove_on_empty_list() -> None:
    """T1.3b: reply_keyboard=[] (lista vacía explícita) envía ReplyKeyboardRemove
    para descartar un teclado pegado; reply_keyboard=None no envía markup."""
    from telegram import ReplyKeyboardRemove

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    # [] explícito → ReplyKeyboardRemove
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Teclado removido",
            reply_keyboard=[],
        )
    )
    sent = channel._app.bot.sent_messages[-1]
    assert isinstance(sent.get("reply_markup"), ReplyKeyboardRemove)

    # None (default) → sin markup
    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Sin teclado",
        )
    )
    sent = channel._app.bot.sent_messages[-1]
    assert sent.get("reply_markup") is None


@pytest.mark.asyncio
async def test_send_menu_commands_uses_chat_scope() -> None:
    """T1.4: menu_commands registra setMyCommands con scope por chat; un
    fallo del registro no rompe el envío."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.set_my_commands = AsyncMock()

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Usa el menú",
            menu_commands=[
                {"command": "agenda", "description": "Ver agenda de hoy"},
                {"command": "compras", "description": "Lista de compras"},
            ],
        )
    )

    channel._app.bot.set_my_commands.assert_awaited_once()
    kwargs = channel._app.bot.set_my_commands.call_args.kwargs
    assert kwargs["scope"] == {"type": "chat", "chat_id": 123}
    assert [c.command for c in kwargs["commands"]] == ["agenda", "compras"]

    # Fallo del registro: best-effort, no lanza.
    channel._app.bot.set_my_commands = AsyncMock(side_effect=RuntimeError("boom"))
    await channel.send(
        OutboundMessage(channel="telegram", chat_id="123", content="ok", menu_commands=[{"command": "x", "description": "y"}])
    )
    assert channel._app.bot.sent_messages[-1]["text"] == "ok"


@pytest.mark.asyncio
async def test_send_ephemeral_falls_back_when_unsupported() -> None:
    """T1.5: ephemeral=True envía is_ephemeral + receiver_user_id; si el
    servidor no lo soporta (BadRequest), reintenta sin ephemeral."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.send_message = AsyncMock(
        side_effect=[BadRequest("Bad Request: is_ephemeral is not supported"), None]
    )

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="respuesta privada",
            metadata={"user_id": 12345},
            ephemeral=True,
        )
    )

    calls = channel._app.bot.send_message.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs.get("is_ephemeral") is True
    assert calls[0].kwargs.get("receiver_user_id") == 12345
    assert "is_ephemeral" not in calls[1].kwargs
    assert calls[1].kwargs["text"] == "respuesta privada"


# T1: Telegram Thinking Blocks — reasoning visible con <tg-thinking> en draft
# y <details> en el mensaje final (spec-telegram-thinking-blocks.md)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoning_delta_opens_draft_with_thinking_block() -> None:
    """T1.1: send_reasoning_delta con rich + chat privado abre un draft
    sendRichMessageDraft con draft_id estable y <tg-thinking>; los deltas
    siguientes reusan el mismo draft_id; send_reasoning_end no envía nada."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_reasoning_delta(
        "123", "El usuario pide ", metadata={"is_group": False}, stream_id="s:0"
    )
    await asyncio.sleep(0.15)
    await channel.send_reasoning_delta(
        "123", "una spec", metadata={"is_group": False}, stream_id="s:0"
    )
    await channel.send_reasoning_end("123", metadata={"is_group": False}, stream_id="s:0")

    draft_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "sendRichMessageDraft"
    ]
    assert len(draft_calls) == 2
    first = draft_calls[0].kwargs["api_kwargs"]
    second = draft_calls[1].kwargs["api_kwargs"]
    assert first["draft_id"] is not None
    assert first["draft_id"] == second["draft_id"]
    assert "<tg-thinking>" in first["rich_message"]["markdown"]
    assert "El usuario pide " in first["rich_message"]["markdown"]
    assert "una spec" in second["rich_message"]["markdown"]
    # send_reasoning_end no envía nada nuevo (siguen siendo 2 drafts).
    assert len(draft_calls) == 2
    assert channel._stream_bufs["123"].reasoning == "El usuario pide una spec"
    assert channel._stream_bufs["123"].using_draft is True


@pytest.mark.asyncio
async def test_reasoning_legacy_uses_expandable_blockquote() -> None:
    """T1.2: sin rich (o en grupos), el reasoning usa preview legacy con
    <blockquote expandable>; send_reasoning_end es no-op."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_reasoning_delta(
        "123", "pensando...", metadata={"is_group": True}, stream_id="s:0"
    )
    await channel.send_reasoning_end("123", metadata={"is_group": True}, stream_id="s:0")

    # Sin drafts rich.
    draft_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "sendRichMessageDraft"
    ]
    assert draft_calls == []
    # Preview legacy con blockquote expandible.
    assert len(channel._app.bot.sent_messages) == 1
    text = channel._app.bot.sent_messages[0]["text"]
    assert "<blockquote expandable>" in text
    assert "pensando..." in text
    assert channel._stream_bufs["123"].using_draft is False


@pytest.mark.asyncio
async def test_reasoning_finalizes_draft_with_details() -> None:
    """T1.3: stream_end con draft activo fija con sendRichMessage(draft_id=...)
    y el markdown final incluye <details> con el reasoning; reply_parameters
    conservado; buffer limpiado."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_reasoning_delta(
        "123", "analizo", metadata={"is_group": False, "message_id": 42}, stream_id="s:0"
    )
    await channel.send_reasoning_end("123", metadata={"is_group": False}, stream_id="s:0")
    await channel.send_delta(
        "123", "Respuesta final", metadata={"is_group": False, "message_id": 42}, stream_id="s:0"
    )
    await channel.send_delta(
        "123", "", metadata={"is_group": False, "message_id": 42},
        stream_id="s:0", stream_end=True,
    )

    final_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "sendRichMessage"
    ]
    assert len(final_calls) == 1
    payload = final_calls[0].kwargs["api_kwargs"]
    assert payload["draft_id"] is not None
    md = payload["rich_message"]["markdown"]
    assert "Respuesta final" in md
    assert "<details>" in md
    assert "🧠 Razonamiento" in md
    assert "analizo" in md
    assert payload["reply_parameters"]["message_id"] == 42
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_reasoning_draft_expired_falls_back_to_legacy() -> None:
    """T1.4: draft expirado en stream_end → path legacy con el contenido
    acumulado (send_message + edit_message_text), sin sendRichMessage."""
    import time as _time

    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_reasoning_delta(
        "123", "analizo", metadata={"is_group": False}, stream_id="s:0"
    )
    buf = channel._stream_bufs["123"]
    assert buf.using_draft is True
    # Simular expiración del draft (más de 30 s sin deltas).
    buf.draft_expires_at = _time.monotonic() - 1.0

    await channel.send_delta("123", "Respuesta", metadata={"is_group": False}, stream_id="s:0")
    await channel.send_delta(
        "123", "", metadata={"is_group": False}, stream_id="s:0", stream_end=True
    )

    final_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "sendRichMessage"
    ]
    assert final_calls == []
    # El contenido se envió por legacy.
    assert len(channel._app.bot.sent_messages) >= 1
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_reasoning_finalize_failure_falls_back_to_legacy() -> None:
    """T1.5: sendRichMessage falla al fijar → fallback legacy con el contenido
    acumulado; error de capacidad → latch-off _rich_send_disabled."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    # Draft OK (1 llamada), fijación falla con error de capacidad.
    channel._app.bot.do_api_request = AsyncMock(
        side_effect=[None, BadRequest("Method not found")]
    )

    await channel.send_reasoning_delta(
        "123", "analizo", metadata={"is_group": False}, stream_id="s:0"
    )
    await channel.send_delta("123", "Respuesta", metadata={"is_group": False}, stream_id="s:0")
    await channel.send_delta(
        "123", "", metadata={"is_group": False}, stream_id="s:0", stream_end=True
    )

    assert channel._rich_send_disabled is True
    # Fallback legacy: el contenido se envió por send_message.
    assert len(channel._app.bot.sent_messages) >= 1
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_reasoning_disabled_is_noop() -> None:
    """T1.6: show_reasoning=False → send_reasoning_delta/end no-op (sin draft,
    sin acumulación, sin mensajes)."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"], rich_messages=True
        ),
        MessageBus(),
    )
    channel.show_reasoning = False
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_reasoning_delta(
        "123", "analizo", metadata={"is_group": False}, stream_id="s:0"
    )
    await channel.send_reasoning_end("123", metadata={"is_group": False}, stream_id="s:0")

    channel._app.bot.do_api_request.assert_not_called()
    assert channel._app.bot.sent_messages == []
    assert "123" not in channel._stream_bufs


@pytest.mark.asyncio
async def test_stream_without_reasoning_has_no_details() -> None:
    """T1.7: send_delta sin reasoning previo → path actual intacto; el mensaje
    final rich no incluye <details> de razonamiento."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            rich_messages=True, stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_delta("123", "hola", stream_id="s:0")
    await channel.send_delta("123", "", stream_id="s:0", stream_end=True)

    # Sin draft: la fijación es editMessageText rich in-place (path actual).
    edit_calls = [
        c for c in channel._app.bot.do_api_request.call_args_list
        if c.args[0] == "editMessageText"
    ]
    assert len(edit_calls) == 1
    md = edit_calls[0].kwargs["api_kwargs"]["rich_message"]["markdown"]
    assert "<details>" not in md
    assert "hola" in md


# ---------------------------------------------------------------------------
# T2: Telegram UX — Task lists rich administradas por el agente, polls de
# aprobación y efectos de mensaje (spec-telegram-ux-checklists-polls-effects.md)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_checklist_sends_rich_task_list() -> None:
    """T2.1: send() con checklist={title, tasks} → sendRichMessage con
    markdown rich `# title` + `- [ ] task` por tarea (checkboxes nativos)."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], rich_messages=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=1))

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            checklist={"title": "Plan SDD", "tasks": ["Spec", "Plan", "Tareas"]},
        )
    )

    call = channel._app.bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    payload = call.kwargs["api_kwargs"]
    md = payload["rich_message"]["markdown"]
    assert md.startswith("# Plan SDD")
    assert "- [ ] Spec" in md
    assert "- [ ] Plan" in md
    assert "- [ ] Tareas" in md
    # El message_id de la task list queda registrado para updates posteriores.
    assert channel._task_lists["123"]["message_id"] == 1
    assert channel._task_lists["123"]["tasks"] == ["Spec", "Plan", "Tareas"]


@pytest.mark.asyncio
async def test_send_checklist_update_edits_rich_in_place() -> None:
    """T2.2: checklist_update={message_id, done} → editMessageText rich con
    `- [x]` en las marcadas, `- [ ]` en el resto y resumen de progreso."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], rich_messages=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock(return_value=SimpleNamespace(message_id=1))

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            checklist={"title": "Plan SDD", "tasks": ["Spec", "Plan", "Tareas"]},
        )
    )
    channel._app.bot.do_api_request.reset_mock()

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            checklist_update={"message_id": 1, "done": [0, 2]},
        )
    )

    call = channel._app.bot.do_api_request.call_args
    assert call.args[0] == "editMessageText"
    payload = call.kwargs["api_kwargs"]
    assert payload["message_id"] == 1
    md = payload["rich_message"]["markdown"]
    assert "- [x] Spec" in md
    assert "- [ ] Plan" in md
    assert "- [x] Tareas" in md
    assert "✅ 2/3 tareas completadas" in md


@pytest.mark.asyncio
async def test_send_poll_sends_native_poll() -> None:
    """T2.3: send() con poll={question, options} → send_poll nativo con
    is_anonymous=False y allows_multiple_answers=False; poll cacheado."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="",
            poll={"question": "¿Apruebas el plan?", "options": ["APROBAR", "CAMBIOS", "RECHAZAR"]},
        )
    )

    assert len(channel._app.bot.sent_polls) == 1
    kwargs = channel._app.bot.sent_polls[0]
    assert kwargs["chat_id"] == 123
    assert kwargs["question"] == "¿Apruebas el plan?"
    assert kwargs["options"] == ["APROBAR", "CAMBIOS", "RECHAZAR"]
    assert kwargs["is_anonymous"] is False
    assert kwargs["allows_multiple_answers"] is False
    # El poll queda cacheado (poll_id → chat_id + options).
    assert "poll_1" in channel._polls_cache
    assert channel._polls_cache["poll_1"]["options"] == ["APROBAR", "CAMBIOS", "RECHAZAR"]


@pytest.mark.asyncio
async def test_poll_answer_publishes_vote_as_turn_context() -> None:
    """T2.4: poll_answer → InboundMessage con prefijo "🗳️ El usuario votó:"
    y la opción resuelta vía cache; se encola como turno normal."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._polls_cache["poll_1"] = {
        "chat_id": 123,
        "options": ["APROBAR", "CAMBIOS", "RECHAZAR"],
    }
    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle

    poll_answer = SimpleNamespace(
        poll_id="poll_1",
        option_ids=[0],
        user=SimpleNamespace(id=12345, username="alice", first_name="Alice"),
    )
    update = SimpleNamespace(poll_answer=poll_answer, effective_user=poll_answer.user)

    await channel._on_poll_answer(update, None)

    assert len(handled) == 1
    assert handled[0]["content"] == "🗳️ El usuario votó: APROBAR"
    assert handled[0]["chat_id"] == "123"
    assert handled[0]["metadata"]["poll_id"] == "poll_1"


@pytest.mark.asyncio
async def test_poll_answer_without_cache_falls_back_to_poll_id() -> None:
    """T2.4: sin cache (gateway reiniciado) → se publica el poll_id crudo."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    handled = []

    async def capture_handle(**kwargs) -> None:
        handled.append(kwargs)

    channel._handle_message = capture_handle

    poll_answer = SimpleNamespace(
        poll_id="poll_desconocido",
        option_ids=[1],
        user=SimpleNamespace(id=12345, username="alice", first_name="Alice"),
    )
    update = SimpleNamespace(poll_answer=poll_answer, effective_user=poll_answer.user)

    await channel._on_poll_answer(update, None)

    assert len(handled) == 1
    assert "poll_desconocido" in handled[0]["content"]


@pytest.mark.asyncio
async def test_send_effect_applies_message_effect_id_rich() -> None:
    """T2.5: effect → message_effect_id en el payload de sendRichMessage."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], rich_messages=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="¡Aprobado! 🎉",
            effect="confeti",
        )
    )

    call = channel._app.bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    payload = call.kwargs["api_kwargs"]
    assert payload["message_effect_id"] == "5046509860389126442"


@pytest.mark.asyncio
async def test_send_effect_applies_message_effect_id_legacy() -> None:
    """T2.5: effect → message_effect_id en send_message (path legacy)."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="¡Aprobado! 🎉",
            effect="confeti",
        )
    )

    assert len(channel._app.bot.sent_messages) == 1
    assert channel._app.bot.sent_messages[0]["message_effect_id"] == "5046509860389126442"


@pytest.mark.asyncio
async def test_send_effect_config_default_applies_when_no_override() -> None:
    """REQ-003: config message_effect_id seteado en el canal se aplica si el
    mensaje no trae override (opt-in por canal)."""
    channel = TelegramChannel(
        TelegramConfig(
            enabled=True,
            token="123:abc",
            allow_from=["*"],
            message_effect_id="confeti",
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(channel="telegram", chat_id="123", content="¡Listo!")
    )

    assert len(channel._app.bot.sent_messages) == 1
    assert channel._app.bot.sent_messages[0]["message_effect_id"] == "5046509860389126442"


@pytest.mark.asyncio
async def test_send_without_effect_omits_message_effect_id() -> None:
    """REQ-001/REQ-005: sin override y sin config.message_effect_id, ningún
    envío lleva message_effect_id (el confeti deja de ser el default)."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(channel="telegram", chat_id="123", content="Respuesta normal")
    )

    assert len(channel._app.bot.sent_messages) == 1
    assert "message_effect_id" not in channel._app.bot.sent_messages[0]


@pytest.mark.asyncio
async def test_send_without_effect_omits_message_effect_id_rich() -> None:
    """REQ-001/REQ-005: igual que el anterior pero por el path rich
    (sendRichMessage): el payload no incluye message_effect_id."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], rich_messages=True),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send(
        OutboundMessage(channel="telegram", chat_id="123", content="Respuesta normal")
    )

    call = channel._app.bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    payload = call.kwargs["api_kwargs"]
    assert "message_effect_id" not in payload


@pytest.mark.asyncio
async def test_send_effect_bad_request_retries_without_effect() -> None:
    """T2.5: BadRequest por efecto no soportado → reintento sin efecto
    (best-effort, sin latch)."""
    from telegram.error import BadRequest

    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)

    real_send = channel._app.bot.send_message
    calls = []

    async def flaky_send(**kwargs):
        calls.append(kwargs)
        if kwargs.get("message_effect_id"):
            raise BadRequest("Bad Request: message effect is not supported")
        return await real_send(**kwargs)

    channel._app.bot.send_message = flaky_send

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="¡Aprobado! 🎉",
            effect="confeti",
        )
    )

    assert len(calls) == 2
    assert calls[0].get("message_effect_id") == "5046509860389126442"
    assert "message_effect_id" not in calls[1]
    assert len(channel._app.bot.sent_messages) == 1


@pytest.mark.asyncio
async def test_stop_cancels_typing_indicator_and_shuts_down_app() -> None:
    """stop() must not crash on the refactored TypingIndicator (regression: _typing_tasks)."""
    channel = TelegramChannel(
        TelegramConfig(enabled=True, token="123:abc", allow_from=["*"]),
        MessageBus(),
    )
    app = _FakeApp(lambda: None)
    app.updater.stop = AsyncMock()
    channel._app = app

    # Start a live typing indicator for a chat
    channel._start_typing("123")
    assert "123" in channel._typing._tasks
    task = channel._typing._tasks["123"]
    assert not task.done()

    await channel.stop()

    # Typing task cancelled and removed, and the app shutdown path was reached
    assert "123" not in channel._typing._tasks
    assert task.cancelling() > 0 or task.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await task
    app.updater.stop.assert_awaited_once()
    assert channel._app is None


@pytest.mark.asyncio
async def test_start_failure_cleans_state_and_stop_is_safe(monkeypatch) -> None:
    """Regression (2026-08-19 incident): when start() fails during initialize()
    (e.g. telegram.error.TimedOut in getMe), the channel must not stay
    'running' and a later stop() must not raise
    RuntimeError('This Updater is not running!')."""
    _FakeHTTPXRequest.clear()
    config = TelegramConfig(
        enabled=True,
        token="123:abc",
        allow_from=["*"],
    )
    channel = TelegramChannel(config, MessageBus())
    app = _FakeApp(lambda: None)
    app.initialize = AsyncMock(side_effect=RuntimeError("simulated getMe timeout"))
    builder = _FakeBuilder(app)

    monkeypatch.setattr("nanobot.channels.telegram.runtime.HTTPXRequest", _FakeHTTPXRequest)
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.Application",
        SimpleNamespace(builder=lambda: builder),
    )

    with pytest.raises(RuntimeError, match="simulated getMe timeout"):
        await channel.start()

    # is_running must reflect reality and stop() must be a safe no-op.
    assert channel.is_running is False
    assert channel._app is None
    await channel.stop()

@pytest.mark.asyncio
async def test_reasoning_delta_non_string_is_ignored() -> None:
    """Regression: a non-string reasoning delta (e.g. datetime from the
    provider) must not crash the stream with a TypeError; it is ignored
    and the accumulated reasoning is preserved."""
    from datetime import datetime

    channel = TelegramChannel(
        TelegramConfig(
            enabled=True, token="123:abc", allow_from=["*"],
            stream_edit_interval=0.1,
        ),
        MessageBus(),
    )
    channel._app = _FakeApp(lambda: None)
    channel._app.bot.do_api_request = AsyncMock()

    await channel.send_reasoning_delta(
        "123", "pensando ", metadata={"is_group": True}, stream_id="s:0"
    )
    await asyncio.sleep(0.15)  # permite el edit del preview inicial
    # Delta no-string: debe ignorarse sin crashear (TypeError antes del fix).
    await channel.send_reasoning_delta(
        "123", datetime.now(), metadata={"is_group": True}, stream_id="s:0"
    )
    await asyncio.sleep(0.15)
    await channel.send_reasoning_delta(
        "123", "más", metadata={"is_group": True}, stream_id="s:0"
    )

    assert channel._stream_bufs["123"].reasoning == "pensando más"
    # El preview legacy se envió una sola vez (primer delta); el delta
    # no-string se ignoró y el siguiente delta siguió acumulando sin crash.
    assert len(channel._app.bot.sent_messages) == 1
    assert "<blockquote expandable>" in channel._app.bot.sent_messages[0]["text"]
