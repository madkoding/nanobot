"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReactionTypeEmoji,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    ReplyParameters,
    Update,
)
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    PollAnswerHandler,
    filters,
)
from telegram.request import HTTPXRequest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel, TypingIndicator
from nanobot.command.builtin import build_help_text
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base
from nanobot.security.network import validate_url_target
from nanobot.utils.helpers import split_message

TELEGRAM_RICH_MAX_LEN = 30000
# Límite seguro para el reasoning acumulado: el blockquote expandible se
# edita in-place con parse_mode=HTML, así que el texto escapado + tags debe
# caber en el límite real de Telegram (4096 chars). 3500 deja margen para
# la expansión HTML (& < >) y los tags del blockquote.
TELEGRAM_REASONING_MAX_LEN = 3500
_DRAFT_TTL_SECONDS = 25.0  # margen bajo el límite de 30 s de sendRichMessageDraft


# Markdown rendering and chunking helpers live in telegram/markdown.py;
# re-export so existing import sites keep working unchanged.
from nanobot.channels.telegram.markdown import (  # noqa: E402
    TELEGRAM_HTML_MAX_LEN,
    TELEGRAM_MAX_MESSAGE_LEN,
    _escape_telegram_html,
    _markdown_to_telegram_html,
    _split_telegram_markdown,
    _split_telegram_markdown_html,
    _split_telegram_markdown_html_chunks,
    _strip_md_block,
    _tool_hint_to_telegram_blockquote,
)

TELEGRAM_REPLY_CONTEXT_MAX_LEN = TELEGRAM_MAX_MESSAGE_LEN  # Max length for reply context in user message


_SEND_MAX_RETRIES = 3
_SEND_RETRY_BASE_DELAY = 0.5  # seconds, doubled each retry
_STREAM_EDIT_INTERVAL_DEFAULT = 0.6  # min seconds between edit_message_text calls
_STREAM_BUFFER_TTL_SECONDS = 120.0  # max lifetime for an orphan streaming buffer


@dataclass
class _StreamBuf:
    """Per-chat streaming accumulator for progressive message editing."""
    text: str = ""
    message_id: int | None = None
    last_edit: float = 0.0
    stream_id: str | None = None
    draft_id: int | None = None  # sendRichMessageDraft id (rich streaming)
    reasoning: str = ""  # razonamiento acumulado (thinking blocks)
    using_draft: bool = False  # el stream vive en un sendRichMessageDraft
    draft_expires_at: float = 0.0  # monotonic deadline; pasado → fallback legacy
    reasoning_open: bool = False  # segmento de reasoning activo
    created_at: float = 0.0
    last_activity: float = 0.0

    def __post_init__(self) -> None:
        now = time.monotonic()
        if self.created_at == 0.0:
            self.created_at = now
        if self.last_activity == 0.0:
            self.last_activity = now


@dataclass
class _QueuedTelegramUpdate:
    """Telegram update staged for per-session ordered processing."""

    kind: Literal["command", "message"]
    update: Update
    context: Any
    sort_key: tuple[int, int]


class TelegramConfig(Base):
    """Telegram channel configuration."""

    enabled: bool = False
    token: str = ""
    mode: Literal["polling", "webhook"] = "polling"
    allow_from: list[str] = Field(default_factory=list)
    proxy: str | None = None
    reply_to_message: bool = False
    react_emoji: str = "👀"
    group_policy: Literal["open", "mention"] = "mention"
    connection_pool_size: int = 32
    pool_timeout: float = 5.0
    streaming: bool = True
    # Enable inline keyboard buttons in Telegram messages.
    inline_keyboards: bool = False
    # Opt in to Bot API 10.1 sendRichMessage for richer markdown rendering.
    rich_messages: bool = False
    stream_edit_interval: float = Field(default=_STREAM_EDIT_INTERVAL_DEFAULT, ge=0.1)
    webhook_url: str = ""
    webhook_listen_host: str = "127.0.0.1"
    webhook_listen_port: int = Field(default=8081, ge=1, le=65535)
    webhook_path: str = "/telegram"
    webhook_secret_token: str = ""
    webhook_max_connections: int = Field(default=4, ge=1, le=100)
    # Efecto de mensaje por defecto (message_effect_id) para celebración
    # automática (aprobaciones, tareas completadas). None → sin efecto.
    message_effect_id: str | None = None

    @field_validator("webhook_path")
    @classmethod
    def webhook_path_must_start_with_slash(cls, value: str) -> str:
        value = value.strip() or "/telegram"
        if not value.startswith("/"):
            raise ValueError('webhook_path must start with "/"')
        return value

    @model_validator(mode="after")
    def validate_webhook_config(self) -> "TelegramConfig":
        if self.mode != "webhook":
            return self

        url = self.webhook_url.strip()
        if not url:
            raise ValueError("webhook_url is required when Telegram mode is webhook")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("webhook_url must be a public HTTPS URL")
        secret = self.webhook_secret_token.strip()
        if not secret:
            raise ValueError("webhook_secret_token is required when Telegram mode is webhook")
        if len(secret) > 256 or re.match(r"^[A-Za-z0-9_-]+$", secret) is None:
            raise ValueError(
                "webhook_secret_token must be 1-256 characters using only A-Z, a-z, 0-9, _ and -"
            )
        return self


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling or webhook mode.

    Long polling is the default. Webhook mode requires a public HTTPS URL and a
    Telegram secret token.
    """

    name = "telegram"
    display_name = "Telegram"

    # Commands registered with Telegram's command menu
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("stop", "Stop the current task"),
        BotCommand("restart", "Restart the bot"),
        BotCommand("status", "Show bot status"),
        BotCommand("history", "Show recent conversation messages"),
        BotCommand("goal", "Start a sustained objective (long-running task)"),
        BotCommand("trigger", "Create a named local trigger"),
        BotCommand("pairing", "Manage DM pairing (approve/deny/list)"),
        BotCommand("model", "Switch runtime model preset"),
        BotCommand("skill", "List enabled skills"),
        BotCommand("dream", "Run Dream memory consolidation now"),
        BotCommand("dream_log", "Show the latest Dream memory change"),
        BotCommand("dream_restore", "Restore Dream memory to an earlier version"),
        BotCommand("dream_prompt", "Tell Dream how to organize memory"),
        BotCommand("help", "Show available commands"),
    ]

    # Regex for slash commands routed to AgentLoop via ``_forward_command``.
    # Hyphenated ``dream-*`` commands stay on a separate handler (below).
    TELEGRAM_BUS_SLASH_COMMAND_RE = re.compile(
        r"^/(?:new|stop|restart|status|dream|history|goal|trigger|pairing|model|skill)(?:@\w+)?(?:\s+.*)?$"
    )

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return TelegramConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = TelegramConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._typing = TypingIndicator(interval=4.0)
        self._media_group_buffers: dict[str, dict] = {}
        self._media_group_tasks: dict[str, asyncio.Task] = {}
        self._message_threads: dict[tuple[str, int], int] = {}
        self._bot_user_id: int | None = None
        self._bot_username: str | None = None
        self._stream_bufs: dict[str, _StreamBuf] = {}  # chat_id -> streaming state
        self._reply_anchored: set[int] = set()  # chat_id -> ya se ancló la conversación
        self._inbound_buffers: dict[str, list[_QueuedTelegramUpdate]] = {}
        self._inbound_workers: dict[str, asyncio.Task] = {}
        self._rich_send_disabled: bool = False  # Latch off if Bot API < 10.1
        self._draft_counter: int = 0  # draft_id estables por stream (sendRichMessageDraft)
        # Reply keyboard / menu commands staged by the message tool during
        # a streaming turn; applied to the consolidated final message.
        self._pending_stream_reply_keyboard: dict[str, list[list[str]]] = {}
        self._pending_stream_menu_commands: dict[str, list[dict]] = {}
        # Task lists rich administradas por el agente: chat_id -> message_id de
        # la última task list enviada (para updates in-place con checklist_update).
        self._task_lists: dict[str, int] = {}
        # Polls nativos: poll_id -> {chat_id, options} para resolver poll_answer.
        self._polls_cache: dict[str, dict] = {}
        self._stream_sweep_task: asyncio.Task | None = None

    def is_allowed(self, sender_id: str) -> bool:
        """Preserve Telegram's legacy id|username allowlist matching."""
        if super().is_allowed(sender_id):
            return True

        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list or "*" in allow_list:
            return False

        sender_str = str(sender_id)
        if sender_str.count("|") != 1:
            return False

        sid, username = sender_str.split("|", 1)
        if not sid.isdigit() or not username:
            return False

        return sid in allow_list or username in allow_list

    def _reply_params_for(self, chat_id: int, reply_to_message_id) -> ReplyParameters | None:
        """Decide si un mensaje de salida lleva quote (reply_parameters).

        - reply_to_message_id ausente → None (sin quote, sin tocar el ancla).
        - config.reply_to_message=True → siempre quote (opt-in explícito).
        - default → quote solo en el primer mensaje de la conversación (ancla).
        """
        if not reply_to_message_id:
            return None
        if self.config.reply_to_message:
            return ReplyParameters(
                message_id=int(reply_to_message_id),
                allow_sending_without_reply=True,
            )
        if chat_id in self._reply_anchored:
            return None
        self._reply_anchored.add(chat_id)
        return ReplyParameters(
            message_id=int(reply_to_message_id),
            allow_sending_without_reply=True,
        )

    @staticmethod
    def _normalize_telegram_command(content: str) -> str:
        """Map Telegram-safe command aliases back to canonical nanobot commands."""
        if not content.startswith("/"):
            return content
        if content == "/dream_log" or content.startswith("/dream_log "):
            return content.replace("/dream_log", "/dream-log", 1)
        if content == "/dream_restore" or content.startswith("/dream_restore "):
            return content.replace("/dream_restore", "/dream-restore", 1)
        if content == "/dream_prompt" or content.startswith("/dream_prompt "):
            return content.replace("/dream_prompt", "/dream-prompt", 1)
        return content

    async def start(self) -> None:
        """Start the Telegram bot."""
        if not self.config.token:
            self.logger.error("bot token not configured")
            return

        self._running = True

        proxy = self.config.proxy or None

        # Separate pools so long-polling (getUpdates) never starves outbound sends.
        api_request = HTTPXRequest(
            connection_pool_size=self.config.connection_pool_size,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        poll_request = HTTPXRequest(
            connection_pool_size=4,
            pool_timeout=self.config.pool_timeout,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=proxy,
        )
        builder = (
            Application.builder()
            .token(self.config.token)
            .request(api_request)
            .get_updates_request(poll_request)
        )
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)

        # Add command handlers (using Regex to support @username suffixes before bot initialization)
        self._app.add_handler(MessageHandler(filters.Regex(r"^/start(?:@\w+)?$"), self._on_start))
        self._app.add_handler(
            MessageHandler(
                filters.Regex(TelegramChannel.TELEGRAM_BUS_SLASH_COMMAND_RE),
                self._forward_command,
            )
        )
        self._app.add_handler(
            MessageHandler(
                filters.Regex(
                    r"^/(dream-log|dream_log|dream-restore|dream_restore|dream-prompt|dream_prompt)(?:@\w+)?(?:\s+.*)?$"
                ),
                self._forward_command,
            )
        )
        self._app.add_handler(MessageHandler(filters.Regex(r"^/help(?:@\w+)?$"), self._on_help))

        # Add message handler for text, photos, video, voice, documents, and locations
        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.VIDEO_NOTE
                 | filters.ANIMATION | filters.VOICE | filters.AUDIO
                 | filters.Document.ALL | filters.LOCATION)
                & ~filters.COMMAND,
                self._on_message
            )
        )

        # Conditionally register inline keyboard callback handler
        if self.config.inline_keyboards:
            self._app.add_handler(CallbackQueryHandler(self._on_callback_query))
            allowed_updates = ["message", "callback_query"]
            self.logger.debug("inline keyboards enabled")
        else:
            allowed_updates = ["message"]

        # Polls nativos: el voto del usuario llega como poll_answer.
        self._app.add_handler(PollAnswerHandler(self._on_poll_answer))
        if "poll_answer" not in allowed_updates:
            allowed_updates.append("poll_answer")

        if self.config.mode == "webhook":
            self.logger.info("Starting bot (webhook mode)...")
        else:
            self.logger.info("Starting bot (polling mode)...")

        # Initialize and start receiving updates
        try:
            await self._app.initialize()
            await self._app.start()
        except BaseException:
            # Limpiar estado: un start() fallido (p.ej. TimedOut en getMe
            # durante initialize) no debe dejar el canal reportando
            # is_running=True ni un _app a medio inicializar — el watchdog
            # reintentaría sobre un objeto roto y stop() lanzaría
            # RuntimeError("This Updater is not running!").
            self._running = False
            self._app = None
            raise

        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        self.logger.info("bot @{} connected", bot_info.username)

        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            self.logger.debug("bot commands registered")
        except Exception as e:
            self.logger.warning("Failed to register bot commands: {}", e)

        self._stream_sweep_task = asyncio.create_task(self._stream_sweep_loop())

        if self.config.mode == "webhook":
            # ``url_path`` is the local HTTP route. ``webhook_url`` is the
            # public HTTPS URL Telegram calls; reverse proxies may rewrite it.
            await self._app.updater.start_webhook(
                listen=self.config.webhook_listen_host,
                port=self.config.webhook_listen_port,
                url_path=self.config.webhook_path.lstrip("/"),
                webhook_url=self.config.webhook_url.strip(),
                allowed_updates=allowed_updates,
                drop_pending_updates=False,
                secret_token=self.config.webhook_secret_token.strip(),
                max_connections=self.config.webhook_max_connections,
            )
        else:
            # Start polling (this runs until stopped)
            await self._app.updater.start_polling(
                allowed_updates=allowed_updates,
                drop_pending_updates=False,  # Process pending messages on startup
                error_callback=self._on_polling_error,
            )

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False

        # Cancel all typing indicators
        self._typing.stop_all()

        for task in self._media_group_tasks.values():
            task.cancel()
        self._media_group_tasks.clear()
        self._media_group_buffers.clear()

        for task in self._inbound_workers.values():
            task.cancel()
        self._inbound_workers.clear()
        self._inbound_buffers.clear()

        if self._stream_sweep_task is not None:
            self._stream_sweep_task.cancel()
            self._stream_sweep_task = None

        if self._app:
            self.logger.info("Stopping bot...")
            # El updater puede no haber arrancado si initialize() falló a
            # mitad de camino; PTB lanza RuntimeError("This Updater is not
            # running!") en ese caso. stop() debe ser un no-op seguro.
            try:
                await self._app.updater.stop()
            except RuntimeError:
                self.logger.debug("Updater was not running; skipping updater.stop()")
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    @staticmethod
    def _get_media_type(path: str) -> str:
        """Guess media type from file extension."""
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in ("jpg", "jpeg", "png", "gif", "webp"):
            return "photo"
        if ext in ("mp4", "mov", "avi", "mkv", "webm", "3gp"):
            return "video"
        if ext == "ogg":
            return "voice"
        if ext in ("mp3", "m4a", "wav", "aac"):
            return "audio"
        return "document"

    @staticmethod
    def _is_remote_media_url(path: str) -> bool:
        return path.startswith(("http://", "https://"))

    # Efecto de mensaje (Bot API 10.2): opt-in. Solo se aplica cuando el
    # agente pide un efecto explícito (effect=...) o el canal configura
    # message_effect_id. Sin override ni config → sin efecto.
    _MESSAGE_EFFECT_CONFETI = "5046509860389126442"
    _MESSAGE_EFFECTS: dict[str, str] = {
        "confeti": _MESSAGE_EFFECT_CONFETI,
        "confetti": _MESSAGE_EFFECT_CONFETI,
    }

    @classmethod
    def _resolve_message_effect(cls, effect: str | None) -> str | None:
        """Resolve a named effect to its message_effect_id (or pass through an id).

        None (sin override ni config) → sin efecto (opt-in, REQ-001).
        """
        if not effect:
            return None
        return cls._MESSAGE_EFFECTS.get(effect.lower(), effect)

    @staticmethod
    def _is_rich_capability_error(exc: Exception) -> bool:
        """True when the error indicates sendRichMessage is unavailable."""
        err = str(exc).lower()
        return (
            "method not found" in err
            or "unknown method" in err
            or "bad request: invalid parameter" in err
        )

    async def _try_send_rich(
        self,
        chat_id: int,
        content: str,
        reply_params=None,
        thread_kwargs: dict | None = None,
        reply_markup=None,
        *,
        is_ephemeral: bool = False,
        receiver_user_id: int | None = None,
        reply_keyboard_markup=None,
        draft_id: int | None = None,
        message_effect_id: str | None = None,
    ) -> bool:
        """Attempt sendRichMessage (Bot API 10.1). Returns True on success.

        Content longer than TELEGRAM_RICH_MAX_LEN is split into rich chunks
        (the rich limit is 32768 chars, well above the legacy 4096).
        """
        if not self._app:
            return False

        chunks = _split_telegram_markdown(content, TELEGRAM_RICH_MAX_LEN)
        for i, chunk in enumerate(chunks):
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "rich_message": {
                    "markdown": chunk,
                },
            }
            if message_effect_id is not None and i == len(chunks) - 1:
                # Efecto de mensaje (Bot API 10.2) solo en el último chunk.
                payload["message_effect_id"] = message_effect_id
            if draft_id is not None and i == len(chunks) - 1:
                # Reemplaza el draft efímero del stream por el mensaje final
                # (mismo draft_id → Telegram lo sustituye en vez de dejarlo
                # congelado como preview separado).
                payload["draft_id"] = draft_id
            if reply_params is not None:
                # sendRichMessage uses reply_parameters (object), not reply_to_message_id.
                if hasattr(reply_params, "message_id"):
                    payload["reply_parameters"] = {
                        "message_id": reply_params.message_id,
                        "allow_sending_without_reply": True,
                    }
                else:
                    payload["reply_parameters"] = reply_params
            if thread_kwargs:
                payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
            if reply_markup is not None:
                payload["reply_markup"] = reply_markup
            elif reply_keyboard_markup is not None and i == len(chunks) - 1:
                # Reply keyboard (ReplyKeyboardMarkup) solo en el último chunk.
                payload["reply_markup"] = reply_keyboard_markup
            if is_ephemeral:
                payload["is_ephemeral"] = True
                if receiver_user_id is not None:
                    payload["receiver_user_id"] = receiver_user_id

            try:
                await self._call_with_retry(
                    self._app.bot.do_api_request,
                    "sendRichMessage",
                    api_kwargs=payload,
                )
            except BadRequest as exc:
                if self._is_rich_capability_error(exc):
                    self.logger.debug("sendRichMessage not available, disabling")
                    self._rich_send_disabled = True
                elif message_effect_id is not None and "effect" in str(exc).lower():
                    # Efecto no soportado (grupos/servidor viejo): reintento
                    # sin efecto (best-effort, sin latch).
                    self.logger.debug("message_effect_id rejected, retrying without it: {}", exc)
                    return await self._try_send_rich(
                        chat_id, content, reply_params, thread_kwargs, reply_markup,
                        is_ephemeral=is_ephemeral,
                        receiver_user_id=receiver_user_id,
                        reply_keyboard_markup=reply_keyboard_markup,
                        draft_id=draft_id,
                        message_effect_id=None,
                    )
                else:
                    self.logger.debug("sendRichMessage rejected: {}", exc)
                return False
            except Exception as exc:
                err_str = str(exc).lower()
                is_timeout = "timed out" in err_str or isinstance(exc, TimedOut)
                if is_timeout:
                    self.logger.debug("sendRichMessage timeout, falling back to legacy path")
                else:
                    self.logger.debug("sendRichMessage failed: {}", exc)
                return False
        return True

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram."""
        if not self._app:
            raise RuntimeError("bot not running")

        progress_event = msg.event if isinstance(msg.event, ProgressEvent) else None

        # Only stop typing indicator and remove reaction for final responses
        if progress_event is None:
            self._stop_typing(msg.chat_id)
            if reply_to_message_id := msg.metadata.get("message_id"):
                with suppress(ValueError):
                    await self._remove_reaction(msg.chat_id, int(reply_to_message_id))

        try:
            chat_id = int(msg.chat_id)
        except ValueError:
            self.logger.exception("Invalid chat_id: {}", msg.chat_id)
            return
        reply_to_message_id = msg.metadata.get("message_id")
        message_thread_id = msg.metadata.get("message_thread_id")
        if message_thread_id is None and reply_to_message_id is not None:
            message_thread_id = self._message_threads.get((msg.chat_id, reply_to_message_id))
        thread_kwargs = {}
        if message_thread_id is not None:
            thread_kwargs["message_thread_id"] = message_thread_id

        reply_params = self._reply_params_for(chat_id, reply_to_message_id)

        # Task list rich (checkboxes nativos) — el agente la administra.
        checklist = getattr(msg, "checklist", None)
        if checklist:
            await self._send_task_list(
                chat_id, checklist, reply_params, thread_kwargs,
                message_effect_id=self._resolve_message_effect(
                    getattr(msg, "effect", None) or self.config.message_effect_id
                ),
            )
            return

        # Edición in-place de una task list existente (progreso).
        checklist_update = getattr(msg, "checklist_update", None)
        if checklist_update:
            await self._update_task_list(chat_id, checklist_update)
            return

        # Poll nativo (decisiones visibles, respuesta única).
        poll = getattr(msg, "poll", None)
        if poll:
            await self._send_poll(chat_id, poll, reply_params, thread_kwargs)
            return

        # Send media files
        for media_path in (msg.media or []):
            try:
                media_type = self._get_media_type(media_path)
                sender = {
                    "photo": self._app.bot.send_photo,
                    "video": self._app.bot.send_video,
                    "voice": self._app.bot.send_voice,
                    "audio": self._app.bot.send_audio,
                }.get(media_type, self._app.bot.send_document)
                param = {
                    "photo": "photo",
                    "video": "video",
                    "voice": "voice",
                    "audio": "audio",
                }.get(media_type, "document")
                extra: dict[str, Any] = {}
                if media_type == "video":
                    extra["supports_streaming"] = True

                # Telegram Bot API accepts HTTP(S) URLs directly for media params.
                if self._is_remote_media_url(media_path):
                    ok, error = validate_url_target(media_path)
                    if not ok:
                        raise ValueError(f"unsafe media URL: {error}")
                    await self._call_with_retry(
                        sender,
                        chat_id=chat_id,
                        **{param: media_path},
                        reply_parameters=reply_params,
                        **thread_kwargs,
                        **extra,
                    )
                    continue

                media_bytes = Path(media_path).read_bytes()
                filename = Path(media_path).name
                send_kwargs = {param: media_bytes, "filename": filename}
                await self._call_with_retry(
                    sender,
                    chat_id=chat_id,
                    reply_parameters=reply_params,
                    **thread_kwargs,
                    **extra,
                    **send_kwargs,
                )
            except Exception:
                filename = media_path.rsplit("/", 1)[-1]
                self.logger.exception("Failed to send media {}", media_path)
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=f"[Failed to send: {filename}]",
                    reply_parameters=reply_params,
                    **thread_kwargs,
                )

        # Send text content
        if msg.content and msg.content != "[empty message]":
            render_as_blockquote = bool(progress_event and progress_event.tool_hint)
            buttons = getattr(msg, "buttons", None) or []
            reply_markup = self._build_keyboard(buttons) if buttons else None
            text = msg.content
            # Fallback: no native keyboard → splice labels into the message so the choices survive.
            if buttons and reply_markup is None:
                text = f"{text}\n\n{self._buttons_as_text(buttons)}"

            # Efecto de mensaje (Bot API 10.2): override por mensaje o default
            # de config (confeti). Best-effort: si el servidor lo rechaza, se
            # reintenta sin efecto (sin latch).
            effect = getattr(msg, "effect", None) or self.config.message_effect_id
            message_effect_id = self._resolve_message_effect(effect)

            # Comandos dinámicos por chat (setMyCommands con scope) — best-effort.
            menu_commands = getattr(msg, "menu_commands", None) or []
            if menu_commands:
                await self._set_chat_menu_commands(chat_id, menu_commands)

            # Reply keyboard (teclado de respuesta) — solo en el último chunk.
            # Una lista vacía ([]) remueve el teclado previo (ReplyKeyboardRemove).
            reply_keyboard = getattr(msg, "reply_keyboard", None)
            reply_markup_final = None
            if reply_keyboard:
                reply_markup_final = self._build_reply_keyboard(reply_keyboard)
            elif reply_keyboard is not None:
                # reply_keyboard=[] explícito → quitar el teclado pegado.
                reply_markup_final = self._build_reply_keyboard_remove()

            # Ephemeral (Bot API 10.2): visible solo para un usuario en grupos.
            ephemeral = bool(getattr(msg, "ephemeral", False))
            receiver_user_id = None
            if ephemeral:
                try:
                    receiver_user_id = int(msg.metadata.get("user_id", 0) or 0) or None
                except (TypeError, ValueError):
                    receiver_user_id = None

            # Bot API 10.1 rich fast-path: send raw markdown via sendRichMessage.
            # All non-blockquote content tries rich first; _rich_send_disabled
            # latches off permanently if the server doesn't support it.
            if (
                not render_as_blockquote
                and self.config.rich_messages
                and not getattr(self, "_rich_send_disabled", False)
            ):
                rich_ok = await self._try_send_rich(
                    chat_id, text, reply_params, thread_kwargs, reply_markup,
                    is_ephemeral=ephemeral,
                    receiver_user_id=receiver_user_id,
                    reply_keyboard_markup=reply_markup_final,
                    message_effect_id=message_effect_id,
                )
                if rich_ok:
                    return

            chunks = _split_telegram_markdown(text, TELEGRAM_MAX_MESSAGE_LEN)
            for i, chunk in enumerate(chunks):
                is_last = (i == len(chunks) - 1)
                await self._send_text(
                    chat_id, chunk, reply_params, thread_kwargs,
                    render_as_blockquote=render_as_blockquote,
                    reply_markup=reply_markup if is_last else None,
                    is_ephemeral=ephemeral if is_last else False,
                    receiver_user_id=receiver_user_id if is_last else None,
                    reply_keyboard_markup=reply_markup_final if is_last else None,
                    message_effect_id=message_effect_id if is_last else None,
                )

    async def _send_task_list(
        self,
        chat_id: int,
        checklist: dict,
        reply_params=None,
        thread_kwargs: dict | None = None,
        *,
        message_effect_id: str | None = None,
    ) -> None:
        """Send a rich task list (native checkboxes) via sendRichMessage.

        The agent manages the list: the message_id is registered per chat so
        later checklist_update calls can edit it in place.
        """
        title = checklist.get("title", "")
        tasks = checklist.get("tasks") or []
        lines = [f"# {title}", ""] if title else []
        lines += [f"- [ ] {task}" for task in tasks]
        markdown = "\n".join(lines)
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "rich_message": {"markdown": markdown},
        }
        if message_effect_id is not None:
            payload["message_effect_id"] = message_effect_id
        if reply_params is not None:
            if hasattr(reply_params, "message_id"):
                payload["reply_parameters"] = {
                    "message_id": reply_params.message_id,
                    "allow_sending_without_reply": True,
                }
            else:
                payload["reply_parameters"] = reply_params
        if thread_kwargs:
            payload.update({k: v for k, v in thread_kwargs.items() if v is not None})
        try:
            result = await self._call_with_retry(
                self._app.bot.do_api_request,
                "sendRichMessage",
                api_kwargs=payload,
            )
        except BadRequest as exc:
            if self._is_rich_capability_error(exc):
                self.logger.debug("sendRichMessage not available, disabling")
                self._rich_send_disabled = True
            elif message_effect_id is not None and "effect" in str(exc).lower():
                self.logger.debug("message_effect_id rejected, retrying without it: {}", exc)
                await self._send_task_list(
                    chat_id, checklist, reply_params, thread_kwargs,
                    message_effect_id=None,
                )
                return
            else:
                self.logger.warning("sendRichMessage rejected for task list: {}", exc)
            return
        except Exception as exc:
            self.logger.warning("sendRichMessage failed for task list: {}", exc)
            return
        # do_api_request devuelve un dict (resultado crudo de la API), no un
        # objeto PTB: soportar ambos para extraer el message_id.
        if isinstance(result, dict):
            message_id = result.get("message_id")
        else:
            message_id = getattr(result, "message_id", None)
        if message_id is not None:
            self._task_lists[str(chat_id)] = {
                "message_id": message_id,
                "tasks": tasks,
            }

    async def _update_task_list(self, chat_id: int, checklist_update: dict) -> None:
        """Edit a task list in place, marking done tasks and showing progress."""
        message_id = checklist_update.get("message_id")
        done = set(checklist_update.get("done") or [])
        registered = self._task_lists.get(str(chat_id))
        tasks: list[str] = []
        if registered and registered.get("message_id") == message_id:
            tasks = registered.get("tasks") or []
        if not tasks:
            self.logger.warning(
                "checklist_update for unknown task list {} in chat {}",
                message_id, chat_id,
            )
            return
        lines = [f"- [x] {task}" if i in done else f"- [ ] {task}" for i, task in enumerate(tasks)]
        done_count = sum(1 for i in range(len(tasks)) if i in done)
        summary = f"✅ {done_count}/{len(tasks)} tareas completadas"
        markdown = "\n".join(lines) + f"\n\n{summary}"
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "rich_message": {"markdown": markdown},
        }
        try:
            await self._call_with_retry(
                self._app.bot.do_api_request,
                "editMessageText",
                api_kwargs=payload,
            )
        except BadRequest as exc:
            if self._is_rich_capability_error(exc):
                self.logger.debug("editMessageText rich not available, disabling")
                self._rich_send_disabled = True
            else:
                self.logger.warning("editMessageText rejected for task list: {}", exc)
        except Exception as exc:
            self.logger.warning("editMessageText failed for task list: {}", exc)

    async def _send_poll(
        self,
        chat_id: int,
        poll: dict,
        reply_params=None,
        thread_kwargs: dict | None = None,
    ) -> None:
        """Send a native poll (visible results, single answer) and cache it."""
        question = poll.get("question", "")
        options = poll.get("options") or []
        try:
            result = await self._call_with_retry(
                self._app.bot.send_poll,
                chat_id=chat_id,
                question=question,
                options=options,
                is_anonymous=False,
                allows_multiple_answers=False,
                reply_parameters=reply_params,
                **thread_kwargs,
            )
        except Exception as exc:
            self.logger.warning("send_poll failed: {}", exc)
            return
        poll_obj = getattr(result, "poll", None)
        poll_id = getattr(poll_obj, "id", None)
        if poll_id:
            self._polls_cache[poll_id] = {
                "chat_id": chat_id,
                "options": options,
            }
            # Limpieza básica: mantener el cache acotado.
            if len(self._polls_cache) > 200:
                oldest = next(iter(self._polls_cache))
                self._polls_cache.pop(oldest, None)

    async def _call_with_retry(self, fn, *args, **kwargs):
        """Call an async Telegram API function with retry on pool/network timeout and RetryAfter."""
        from telegram.error import RetryAfter

        for attempt in range(1, _SEND_MAX_RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except TimedOut:
                if attempt == _SEND_MAX_RETRIES:
                    raise
                delay = _SEND_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                self.logger.warning(
                    "timeout (attempt {}/{}), retrying in {:.1f}s",
                    attempt, _SEND_MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
            except RetryAfter as e:
                if attempt == _SEND_MAX_RETRIES:
                    raise
                delay = float(e.retry_after)
                self.logger.warning(
                    "Flood Control (attempt {}/{}), retrying in {:.1f}s",
                    attempt, _SEND_MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        reply_params=None,
        thread_kwargs: dict | None = None,
        render_as_blockquote: bool = False,
        reply_markup=None,
        *,
        is_ephemeral: bool = False,
        receiver_user_id: int | None = None,
        reply_keyboard_markup=None,
        message_effect_id: str | None = None,
    ) -> None:
        """Send a plain text message with HTML fallback."""
        markup = reply_markup if reply_markup is not None else reply_keyboard_markup
        html = _tool_hint_to_telegram_blockquote(text) if render_as_blockquote else _markdown_to_telegram_html(text)
        send_kwargs: dict[str, Any] = {
            "chat_id": chat_id,
            "text": html,
            "parse_mode": "HTML",
            "reply_parameters": reply_params,
            "reply_markup": markup,
            **(thread_kwargs or {}),
        }
        if message_effect_id is not None:
            send_kwargs["message_effect_id"] = message_effect_id
        if is_ephemeral:
            send_kwargs["is_ephemeral"] = True
            if receiver_user_id is not None:
                send_kwargs["receiver_user_id"] = receiver_user_id
        try:
            await self._call_with_retry(
                self._app.bot.send_message,
                **send_kwargs,
            )
        except BadRequest as e:
            # Efecto no soportado (grupos/servidor viejo): reintento sin efecto.
            if message_effect_id is not None and "effect" in str(e).lower():
                self.logger.debug("message_effect_id rejected, retrying without it: {}", e)
                await self._send_text(
                    chat_id, text, reply_params, thread_kwargs,
                    render_as_blockquote=render_as_blockquote,
                    reply_markup=reply_markup,
                    is_ephemeral=is_ephemeral,
                    receiver_user_id=receiver_user_id,
                    reply_keyboard_markup=reply_keyboard_markup,
                    message_effect_id=None,
                )
                return
            # Ephemeral no soportado (Bot API < 10.2): reintentar sin ephemeral.
            if is_ephemeral and "ephemeral" in str(e).lower():
                self.logger.debug("is_ephemeral not supported, retrying without it: {}", e)
                await self._send_text(
                    chat_id, text, reply_params, thread_kwargs,
                    render_as_blockquote=render_as_blockquote,
                    reply_markup=reply_markup,
                    is_ephemeral=False,
                    receiver_user_id=None,
                    reply_keyboard_markup=reply_keyboard_markup,
                    message_effect_id=message_effect_id,
                )
                return
            self.logger.warning("HTML parse failed, falling back to plain text: {}", e)
            try:
                plain_kwargs: dict[str, Any] = {
                    "chat_id": chat_id,
                    "text": text,
                    "reply_parameters": reply_params,
                    "reply_markup": markup,
                    **(thread_kwargs or {}),
                }
                if message_effect_id is not None:
                    plain_kwargs["message_effect_id"] = message_effect_id
                if is_ephemeral:
                    plain_kwargs["is_ephemeral"] = True
                    if receiver_user_id is not None:
                        plain_kwargs["receiver_user_id"] = receiver_user_id
                await self._call_with_retry(
                    self._app.bot.send_message,
                    **plain_kwargs,
                )
            except Exception:
                self.logger.exception("Error sending message")
                raise

    @staticmethod
    def _is_not_modified_error(exc: Exception) -> bool:
        return isinstance(exc, BadRequest) and "message is not modified" in str(exc).lower()

    @staticmethod
    def _is_message_too_long_error(exc: Exception) -> bool:
        """True si Telegram rechaza el payload por exceder el límite de 4096 chars."""
        return isinstance(exc, BadRequest) and "message is too long" in str(exc).lower()

    def _next_draft_id(self) -> int:
        """Return a stable draft_id for sendRichMessageDraft (per stream)."""
        self._draft_counter += 1
        return self._draft_counter

    @staticmethod
    def _stream_buf_key(chat_id: str, stream_id: str | None) -> str:
        """Key for the streaming accumulator.

        We keep ``chat_id`` as the dict key for backwards compatibility, but
        each buffer stores its ``stream_id``.  A new ``stream_id`` replaces
        any previous buffer for the same chat so aborted turns can never
        resume editing an older message.
        """
        return chat_id

    def _sweep_stream_buffers(self, now: float | None = None) -> None:
        """Discard streaming buffers that have been idle beyond the TTL."""
        if now is None:
            now = time.monotonic()
        stale = [
            key
            for key, buf in list(self._stream_bufs.items())
            if now - buf.last_activity > _STREAM_BUFFER_TTL_SECONDS
        ]
        for key in stale:
            self.logger.debug("Sweeping stale stream buffer for {}", key)
            self._stream_bufs.pop(key, None)

    def _reset_stream_buffer(
        self,
        chat_id: str,
        stream_id: str | None,
        *,
        reason: str = "new stream",
    ) -> _StreamBuf:
        """Start a fresh stream buffer for ``chat_id``.

        Any leftover buffer from a previous turn is discarded; the previous
        Telegram message simply stays frozen at whatever partial state it had.
        """
        key = self._stream_buf_key(chat_id, stream_id)
        old = self._stream_bufs.pop(key, None)
        if old is not None:
            self.logger.debug(
                "Resetting stream buffer for {} ({} → {}): {}",
                chat_id,
                old.stream_id,
                stream_id,
                reason,
            )
        buf = _StreamBuf(stream_id=stream_id)
        self._stream_bufs[key] = buf
        return buf

    async def _stream_sweep_loop(self) -> None:
        """Background task that periodically discards orphan stream buffers."""
        try:
            while self._running:
                await asyncio.sleep(30.0)
                self._sweep_stream_buffers()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning("Stream buffer sweep failed: {}", e)

    async def send_reasoning_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
    ) -> None:
        """Stream a chunk of model reasoning/thinking content.

        Rich + private chat → sendRichMessageDraft with <tg-thinking> (native
        Thinking Block, animated "Thinking…"). Otherwise → legacy preview with
        an expandable blockquote. The reasoning is accumulated in _StreamBuf
        and rendered as <details> in the final message.
        """
        if not self._app or not self.show_reasoning or not delta:
            return
        meta = metadata or {}
        int_chat_id = int(chat_id)
        key = self._stream_buf_key(chat_id, stream_id)
        self._sweep_stream_buffers()
        buf = self._stream_bufs.get(key)
        if buf is None or (stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id):
            buf = self._reset_stream_buffer(chat_id, stream_id, reason="new reasoning stream")
        elif buf.stream_id is None:
            buf.stream_id = stream_id
        # Protect the accumulator against non-string deltas (e.g. a provider
        # emitting a datetime/object token) — concatenating those would crash
        # the whole reasoning stream with a TypeError.
        if not isinstance(delta, str):
            self.logger.warning("Ignoring non-string reasoning delta: {!r}", delta)
            return
        buf.reasoning = (buf.reasoning + delta)[:TELEGRAM_REASONING_MAX_LEN]
        buf.reasoning_open = True

        now = time.monotonic()
        buf.last_activity = now
        rich_ok = (
            self.config.rich_messages
            and not getattr(self, "_rich_send_disabled", False)
            and not meta.get("is_group", False)
        )
        if rich_ok:
            if buf.draft_id is None:
                buf.draft_id = self._next_draft_id()
                buf.using_draft = True
                buf.draft_expires_at = now + _DRAFT_TTL_SECONDS
            elif now > buf.draft_expires_at:
                # Draft expirado: se autolimpia; switch a legacy.
                buf.using_draft = False
                buf.draft_id = None
                await self._send_legacy_preview(int_chat_id, buf, meta, {})
                return
            if (now - buf.last_edit) >= self.config.stream_edit_interval:
                payload: dict[str, Any] = {
                    "chat_id": int_chat_id,
                    "draft_id": buf.draft_id,
                    "rich_message": {"markdown": f"<tg-thinking>{buf.reasoning}</tg-thinking>"},
                }
                try:
                    await self._call_with_retry(
                        self._app.bot.do_api_request,
                        "sendRichMessageDraft",
                        api_kwargs=payload,
                    )
                    buf.last_edit = now
                except BadRequest as exc:
                    if self._is_rich_capability_error(exc):
                        self.logger.debug("sendRichMessageDraft not available, disabling")
                        self._rich_send_disabled = True
                    buf.using_draft = False
                    buf.draft_id = None
                    await self._send_legacy_preview(int_chat_id, buf, meta, {})
                except Exception as exc:
                    self.logger.debug("sendRichMessageDraft failed: {}", exc)
            return

        # Legacy: preview con blockquote expandible (se edita in-place).
        if buf.message_id is None:
            await self._send_legacy_preview(int_chat_id, buf, meta, {})
        elif (now - buf.last_edit) >= self.config.stream_edit_interval:
            try:
                await self._call_with_retry(
                    self._app.bot.edit_message_text,
                    chat_id=int_chat_id, message_id=buf.message_id,
                    text=self._reasoning_blockquote(buf.reasoning),
                    parse_mode="HTML",
                )
                buf.last_edit = now
            except Exception as e:
                if self._is_not_modified_error(e):
                    buf.last_edit = now
                    return
                if self._is_message_too_long_error(e):
                    # El reasoning acumulado excede el límite de Telegram
                    # (4096 chars). Truncarlo al máximo seguro y reintentar una
                    # sola vez; si sigue fallando, degradar a texto plano.
                    self.logger.warning(
                        "Reasoning edit too long ({} chars), truncating to {}: {}",
                        len(buf.reasoning), TELEGRAM_REASONING_MAX_LEN, e,
                    )
                    buf.reasoning = buf.reasoning[:TELEGRAM_REASONING_MAX_LEN]
                    try:
                        await self._call_with_retry(
                            self._app.bot.edit_message_text,
                            chat_id=int_chat_id, message_id=buf.message_id,
                            text=self._reasoning_blockquote(buf.reasoning),
                            parse_mode="HTML",
                        )
                        buf.last_edit = now
                        return
                    except Exception as e2:
                        if self._is_not_modified_error(e2):
                            buf.last_edit = now
                            return
                        self.logger.warning("Reasoning edit failed after truncation: {}", e2)
                        return
                self.logger.warning("Reasoning edit failed: {}", e)

    async def send_reasoning_end(
        self,
        chat_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
    ) -> None:
        """Mark the end of a reasoning stream segment.

        The draft/legacy preview keeps the accumulated thinking; the final
        message (stream_end) renders it as <details>.
        """
        key = self._stream_buf_key(chat_id, stream_id)
        buf = self._stream_bufs.get(key)
        if buf is not None:
            buf.reasoning_open = False
            buf.last_activity = time.monotonic()

    def _reasoning_blockquote(self, reasoning: str) -> str:
        """Render accumulated reasoning as an expandable blockquote (legacy)."""
        return f"<blockquote expandable>{_escape_telegram_html(reasoning)}</blockquote>" if reasoning else ""

    def _reasoning_details(self, reasoning: str) -> str:
        """Render accumulated reasoning as a collapsible <details> (rich final)."""
        if not reasoning:
            return ""
        return (
            "<details><summary>🧠 Razonamiento</summary>\n\n"
            f"{reasoning}\n\n</details>"
        )

    async def _send_legacy_preview(
        self,
        int_chat_id: int,
        buf: "_StreamBuf",
        meta: dict[str, Any],
        thread_kwargs: dict,
    ) -> None:
        """Send (or edit) the legacy streaming preview with reasoning blockquote."""
        now = time.monotonic()
        if buf.message_id is None:
            preview = _strip_md_block(buf.text)
            if buf.reasoning:
                preview = f"{self._reasoning_blockquote(buf.reasoning)}\n\n{preview}"
            preview_kwargs: dict[str, Any] = {
                "chat_id": int_chat_id,
                "text": preview,
                **thread_kwargs,
            }
            rp = self._reply_params_for(int_chat_id, meta.get("message_id"))
            if rp is not None:
                preview_kwargs["reply_parameters"] = {
                    "message_id": rp.message_id,
                    "allow_sending_without_reply": True,
                }
            try:
                sent = await self._call_with_retry(
                    self._app.bot.send_message,
                    **preview_kwargs,
                )
                buf.message_id = sent.message_id
                buf.last_edit = now
            except Exception as e:
                self.logger.warning("Stream initial send failed: {}", e)
                raise  # Let ChannelManager handle retry
        elif (now - buf.last_edit) >= self.config.stream_edit_interval:
            preview = _strip_md_block(buf.text)
            if buf.reasoning:
                preview = f"{self._reasoning_blockquote(buf.reasoning)}\n\n{preview}"
            try:
                await self._call_with_retry(
                    self._app.bot.edit_message_text,
                    chat_id=int_chat_id, message_id=buf.message_id,
                    text=preview,
                )
                buf.last_edit = now
            except Exception as e:
                if self._is_not_modified_error(e):
                    buf.last_edit = now
                    return
                self.logger.warning("Stream edit failed: {}", e)
                raise  # Let ChannelManager handle retry

    async def _finalize_stream(
        self,
        chat_id: str,
        buf: "_StreamBuf",
        int_chat_id: int,
        meta: dict[str, Any],
        thread_kwargs: dict,
        reply_keyboard_markup,
        staging_menu_commands: list[dict] | None,
    ) -> None:
        """Finalize a stream: fix the draft (draft_id) or edit the legacy
        preview in place, appending the reasoning as <details>."""
        raw_text = buf.text
        details = self._reasoning_details(buf.reasoning)
        final_markdown = f"{raw_text}\n\n{details}" if details else raw_text

        # Draft rich activo y no expirado → fijar con sendRichMessage(draft_id=...).
        if (
            buf.using_draft
            and buf.draft_id is not None
            and time.monotonic() <= buf.draft_expires_at
            and self.config.rich_messages
            and not getattr(self, "_rich_send_disabled", False)
        ):
            payload: dict[str, Any] = {
                "chat_id": int_chat_id,
                "draft_id": buf.draft_id,
                "rich_message": {"markdown": final_markdown},
                **thread_kwargs,
            }
            rp = self._reply_params_for(int_chat_id, meta.get("message_id"))
            if rp is not None:
                payload["reply_parameters"] = {
                    "message_id": rp.message_id,
                    "allow_sending_without_reply": True,
                }
            if reply_keyboard_markup is not None:
                payload["reply_markup"] = reply_keyboard_markup
            try:
                await self._call_with_retry(
                    self._app.bot.do_api_request,
                    "sendRichMessage",
                    api_kwargs=payload,
                )
                if staging_menu_commands:
                    await self._set_chat_menu_commands(int_chat_id, staging_menu_commands)
                self._stream_bufs.pop(chat_id, None)
                return
            except BadRequest as exc:
                if self._is_rich_capability_error(exc):
                    self.logger.debug("sendRichMessage not available, disabling")
                    self._rich_send_disabled = True
                else:
                    self.logger.debug("sendRichMessage rejected: {}", exc)
            except Exception as exc:
                self.logger.debug("sendRichMessage failed: {}", exc)
            # Fall through to legacy.

        # Rich final in-place (sin draft): editMessageText(rich_message=...).
        if (
            self.config.rich_messages
            and not getattr(self, "_rich_send_disabled", False)
            and buf.message_id is not None
        ):
            edit_kwargs: dict[str, Any] = {
                "chat_id": int_chat_id,
                "message_id": buf.message_id,
                "rich_message": {"markdown": final_markdown},
                **thread_kwargs,
            }
            if reply_keyboard_markup is not None:
                edit_kwargs["reply_markup"] = reply_keyboard_markup
            try:
                await self._call_with_retry(
                    self._app.bot.do_api_request,
                    "editMessageText",
                    api_kwargs=edit_kwargs,
                )
                if staging_menu_commands:
                    await self._set_chat_menu_commands(int_chat_id, staging_menu_commands)
                self._stream_bufs.pop(chat_id, None)
                return
            except BadRequest as exc:
                if self._is_rich_capability_error(exc):
                    self.logger.debug("editMessageText rich not available, disabling")
                    self._rich_send_disabled = True
                else:
                    self.logger.debug("editMessageText rich rejected: {}", exc)
            except Exception as exc:
                self.logger.debug("editMessageText rich failed: {}", exc)
            # Fall through to the legacy HTML edit path (edits in place).

        # Legacy path: edit existing streaming message with HTML.
        html_chunks = _split_telegram_markdown_html(raw_text, TELEGRAM_HTML_MAX_LEN)
        primary_html = html_chunks[0]
        extra_html_chunks = html_chunks[1:]
        if buf.reasoning:
            primary_html = f"{self._reasoning_blockquote(buf.reasoning)}\n\n{primary_html}"
        if buf.message_id is None:
            # No hay preview legacy (el stream vivía en el draft): enviar el
            # contenido final como mensaje nuevo (nunca se pierde texto).
            rp = self._reply_params_for(int_chat_id, meta.get("message_id"))
            send_kwargs: dict[str, Any] = {
                "chat_id": int_chat_id,
                "text": primary_html,
                "parse_mode": "HTML",
                "reply_markup": reply_keyboard_markup,
                **thread_kwargs,
            }
            if rp is not None:
                send_kwargs["reply_parameters"] = {
                    "message_id": rp.message_id,
                    "allow_sending_without_reply": True,
                }
            try:
                await self._call_with_retry(
                    self._app.bot.send_message,
                    **send_kwargs,
                )
            except Exception:
                # Fall back to _send_text which handles HTML→plain gracefully.
                await self._send_text(int_chat_id, primary_html)
            for extra_html_chunk in extra_html_chunks:
                try:
                    await self._call_with_retry(
                        self._app.bot.send_message,
                        chat_id=int_chat_id, text=extra_html_chunk,
                        parse_mode="HTML",
                        **thread_kwargs,
                    )
                except Exception:
                    await self._send_text(int_chat_id, extra_html_chunk)
            self._stream_bufs.pop(chat_id, None)
            return
        try:
            await self._call_with_retry(
                self._app.bot.edit_message_text,
                chat_id=int_chat_id, message_id=buf.message_id,
                text=primary_html, parse_mode="HTML",
                reply_markup=reply_keyboard_markup,
            )
        except BadRequest as e:
            # Only fall back to plain text on actual HTML parse/format errors.
            # Network errors (TimedOut, NetworkError) should propagate immediately
            # to avoid doubling connection demand during pool exhaustion.
            if self._is_not_modified_error(e):
                self.logger.debug("Final stream edit already applied for {}", chat_id)
                self._stream_bufs.pop(chat_id, None)
                return
            self.logger.debug("Final stream edit failed (HTML), trying plain: {}", e)
            # Fall back to raw markdown (not HTML) so users don't see raw tags.
            primary_plain = split_message(raw_text, TELEGRAM_MAX_MESSAGE_LEN)[0] if len(raw_text) > TELEGRAM_MAX_MESSAGE_LEN else raw_text
            try:
                await self._call_with_retry(
                    self._app.bot.edit_message_text,
                    chat_id=int_chat_id, message_id=buf.message_id,
                    text=primary_plain,
                )
            except Exception as e2:
                if self._is_not_modified_error(e2):
                    self.logger.debug("Final stream plain edit already applied for {}", chat_id)
                else:
                    self.logger.warning("Final stream edit failed: {}", e2)
                    raise  # Let ChannelManager handle retry
        for extra_html_chunk in extra_html_chunks:
            try:
                await self._call_with_retry(
                    self._app.bot.send_message,
                    chat_id=int_chat_id, text=extra_html_chunk,
                    parse_mode="HTML",
                    **thread_kwargs,
                )
            except Exception:
                # Fall back to _send_text which handles HTML→plain gracefully.
                await self._send_text(int_chat_id, extra_html_chunk)
        self._stream_bufs.pop(chat_id, None)

    async def send_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
        stream_end: bool = False,
        resuming: bool = False,
    ) -> None:
        """Progressive message editing: send on first delta, edit on subsequent ones."""
        if not self._app:
            return
        meta = metadata or {}
        int_chat_id = int(chat_id)

        key = self._stream_buf_key(chat_id, stream_id)
        self._sweep_stream_buffers()

        if stream_end:
            buf = self._stream_bufs.get(key)
            if not buf or not buf.text:
                return
            if buf.message_id is None and buf.draft_id is None:
                return
            if stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id:
                return
            self._stop_typing(chat_id)
            if reply_to_message_id := meta.get("message_id"):
                with suppress(ValueError):
                    await self._remove_reaction(chat_id, int(reply_to_message_id))
            thread_kwargs = {}
            if message_thread_id := meta.get("message_thread_id"):
                thread_kwargs["message_thread_id"] = message_thread_id
            staging_reply_keyboard = self._pending_stream_reply_keyboard.pop(chat_id, None)
            staging_menu_commands = self._pending_stream_menu_commands.pop(chat_id, None)
            reply_keyboard_markup = (
                self._build_reply_keyboard(staging_reply_keyboard)
                if staging_reply_keyboard else None
            )
            await self._finalize_stream(
                chat_id, buf, int_chat_id, meta, thread_kwargs,
                reply_keyboard_markup, staging_menu_commands,
            )
            return

        buf = self._stream_bufs.get(key)
        if buf is None or (stream_id is not None and buf.stream_id is not None and buf.stream_id != stream_id):
            buf = self._reset_stream_buffer(chat_id, stream_id, reason="new text stream")
        elif buf.stream_id is None:
            buf.stream_id = stream_id
        buf.text += delta

        if not buf.text.strip():
            return

        now = time.monotonic()
        buf.last_activity = now
        thread_kwargs = {}
        if message_thread_id := meta.get("message_thread_id"):
            thread_kwargs["message_thread_id"] = message_thread_id

        # Draft rich activo → actualizar el draft con thinking + contenido.
        if buf.using_draft and buf.draft_id is not None:
            if now > buf.draft_expires_at:
                # Draft expirado (~30 s sin deltas): se autolimpia; switch a
                # legacy con el contenido acumulado (nunca se pierde texto).
                buf.using_draft = False
                buf.draft_id = None
                await self._send_legacy_preview(int_chat_id, buf, meta, thread_kwargs)
                return
            if (now - buf.last_edit) >= self.config.stream_edit_interval:
                markdown = f"<tg-thinking>{buf.reasoning}</tg-thinking>\n\n{buf.text}"
                payload: dict[str, Any] = {
                    "chat_id": int_chat_id,
                    "draft_id": buf.draft_id,
                    "rich_message": {"markdown": markdown},
                }
                try:
                    await self._call_with_retry(
                        self._app.bot.do_api_request,
                        "sendRichMessageDraft",
                        api_kwargs=payload,
                    )
                    buf.last_edit = now
                except BadRequest as exc:
                    if self._is_rich_capability_error(exc):
                        self.logger.debug("sendRichMessageDraft not available, disabling")
                        self._rich_send_disabled = True
                    buf.using_draft = False
                    buf.draft_id = None
                    await self._send_legacy_preview(int_chat_id, buf, meta, thread_kwargs)
                except Exception as exc:
                    self.logger.debug("sendRichMessageDraft failed: {}", exc)
            return

        if buf.message_id is None:
            await self._send_legacy_preview(int_chat_id, buf, meta, thread_kwargs)
        elif (now - buf.last_edit) >= self.config.stream_edit_interval:
            if len(buf.text) > TELEGRAM_MAX_MESSAGE_LEN:
                await self._flush_stream_overflow(int_chat_id, buf, thread_kwargs)
                buf.last_edit = now
                return
            preview = _strip_md_block(buf.text)
            try:
                await self._call_with_retry(
                    self._app.bot.edit_message_text,
                    chat_id=int_chat_id, message_id=buf.message_id,
                    text=preview,
                )
                buf.last_edit = now
            except Exception as e:
                if self._is_not_modified_error(e):
                    buf.last_edit = now
                    return
                self.logger.warning("Stream edit failed: {}", e)
                raise  # Let ChannelManager handle retry

    async def _flush_stream_overflow(
        self,
        chat_id: int,
        buf: "_StreamBuf",
        thread_kwargs: dict,
    ) -> None:
        """Split an oversized stream buffer mid-flight.

        Edits the current stream message with the first chunk, sends any
        intermediate chunks as standalone messages, then opens a new message
        for the tail so subsequent deltas continue streaming into it.
        """
        chunks = _split_telegram_markdown_html_chunks(buf.text, TELEGRAM_HTML_MAX_LEN)
        if len(chunks) <= 1:
            return
        first_markdown, first_html = chunks[0]
        try:
            await self._call_with_retry(
                self._app.bot.edit_message_text,
                chat_id=chat_id, message_id=buf.message_id,
                text=first_html,
                parse_mode="HTML",
            )
        except BadRequest as e:
            if not self._is_not_modified_error(e):
                self.logger.warning(
                    "Stream overflow HTML edit failed, falling back to plain text: {}", e
                )
                try:
                    await self._call_with_retry(
                        self._app.bot.edit_message_text,
                        chat_id=chat_id, message_id=buf.message_id,
                        text=first_markdown,
                    )
                except Exception as plain_error:
                    if not self._is_not_modified_error(plain_error):
                        self.logger.warning("Stream overflow plain edit failed: {}", plain_error)
                        raise
        except Exception as e:
            self.logger.warning("Stream overflow edit failed: {}", e)
            raise

        async def send_chunk(markdown: str, html: str) -> Any:
            try:
                return await self._call_with_retry(
                    self._app.bot.send_message,
                    chat_id=chat_id, text=html, parse_mode="HTML", **thread_kwargs,
                )
            except BadRequest as e:
                self.logger.warning(
                    "Stream overflow HTML send failed, falling back to plain text: {}", e
                )
                return await self._call_with_retry(
                    self._app.bot.send_message,
                    chat_id=chat_id, text=markdown, **thread_kwargs,
                )

        for markdown, html in chunks[1:-1]:
            await send_chunk(markdown, html)
        markdown_tail, tail_html = chunks[-1]
        sent = await send_chunk(markdown_tail, tail_html)
        buf.message_id = sent.message_id
        buf.text = markdown_tail

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return

        user = update.effective_user
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            await self._send_pairing_code_if_private(sender_id, update.message, user)
            return
        # /start reinicia la conversación: se re-ancla el chat para que el
        # siguiente mensaje de salida vuelva a llevar quote (reply_parameters).
        self._reply_anchored.discard(update.message.chat_id)
        await update.message.reply_text(
            f"👋 Hi {user.first_name}! I'm nanobot.\n\n"
            "Send me a message and I'll respond!\n"
            "Type /help to see available commands."
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command for allowed users only."""
        if not update.message or not update.effective_user:
            return
        user = update.effective_user
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            await self._send_pairing_code_if_private(sender_id, update.message, user)
            return
        await update.message.reply_text(build_help_text())

    @staticmethod
    def _sender_id(user) -> str:
        """Build sender_id with username for allowlist matching."""
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    async def _send_pairing_code_if_private(self, sender_id: str, message, user) -> None:
        if message.chat.type != "private":
            return
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(message.chat_id),
            content="",
            metadata=self._build_message_metadata(message, user),
            is_dm=True,
        )

    @staticmethod
    def _derive_topic_session_key(message) -> str | None:
        """Derive topic-scoped session key for Telegram chats with threads."""
        message_thread_id = getattr(message, "message_thread_id", None)
        if message_thread_id is None:
            return None
        return f"telegram:{message.chat_id}:topic:{message_thread_id}"

    @staticmethod
    def _build_message_metadata(message, user) -> dict:
        """Build common Telegram inbound metadata payload."""
        reply_to = getattr(message, "reply_to_message", None)
        return {
            "message_id": message.message_id,
            "user_id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "is_group": message.chat.type != "private",
            "message_thread_id": getattr(message, "message_thread_id", None),
            "is_forum": bool(getattr(message.chat, "is_forum", False)),
            "reply_to_message_id": getattr(reply_to, "message_id", None) if reply_to else None,
        }

    async def _extract_reply_context(self, message) -> str | None:
        """Extract text from the message being replied to, if any."""
        reply = getattr(message, "reply_to_message", None)
        if not reply:
            return None
        text = getattr(reply, "text", None) or getattr(reply, "caption", None) or ""
        if len(text) > TELEGRAM_REPLY_CONTEXT_MAX_LEN:
            text = text[:TELEGRAM_REPLY_CONTEXT_MAX_LEN] + "..."

        if not text:
            return None

        bot_id, _ = await self._ensure_bot_identity()
        reply_user = getattr(reply, "from_user", None)

        if bot_id and reply_user and getattr(reply_user, "id", None) == bot_id:
            return f"[Reply to bot: {text}]"
        elif reply_user and getattr(reply_user, "username", None):
            return f"[Reply to @{reply_user.username}: {text}]"
        elif reply_user and getattr(reply_user, "first_name", None):
            return f"[Reply to {reply_user.first_name}: {text}]"
        else:
            return f"[Reply to: {text}]"

    async def _download_message_media(
        self, msg, *, add_failure_content: bool = False
    ) -> tuple[list[str], list[str]]:
        """Download media from a message (current or reply). Returns (media_paths, content_parts)."""
        media_file = None
        media_type = None
        if getattr(msg, "photo", None):
            media_file = msg.photo[-1]
            media_type = "image"
        elif getattr(msg, "voice", None):
            media_file = msg.voice
            media_type = "voice"
        elif getattr(msg, "audio", None):
            media_file = msg.audio
            media_type = "audio"
        elif getattr(msg, "document", None):
            media_file = msg.document
            media_type = "file"
        elif getattr(msg, "video", None):
            media_file = msg.video
            media_type = "video"
        elif getattr(msg, "video_note", None):
            media_file = msg.video_note
            media_type = "video"
        elif getattr(msg, "animation", None):
            media_file = msg.animation
            media_type = "animation"
        if not media_file or not self._app:
            return [], []
        try:
            file = await self._app.bot.get_file(media_file.file_id)
            ext = self._get_extension(
                media_type,
                getattr(media_file, "mime_type", None),
                getattr(media_file, "file_name", None),
            )
            media_dir = get_media_dir("telegram")
            unique_id = getattr(media_file, "file_unique_id", media_file.file_id)
            file_path = media_dir / f"{unique_id}{ext}"
            await file.download_to_drive(str(file_path))
            path_str = str(file_path)
            if media_type in ("voice", "audio"):
                transcription = await self.transcribe_audio(file_path)
                if transcription:
                    self.logger.info("Transcribed {}: {}...", media_type, transcription[:50])
                    return [path_str], [f"[transcription: {transcription}]"]
                return [path_str], [f"[{media_type}: {path_str}]"]
            return [path_str], [f"[{media_type}: {path_str}]"]
        except Exception as e:
            self.logger.warning("Failed to download message media: {}", e)
            if add_failure_content:
                return [], [f"[{media_type}: download failed]"]
            return [], []

    async def _ensure_bot_identity(self) -> tuple[int | None, str | None]:
        """Load bot identity once and reuse it for mention/reply checks."""
        if self._bot_user_id is not None or self._bot_username is not None:
            return self._bot_user_id, self._bot_username
        if not self._app:
            return None, None
        bot_info = await self._app.bot.get_me()
        self._bot_user_id = getattr(bot_info, "id", None)
        self._bot_username = getattr(bot_info, "username", None)
        return self._bot_user_id, self._bot_username

    @staticmethod
    def _has_mention_entity(
        text: str,
        entities,
        bot_username: str,
        bot_id: int | None,
    ) -> bool:
        """Check Telegram mention entities against the bot username."""
        handle = f"@{bot_username}".lower()
        for entity in entities or []:
            entity_type = getattr(entity, "type", None)
            if entity_type == "text_mention":
                user = getattr(entity, "user", None)
                if user is not None and bot_id is not None and getattr(user, "id", None) == bot_id:
                    return True
                continue
            if entity_type != "mention":
                continue
            offset = getattr(entity, "offset", None)
            length = getattr(entity, "length", None)
            if offset is None or length is None:
                continue
            if text[offset : offset + length].lower() == handle:
                return True
        return handle in text.lower()

    async def _is_group_message_for_bot(self, message) -> bool:
        """Allow group messages when policy is open, @mentioned, or replying to the bot."""
        if message.chat.type == "private" or self.config.group_policy == "open":
            return True

        bot_id, bot_username = await self._ensure_bot_identity()
        if bot_username:
            text = message.text or ""
            caption = message.caption or ""
            if self._has_mention_entity(
                text,
                getattr(message, "entities", None),
                bot_username,
                bot_id,
            ):
                return True
            if self._has_mention_entity(
                caption,
                getattr(message, "caption_entities", None),
                bot_username,
                bot_id,
            ):
                return True

        reply_user = getattr(getattr(message, "reply_to_message", None), "from_user", None)
        return bool(bot_id and reply_user and reply_user.id == bot_id)

    def _remember_thread_context(self, message) -> None:
        """Cache Telegram thread context by chat/message id for follow-up replies."""
        message_thread_id = getattr(message, "message_thread_id", None)
        if message_thread_id is None:
            return
        key = (str(message.chat_id), message.message_id)
        self._message_threads[key] = message_thread_id
        if len(self._message_threads) > 1000:
            self._message_threads.pop(next(iter(self._message_threads)))

    @staticmethod
    def _queue_key_for_message(message) -> str:
        """Return the final nanobot session key used for ordered Telegram ingress."""
        return TelegramChannel._derive_topic_session_key(message) or f"telegram:{message.chat_id}"

    @staticmethod
    def _sort_key_for_update(update: Update) -> tuple[int, int]:
        """Sort by chat message id first, then Telegram update id."""
        message = getattr(update, "message", None)
        message_id = int(getattr(message, "message_id", 0) or 0)
        update_id = int(getattr(update, "update_id", 0) or 0)
        return (message_id, update_id)

    def _enqueue_ordered_update(
        self,
        *,
        kind: Literal["command", "message"],
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Stage a Telegram update behind a short per-session reorder window."""
        message = update.message
        key = self._queue_key_for_message(message)
        self._inbound_buffers.setdefault(key, []).append(
            _QueuedTelegramUpdate(
                kind=kind,
                update=update,
                context=context,
                sort_key=self._sort_key_for_update(update),
            )
        )
        if key not in self._inbound_workers:
            self._inbound_workers[key] = asyncio.create_task(
                self._drain_ordered_updates(key)
            )

    async def _drain_ordered_updates(self, key: str) -> None:
        """Drain one Telegram session buffer in stable message order."""
        try:
            while self._running:
                await asyncio.sleep(0.2)
                batch = self._inbound_buffers.get(key, [])
                if not batch:
                    break
                self._inbound_buffers[key] = []
                batch.sort(key=lambda item: item.sort_key)
                for item in batch:
                    try:
                        if item.kind == "command":
                            await self._process_forward_command(item.update, item.context)
                        else:
                            await self._process_message_update(item.update, item.context)
                    except Exception as e:
                        self.logger.warning(
                            "Telegram queued update handling failed for {}: {}",
                            key,
                            e,
                        )
            if not self._inbound_buffers.get(key):
                self._inbound_buffers.pop(key, None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning("Telegram ordered update worker failed for {}: {}", key, e)
        finally:
            if not self._inbound_buffers.get(key):
                self._inbound_workers.pop(key, None)

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward slash commands to the bus for unified handling in AgentLoop."""
        if not update.message or not update.effective_user:
            return
        if not self._running:
            await self._process_forward_command(update, context)
            return
        self._enqueue_ordered_update(kind="command", update=update, context=context)

    async def _process_forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process a queued slash command."""
        message = update.message
        user = update.effective_user
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            await self._send_pairing_code_if_private(sender_id, message, user)
            return
        self._remember_thread_context(message)

        # Strip @bot_username suffix if present
        content = message.text or ""
        if content.startswith("/") and "@" in content:
            cmd_part, *rest = content.split(" ", 1)
            cmd_part = cmd_part.split("@")[0]
            content = f"{cmd_part} {rest[0]}" if rest else cmd_part
        content = self._normalize_telegram_command(content)

        # /new reinicia la conversación: se re-ancla el chat para que el
        # siguiente mensaje de salida vuelva a llevar quote (reply_parameters).
        if content == "/new":
            self._reply_anchored.discard(message.chat_id)

        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(message.chat_id),
            content=content,
            metadata=self._build_message_metadata(message, user),
            session_key=self._derive_topic_session_key(message),
            is_dm=message.chat.type == "private",
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return
        if not self._running:
            await self._process_message_update(update, context)
            return
        self._enqueue_ordered_update(kind="message", update=update, context=context)

    async def _process_message_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process a queued Telegram message update."""

        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            await self._send_pairing_code_if_private(sender_id, message, user)
            return
        self._remember_thread_context(message)

        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id

        if not await self._is_group_message_for_bot(message):
            return

        # Build content from text and/or media
        content_parts = []
        media_paths = []

        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)

        # Location content
        if message.location:
            lat = message.location.latitude
            lon = message.location.longitude
            content_parts.append(f"[location: {lat}, {lon}]")

        # Download current message media
        current_media_paths, current_media_parts = await self._download_message_media(
            message, add_failure_content=True
        )
        media_paths.extend(current_media_paths)
        content_parts.extend(current_media_parts)
        if current_media_paths:
            self.logger.debug("Downloaded message media to {}", current_media_paths[0])

        # Reply context: text and/or media from the replied-to message
        reply = getattr(message, "reply_to_message", None)
        if reply is not None:
            reply_ctx = await self._extract_reply_context(message)
            reply_media, reply_media_parts = await self._download_message_media(reply)
            if reply_media:
                media_paths = reply_media + media_paths
                self.logger.debug("Attached replied-to media: {}", reply_media[0])
            tag = reply_ctx or (f"[Reply to: {reply_media_parts[0]}]" if reply_media_parts else None)
            if tag:
                content_parts.insert(0, tag)
        content = "\n".join(content_parts) if content_parts else "[empty message]"

        self.logger.debug("message from {}: {}...", sender_id, content[:50])

        str_chat_id = str(chat_id)
        metadata = self._build_message_metadata(message, user)
        session_key = self._derive_topic_session_key(message)

        # Telegram media groups: buffer briefly, forward as one aggregated turn.
        if media_group_id := getattr(message, "media_group_id", None):
            key = f"{str_chat_id}:{media_group_id}"
            if key not in self._media_group_buffers:
                self._media_group_buffers[key] = {
                    "sender_id": sender_id, "chat_id": str_chat_id,
                    "contents": [], "media": [],
                    "metadata": metadata,
                    "session_key": session_key,
                }
                self._start_typing(str_chat_id)
                await self._add_reaction(str_chat_id, message.message_id, self.config.react_emoji)
            buf = self._media_group_buffers[key]
            if content and content != "[empty message]":
                buf["contents"].append(content)
            buf["media"].extend(media_paths)
            if key not in self._media_group_tasks:
                self._media_group_tasks[key] = asyncio.create_task(self._flush_media_group(key))
            return

        # Start typing indicator before processing
        self._start_typing(str_chat_id)
        await self._add_reaction(str_chat_id, message.message_id, self.config.react_emoji)

        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str_chat_id,
            content=content,
            media=media_paths,
            metadata=metadata,
            session_key=session_key,
        )

    async def _flush_media_group(self, key: str) -> None:
        """Wait briefly, then forward buffered media-group as one turn."""
        try:
            await asyncio.sleep(0.6)
            if not (buf := self._media_group_buffers.pop(key, None)):
                return
            content = "\n".join(buf["contents"]) or "[empty message]"
            await self._handle_message(
                sender_id=buf["sender_id"], chat_id=buf["chat_id"],
                content=content, media=list(dict.fromkeys(buf["media"])),
                metadata=buf["metadata"],
                session_key=buf.get("session_key"),
            )
        finally:
            self._media_group_tasks.pop(key, None)

    async def _add_reaction(self, chat_id: str, message_id: int, emoji: str) -> None:
        """Add emoji reaction to a message (best-effort, non-blocking)."""
        if not self._app or not emoji:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as e:
            self.logger.debug("reaction failed: {}", e)

    async def _remove_reaction(self, chat_id: str, message_id: int) -> None:
        """Remove emoji reaction from a message (best-effort, non-blocking)."""
        if not self._app:
            return
        try:
            await self._app.bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=message_id,
                reaction=[],
            )
        except Exception as e:
            self.logger.debug("reaction removal failed: {}", e)

    def _start_typing(self, chat_id: str) -> None:
        """Start sending 'typing...' indicator for a chat."""
        if not self._app:
            return
        self._typing.start(chat_id, self._send_typing_action(chat_id))

    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        self._typing.stop(chat_id)

    def _send_typing_action(self, chat_id: str) -> Callable[[], Awaitable[object]]:
        async def _action() -> object:
            if self._app is None:
                raise RuntimeError("telegram application not ready")
            return await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")

        return _action

    @staticmethod
    def _format_telegram_error(exc: Exception) -> str:
        """Return a short, readable error summary for logs."""
        text = str(exc).strip()
        if text:
            return text
        if exc.__cause__ is not None:
            cause = exc.__cause__
            cause_text = str(cause).strip()
            if cause_text:
                return f"{exc.__class__.__name__} ({cause_text})"
            return f"{exc.__class__.__name__} ({cause.__class__.__name__})"
        return exc.__class__.__name__

    def _on_polling_error(self, exc: Exception) -> None:
        """Keep long-polling network failures to a single readable line."""
        summary = self._format_telegram_error(exc)
        if isinstance(exc, (NetworkError, TimedOut)):
            self.logger.warning("polling network issue: {}", summary)
        else:
            self.logger.error("polling error: {}", summary)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        summary = self._format_telegram_error(context.error)

        if isinstance(context.error, (NetworkError, TimedOut)):
            self.logger.warning("network issue: {}", summary)
        else:
            self.logger.error("error: {}", summary)

    def _get_extension(
        self,
        media_type: str,
        mime_type: str | None,
        filename: str | None = None,
    ) -> str:
        """Get file extension based on media type or original filename."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                "image/webp": ".webp",
                "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
                "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
                "video/x-matroska": ".mkv", "video/3gpp": ".3gp",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]

        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "video": ".mp4", "file": ""}
        if ext := type_map.get(media_type, ""):
            return ext

        if filename:
            return "".join(Path(filename).suffixes)

        return ""

    def _build_keyboard(self, buttons: list) -> InlineKeyboardMarkup | None:
        """Build inline keyboard markup if inline_keyboards is enabled."""
        if not buttons or not self.config.inline_keyboards:
            return None
        keyboard = [
            [InlineKeyboardButton(label, callback_data=self._safe_callback_data(label)) for label in row]
            for row in buttons
        ]
        return InlineKeyboardMarkup(keyboard)

    @staticmethod
    def _build_reply_keyboard(rows: list[list[str]]) -> ReplyKeyboardMarkup:
        """Build a reply keyboard (replaces the user's keyboard) with options."""
        keyboard = [[KeyboardButton(label) for label in row] for row in rows if row]
        return ReplyKeyboardMarkup(
            keyboard,
            one_time_keyboard=True,
            input_field_placeholder="Elige una opción…",
            resize_keyboard=True,
        )

    @staticmethod
    def _build_reply_keyboard_remove() -> ReplyKeyboardRemove:
        """Build a ReplyKeyboardRemove to dismiss a previously shown keyboard."""
        return ReplyKeyboardRemove()

    async def _set_chat_menu_commands(self, chat_id: int, commands: list[dict]) -> None:
        """Register per-chat dynamic commands (setMyCommands with chat scope).

        Best-effort: a failure only logs at debug level and never breaks the
        message send.
        """
        if not self._app:
            return
        try:
            bot_commands = [
                BotCommand(command=str(c.get("command", "")), description=str(c.get("description", "")))
                for c in commands
                if c.get("command")
            ]
            await self._call_with_retry(
                self._app.bot.set_my_commands,
                commands=bot_commands,
                scope={"type": "chat", "chat_id": chat_id},
            )
        except Exception as e:
            self.logger.debug("setMyCommands (chat scope) failed: {}", e)

    @staticmethod
    def _safe_callback_data(label: str) -> str:
        # Telegram caps callback_data at 64 bytes UTF-8; truncate at a char boundary so the keyboard still sends.
        encoded = label.encode("utf-8")
        if len(encoded) <= 64:
            return label
        return encoded[:64].decode("utf-8", errors="ignore")

    @staticmethod
    def _buttons_as_text(buttons: list[list[str]]) -> str:
        # Buttons are semantic options; when we can't render a keyboard, the user still needs to see them.
        return "\n".join(" ".join(f"[{label}]" for label in row) for row in buttons if row)

    async def _on_poll_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle poll answers: publish the vote as a normal agent turn."""
        if not update.poll_answer or not update.effective_user:
            return
        poll_answer = update.poll_answer
        user = update.effective_user
        sender_id = self._sender_id(user)
        if not self.is_allowed(sender_id):
            return
        poll_id = poll_answer.poll_id
        option_ids = poll_answer.option_ids or []
        cached = self._polls_cache.get(poll_id)
        if cached and option_ids:
            options = cached.get("options") or []
            chosen = [options[i] for i in option_ids if 0 <= i < len(options)]
            label = ", ".join(chosen) if chosen else poll_id
        else:
            # Sin cache (gateway reiniciado): publicar el poll_id crudo.
            label = poll_id
        chat_id = str(cached.get("chat_id")) if cached else str(user.id)
        self.logger.debug("Poll answer from {}: {} -> {}", sender_id, poll_id, label)
        self._start_typing(chat_id)
        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=f"🗳️ El usuario votó: {label}",
            metadata={
                "poll_id": poll_id,
                "option_ids": option_ids,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_poll_answer": True,
            },
        )

    async def _on_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard button clicks (callback queries)."""
        if not update.callback_query or not update.effective_user:
            return
        query = update.callback_query
        user = update.effective_user
        chat_id = query.message.chat_id if query.message else None
        sender_id = self._sender_id(user)
        if not chat_id:
            self.logger.warning("Callback query without chat_id")
            return
        if not self.is_allowed(sender_id):
            return
        button_label = query.data or ""
        await query.answer()
        if query.message:
            with suppress(Exception):
                await query.message.edit_reply_markup(reply_markup=None)
        self.logger.debug("Inline button tap from {}: {}", sender_id, button_label)
        self._start_typing(str(chat_id))
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(chat_id),
            content=button_label,
            metadata={
                "callback_query_id": query.id,
                "button_label": button_label,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_callback": True,
            },
        )
