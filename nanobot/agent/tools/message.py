"""Message tool for sending messages to users."""

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.context import current_request_context
from nanobot.agent.tools.path_utils import resolve_workspace_path
from nanobot.agent.tools.schema import (
    ArraySchema,
    BooleanSchema,
    IntegerSchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.bus.events import OutboundMessage
from nanobot.config.paths import get_workspace_path
from nanobot.security.workspace_access import current_tool_workspace


@tool_parameters(
    tool_parameters_schema(
        content=StringSchema("Message content for proactive/cross-channel delivery. Not for normal replies."),
        channel=StringSchema("Target channel for cross-channel delivery. Not for current chat."),
        chat_id=StringSchema("Target chat ID. Omit for current conversation. Not for current chat."),
        media=ArraySchema(
            StringSchema(""),
            description="File paths to attach (e.g. generated image artifacts).",
        ),
        buttons=ArraySchema(
            ArraySchema(StringSchema("")),
            description="Inline keyboard buttons: list of rows, each row is list of labels.",
        ),
        rich=BooleanSchema(
            description="Force Telegram Rich Message (Bot API 10.1) for this message: headings, tables, todo-lists, details. Default: channel config.",
        ),
        reply_keyboard=ArraySchema(
            ArraySchema(StringSchema("")),
            description="Reply keyboard replacing the user's keyboard: list of rows, each row is list of labels (Telegram only).",
        ),
        menu_commands=ArraySchema(
            ObjectSchema(
                properties={
                    "command": StringSchema("Command name without slash, e.g. 'agenda'."),
                    "description": StringSchema("Short description shown in the menu."),
                },
                required=["command", "description"],
            ),
            description="Per-chat dynamic commands for the Telegram menu button (setMyCommands with chat scope).",
        ),
        ephemeral=BooleanSchema(
            description="Send as ephemeral message visible only to the target user in groups (Telegram Bot API 10.2).",
        ),
        checklist=ObjectSchema(
            properties={
                "title": StringSchema("Task list title."),
                "tasks": ArraySchema(
                    StringSchema(""),
                    description="Task descriptions (1-30).",
                    min_items=1,
                    max_items=30,
                ),
            },
            required=["title", "tasks"],
            description="Telegram task list rich (native checkboxes - [ ] / - [x]) managed by the agent.",
        ),
        checklist_update=ObjectSchema(
            properties={
                "message_id": IntegerSchema("Message id of the task list to edit in place."),
                "done": ArraySchema(
                    IntegerSchema(""),
                    description="0-based indices of completed tasks.",
                ),
            },
            required=["message_id", "done"],
            description="Update a previously sent task list in place (mark tasks done + progress summary).",
        ),
        poll=ObjectSchema(
            properties={
                "question": StringSchema("Poll question."),
                "options": ArraySchema(
                    StringSchema(""),
                    description="Poll options (2-10).",
                    min_items=2,
                    max_items=10,
                ),
            },
            required=["question", "options"],
            description="Send a native Telegram poll (visible results, single answer).",
        ),
        effect=StringSchema(
            description="Message effect (Telegram Bot API 10.2), e.g. 'confeti'. Applied to the message.",
        ),
        required=["content"],
    )
)
class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
        workspace: str | Path | None = None,
        restrict_to_workspace: bool = False,
    ):
        self._send_callback = send_callback
        self._workspace = (
            Path(workspace).expanduser() if workspace is not None else get_workspace_path()
        )
        self._restrict_to_workspace = restrict_to_workspace
        self._fallback_channel = default_channel
        self._fallback_chat_id = default_chat_id
        self._fallback_message_id = default_message_id
        self._fallback_metadata: dict[str, Any] = {}
        self._sent_in_turn_var: ContextVar[bool] = ContextVar("message_sent_in_turn", default=False)
        self._turn_delivered_media_var: ContextVar[tuple[str, ...]] = ContextVar(
            "message_turn_delivered_media",
            default=(),
        )
        self._record_channel_delivery_var: ContextVar[bool] = ContextVar(
            "message_record_channel_delivery",
            default=False,
        )
        self._suppress_delivery_var: ContextVar[bool] = ContextVar(
            "message_suppress_delivery",
            default=False,
        )

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        send_callback = ctx.bus.publish_outbound if ctx.bus else None
        return cls(
            send_callback=send_callback,
            workspace=ctx.workspace,
            restrict_to_workspace=ctx.config.restrict_to_workspace,
        )

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False
        self._turn_delivered_media_var.set(())

    def turn_delivered_media_paths(self) -> list[str]:
        """Absolute paths attached via this tool to the active chat in the current turn."""
        return list(self._turn_delivered_media_var.get())

    def set_record_channel_delivery(self, active: bool):
        """Mark tool-sent messages as proactive channel deliveries."""
        return self._record_channel_delivery_var.set(active)

    def reset_record_channel_delivery(self, token) -> None:
        """Restore previous proactive delivery recording state."""
        self._record_channel_delivery_var.reset(token)

    def set_suppress_delivery(self, active: bool):
        """Acknowledge but don't deliver tool sends (heartbeat internal check)."""
        return self._suppress_delivery_var.set(active)

    def reset_suppress_delivery(self, token) -> None:
        """Restore previous delivery-suppression state."""
        self._suppress_delivery_var.reset(token)

    @property
    def _sent_in_turn(self) -> bool:
        return self._sent_in_turn_var.get()

    @_sent_in_turn.setter
    def _sent_in_turn(self, value: bool) -> None:
        self._sent_in_turn_var.set(value)

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Send a message to a user/channel with optional file attachments. "
            "For proactive/cross-channel delivery only. Not for normal replies."
        )

    def _resolve_media(self, media: list[str]) -> list[str]:
        """Resolve local media attachments and enforce workspace restriction when enabled."""
        resolved: list[str] = []
        access = current_tool_workspace(
            self._workspace,
            restrict_to_workspace=self._restrict_to_workspace,
        )
        workspace = access.project_path or self._workspace
        for p in media:
            if p.startswith(("http://", "https://")):
                resolved.append(p)
            elif not access.restrict_to_workspace:
                path = Path(p).expanduser()
                resolved.append(p if path.is_absolute() else str(workspace / path))
            else:
                resolved.append(str(resolve_workspace_path(p, workspace, access.allowed_root)))
        return resolved

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        buttons: list[list[str]] | None = None,
        rich: bool | None = None,
        reply_keyboard: list[list[str]] | None = None,
        menu_commands: list[dict] | None = None,
        ephemeral: bool | None = None,
        checklist: dict | None = None,
        checklist_update: dict | None = None,
        poll: dict | None = None,
        effect: str | None = None,
        **kwargs: Any,
    ) -> str:
        from nanobot.utils.helpers import strip_think

        content = strip_think(content)

        if buttons is not None:
            if not isinstance(buttons, list) or any(
                not isinstance(row, list) or any(not isinstance(label, str) for label in row)
                for row in buttons
            ):
                return ToolResult.error("Error: buttons must be a list of list of strings")
        if reply_keyboard is not None:
            if not isinstance(reply_keyboard, list) or any(
                not isinstance(row, list) or any(not isinstance(label, str) for label in row)
                for row in reply_keyboard
            ):
                return ToolResult.error("Error: reply_keyboard must be a list of list of strings")
        if menu_commands is not None:
            if not isinstance(menu_commands, list) or any(
                not isinstance(cmd, dict)
                or not isinstance(cmd.get("command"), str)
                or not isinstance(cmd.get("description"), str)
                for cmd in menu_commands
            ):
                return ToolResult.error(
                    "Error: menu_commands must be a list of objects with 'command' and 'description' strings"
                )
        if checklist is not None:
            if (
                not isinstance(checklist, dict)
                or not isinstance(checklist.get("title"), str)
                or not isinstance(checklist.get("tasks"), list)
                or not (1 <= len(checklist["tasks"]) <= 30)
                or any(not isinstance(t, str) for t in checklist["tasks"])
            ):
                return ToolResult.error(
                    "Error: checklist must be an object with 'title' (str) and 'tasks' (list of 1-30 strings)"
                )
        if checklist_update is not None:
            if (
                not isinstance(checklist_update, dict)
                or not isinstance(checklist_update.get("message_id"), int)
                or not isinstance(checklist_update.get("done"), list)
                or any(not isinstance(i, int) for i in checklist_update["done"])
            ):
                return ToolResult.error(
                    "Error: checklist_update must be an object with 'message_id' (int) and 'done' (list of ints)"
                )
        if poll is not None:
            if (
                not isinstance(poll, dict)
                or not isinstance(poll.get("question"), str)
                or not isinstance(poll.get("options"), list)
                or not (2 <= len(poll["options"]) <= 10)
                or any(not isinstance(o, str) for o in poll["options"])
            ):
                return ToolResult.error(
                    "Error: poll must be an object with 'question' (str) and 'options' (list of 2-10 strings)"
                )
        request_ctx = current_request_context()
        default_channel = (
            request_ctx.channel if request_ctx is not None else self._fallback_channel
        )
        default_chat_id = (
            request_ctx.chat_id if request_ctx is not None else self._fallback_chat_id
        )
        default_message_id = (
            request_ctx.message_id
            if request_ctx is not None
            else self._fallback_message_id
        )
        default_metadata = (
            request_ctx.metadata
            if request_ctx is not None
            else self._fallback_metadata
        )
        channel = channel or default_channel
        explicit_chat_id = chat_id
        if (
            default_channel == "websocket"
            and channel == "websocket"
            and explicit_chat_id is not None
            and str(explicit_chat_id).strip() != ""
            and str(explicit_chat_id).strip() != str(default_chat_id).strip()
        ):
            return ToolResult.error(
                "Error: chat_id does not match the active WebSocket conversation. "
                "Omit chat_id (and usually channel) so delivery uses the current "
                "conversation id from context — WebSocket client_id strings "
                "(e.g. anon-…) are not chat ids."
            )
        chat_id = chat_id or default_chat_id
        # Only inherit default message_id when targeting the same channel+chat.
        # Cross-chat sends must not carry the original message_id, because
        # some channels (e.g. Feishu) use it to determine the target
        # conversation via their Reply API, which would route the message
        # to the wrong chat entirely.
        same_target = channel == default_channel and chat_id == default_chat_id
        if same_target:
            message_id = message_id or default_message_id
        else:
            message_id = None

        if not channel or not chat_id:
            return ToolResult.error("Error: No target channel/chat specified")

        if not self._send_callback:
            return ToolResult.error("Error: Message sending not configured")

        if media:
            try:
                media = self._resolve_media(media)
            except (OSError, PermissionError, ValueError) as e:
                return ToolResult.error(f"Error: media path is not allowed: {str(e)}")

        metadata = dict(default_metadata) if same_target else {}
        if message_id:
            metadata["message_id"] = message_id
        if self._record_channel_delivery_var.get() or media:
            metadata["_record_channel_delivery"] = True

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            buttons=buttons or [],
            metadata=metadata,
            rich=rich,
            reply_keyboard=reply_keyboard or [],
            menu_commands=menu_commands or [],
            ephemeral=bool(ephemeral),
            checklist=checklist,
            checklist_update=checklist_update,
            poll=poll,
            effect=effect,
        )

        if self._suppress_delivery_var.get():
            logger.debug("MessageTool: delivery suppressed during internal check")
            return f"Message acknowledged for {channel}:{chat_id} (not delivered)"

        try:
            await self._send_callback(msg)
            if channel == default_channel and chat_id == default_chat_id:
                self._sent_in_turn = True
                if media:
                    prev = self._turn_delivered_media_var.get()
                    self._turn_delivered_media_var.set(prev + tuple(str(p) for p in media))
            media_info = f" with {len(media)} attachments" if media else ""
            button_info = f" with {sum(len(row) for row in buttons)} button(s)" if buttons else ""
            return f"Message sent to {channel}:{chat_id}{media_info}{button_info}"
        except Exception as e:
            return ToolResult.error(f"Error sending message: {str(e)}")
