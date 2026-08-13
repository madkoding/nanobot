"""Subagent manager for background task execution."""

import asyncio
import base64
import json
import time
import uuid
import warnings
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import (
    RequestContext,
    ToolContext,
    bind_request_context,
    reset_request_context,
)
from nanobot.agent.tools.exec_session import ExecSessionManager
from nanobot.agent.tools.file_state import FileStates
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import AgentDefaults, ToolsConfig
from nanobot.providers.base import LLMProvider
from nanobot.security.workspace_access import (
    WorkspaceScope,
    bind_workspace_scope,
    reset_workspace_scope,
    workspace_sandbox_status,
)
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template


@dataclass(slots=True)
class SubagentStatus:
    """Real-time status of a running subagent."""

    task_id: str
    label: str
    task_description: str
    started_at: float          # time.monotonic()
    phase: str = "initializing"  # initializing | awaiting_tools | tools_completed | final_response | done | error
    iteration: int = 0
    tool_events: list = field(default_factory=list)   # [{name, status, detail}, ...]
    usage: dict = field(default_factory=dict)          # token usage
    stop_reason: str | None = None
    error: str | None = None
    result: str | None = None
    finished_at: float | None = None
    chat_id: str | None = None
    persisted_at: float | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize for WS / HTTP transport."""
        return {
            "task_id": self.task_id,
            "label": self.label,
            "task_description": self.task_description,
            "phase": self.phase,
            "iteration": self.iteration,
            "tool_events": list(self.tool_events),
            "usage": dict(self.usage),
            "stop_reason": self.stop_reason,
            "error": self.error,
            "result": self.result,
            "chat_id": self.chat_id,
            "persisted_at": self.persisted_at,
        }


#: How long a finished subagent keeps its status snapshot for HTTP fetch.
#: ponytail: 24h window — enough to reopen the panel after the user comes back,
#: without keeping finished snapshots forever.
SUBAGENT_STATUS_TTL_S = 86400.0


def _storage_key(key: str) -> str:
    """Collision-resistant encoding for subagent snapshot subdirectories."""
    return base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")


def _subagent_snapshot_dir(workspace: Path, session_key: str | None) -> Path:
    """Return the directory where a session's subagent snapshots live."""
    base = workspace / "subagents"
    if session_key:
        return base / _storage_key(session_key)
    return base / "_unknown_"


def _persist_subagent_status(
    workspace: Path,
    session_key: str | None,
    status: SubagentStatus,
) -> None:
    """Persist a subagent snapshot so it survives gateway restarts."""
    if status.phase not in ("done", "error"):
        # ponytail: only finished snapshots are persisted. Running subagents
        # would need task-reconstruction; that is left for a future phase.
        return
    directory = _subagent_snapshot_dir(workspace, session_key)
    directory.mkdir(parents=True, exist_ok=True)
    status.persisted_at = time.time()
    path = directory / f"{status.task_id}.json"
    try:
        path.write_text(json.dumps(status.to_payload(), ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Failed to persist subagent status for {}", status.task_id)


def _subagent_status_from_payload(payload: dict[str, Any]) -> SubagentStatus | None:
    """Reconstruct a SubagentStatus from a persisted payload."""
    try:
        return SubagentStatus(
            task_id=payload["task_id"],
            label=payload.get("label", ""),
            task_description=payload.get("task_description", ""),
            started_at=payload.get("started_at", 0.0),
            phase=payload.get("phase", "done"),
            iteration=payload.get("iteration", 0),
            tool_events=list(payload.get("tool_events", [])),
            usage=dict(payload.get("usage", {})),
            stop_reason=payload.get("stop_reason"),
            error=payload.get("error"),
            result=payload.get("result"),
            finished_at=payload.get("finished_at"),
            chat_id=payload.get("chat_id"),
        )
    except Exception:
        logger.warning("Skipping malformed subagent snapshot: {}", payload)
        return None


def _load_persisted_subagent_statuses(
    workspace: Path,
    ttl_s: float = SUBAGENT_STATUS_TTL_S,
) -> dict[str, SubagentStatus]:
    """Load non-expired finished subagent snapshots from disk.

    Removes expired snapshot files while scanning.
    """
    base = workspace / "subagents"
    if not base.exists():
        return {}
    now = time.time()
    loaded: dict[str, SubagentStatus] = {}
    for session_dir in base.iterdir():
        if not session_dir.is_dir():
            continue
        for path in session_dir.glob("*.json"):
            if path.suffixes == [".pending", ".json"]:
                # Pending records are handled by resume_pending, not here.
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read subagent snapshot {}", path)
                continue
            persisted_at = payload.get("persisted_at")
            if not isinstance(persisted_at, (int, float)) or now - persisted_at > ttl_s:
                try:
                    path.unlink()
                except Exception:
                    pass
                continue
            status = _subagent_status_from_payload(payload)
            if status is not None:
                status.persisted_at = persisted_at
                loaded[status.task_id] = status
        # Remove empty session directories to keep the workspace tidy.
        try:
            if not any(session_dir.iterdir()):
                session_dir.rmdir()
        except Exception:
            pass
    return loaded


def _pending_path(workspace: Path, session_key: str | None, task_id: str) -> Path:
    """Path for a subagent pending record."""
    return _subagent_snapshot_dir(workspace, session_key) / f"{task_id}.pending.json"


def _persist_subagent_pending(
    workspace: Path,
    task_id: str,
    task: str,
    label: str | None,
    origin_channel: str,
    origin_chat_id: str,
    session_key: str | None,
    origin_message_id: str | None,
    temperature: float | None,
    workspace_scope: WorkspaceScope | None,
    model_preset: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> None:
    """Persist a pending subagent record so it can be relaunched after restart."""
    directory = _subagent_snapshot_dir(workspace, session_key)
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "task": task,
        "label": label,
        "origin_channel": origin_channel,
        "origin_chat_id": origin_chat_id,
        "session_key": session_key,
        "origin_message_id": origin_message_id,
        "temperature": temperature,
        "workspace_scope": workspace_scope.to_dict() if workspace_scope is not None else None,
        "model_preset": model_preset,
        "persisted_at": time.time(),
    }
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
    path = _pending_path(workspace, session_key, task_id)
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("Failed to persist subagent pending record for {}", task_id)


def _delete_subagent_pending(
    workspace: Path,
    session_key: str | None,
    task_id: str,
) -> None:
    """Remove a pending subagent record once it finishes."""
    path = _pending_path(workspace, session_key, task_id)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _load_subagent_pendings(workspace: Path) -> list[dict[str, Any]]:
    """Load all pending subagent records from disk.

    Removes expired pending records while scanning.
    """
    base = workspace / "subagents"
    if not base.exists():
        return []
    now = time.time()
    loaded: list[dict[str, Any]] = []
    for session_dir in base.iterdir():
        if not session_dir.is_dir():
            continue
        for path in session_dir.glob("*.pending.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read subagent pending record {}", path)
                continue
            persisted_at = payload.get("persisted_at")
            if not isinstance(persisted_at, (int, float)) or now - persisted_at > SUBAGENT_STATUS_TTL_S:
                try:
                    path.unlink()
                except Exception:
                    pass
                continue
            loaded.append(payload)
    return loaded


SubagentEventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class _SubagentHook(AgentHook):
    """Hook for subagent execution — logs tool calls and updates status."""

    def __init__(self, task_id: str, status: SubagentStatus | None = None) -> None:
        super().__init__()
        self._task_id = task_id
        self._status = status

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for tool_call in context.tool_calls:
            args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
            logger.debug(
                "Subagent [{}] executing: {} with arguments: {}",
                self._task_id, tool_call.name, args_str,
            )

    async def after_iteration(self, context: AgentHookContext) -> None:
        if self._status is None:
            return
        self._status.iteration = context.iteration
        self._status.tool_events = list(context.tool_events)
        self._status.usage = dict(context.usage)
        if context.error:
            self._status.error = str(context.error)


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        workspace: Path | None = None,
        bus: MessageBus | None = None,
        max_tool_result_chars: int | None = None,
        model: str | None = None,
        tools_config: ToolsConfig | None = None,
        restrict_to_workspace: bool = False,
        disabled_skills: list[str] | None = None,
        max_iterations: int | None = None,
        max_concurrent_subagents: int | None = None,
        fail_on_tool_error: bool | None = None,
        llm_wall_timeout_for_session: Callable[[str | None], float | None] | None = None,
    ):
        if workspace is None:
            raise TypeError("SubagentManager.__init__() missing required argument: 'workspace'")
        if bus is None:
            raise TypeError("SubagentManager.__init__() missing required argument: 'bus'")
        if max_tool_result_chars is None:
            raise TypeError(
                "SubagentManager.__init__() missing required argument: 'max_tool_result_chars'"
            )
        if model is not None and provider is None:
            raise TypeError("SubagentManager model compatibility argument requires provider")

        defaults = AgentDefaults()
        self._compat_runtime: LLMRuntime | None = None
        if provider is not None:
            warnings.warn(
                "SubagentManager provider/model constructor arguments are deprecated; "
                "pass runtime=... to spawn() instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self._compat_runtime = LLMRuntime.capture(
                provider,
                model or provider.get_default_model(),
                context_window_tokens=defaults.context_window_tokens,
            )
        self.workspace = workspace
        self.bus = bus
        self.tools_config = tools_config or ToolsConfig()
        self.max_tool_result_chars = max_tool_result_chars
        self.restrict_to_workspace = restrict_to_workspace
        self.disabled_skills = set(disabled_skills or [])
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else defaults.max_tool_iterations
        )
        self.max_concurrent_subagents = (
            max_concurrent_subagents
            if max_concurrent_subagents is not None
            else defaults.max_concurrent_subagents
        )
        self.fail_on_tool_error = (
            fail_on_tool_error
            if fail_on_tool_error is not None
            else defaults.fail_on_tool_error
        )
        self.runner = AgentRunner()
        self._exec_session_manager = ExecSessionManager()
        self._llm_wall_timeout_for_session = llm_wall_timeout_for_session
        self._running_tasks: dict[str, asyncio.Task[str]] = {}
        self._task_statuses: dict[str, SubagentStatus] = {}
        # Restored from disk on startup so finished subagent panels still work
        # after a gateway restart. In-memory dict also keeps recent snapshots.
        self._finished_statuses: dict[str, SubagentStatus] = _load_persisted_subagent_statuses(
            self.workspace, SUBAGENT_STATUS_TTL_S,
        )
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}
        self._event_callback: SubagentEventCallback | None = None
        # Pending records loaded on startup and relaunched via resume_pending.
        self._pending_records: list[dict[str, Any]] = _load_subagent_pendings(self.workspace)
        # Optional resolver for named model presets (set by AgentLoop).
        self._runtime_resolver: Any = None

    def set_runtime_resolver(self, resolver: Any) -> None:
        """Attach the loop's model runtime resolver for named preset resolution."""
        self._runtime_resolver = resolver

    def set_event_callback(self, callback: SubagentEventCallback | None) -> None:
        """Register an async callback invoked with each status update."""
        self._event_callback = callback

    async def resume_pending(
        self,
        resolve_runtime: Callable[[str | None], Awaitable[LLMRuntime | None] | LLMRuntime | None],
    ) -> list[str]:
        """Relaunch subagents that were running when the gateway shut down.

        Returns the list of task_ids that were resumed.
        """
        if not self._pending_records:
            return []
        records = self._pending_records
        self._pending_records = []
        resumed: list[str] = []
        for record in records:
            task_id = record.get("task_id")
            if not isinstance(task_id, str):
                continue
            # If a finished snapshot already exists, the subagent completed
            # before shutdown; just clean up the pending record.
            if task_id in self._finished_statuses:
                _delete_subagent_pending(
                    self.workspace,
                    record.get("session_key"),
                    task_id,
                )
                continue
            runtime = resolve_runtime(record.get("session_key"))
            if asyncio.iscoroutine(runtime):
                runtime = await runtime
            if runtime is None:
                # Keep the record for a future resume attempt if a runtime
                # becomes available later.
                self._pending_records.append(record)
                continue
            ws_scope = None
            raw_scope = record.get("workspace_scope")
            if isinstance(raw_scope, dict):
                try:
                    ws_scope = WorkspaceScope.from_dict(raw_scope)
                except Exception:
                    logger.warning("Could not restore workspace scope for {}", task_id)
            checkpoint = record.get("checkpoint")
            initial_messages = None
            if isinstance(checkpoint, dict):
                initial_messages = checkpoint.get("messages")
            if not isinstance(initial_messages, list) or not initial_messages:
                initial_messages = None
            try:
                await self.spawn(
                    task=record.get("task", ""),
                    label=record.get("label"),
                    origin_channel=record.get("origin_channel", "cli"),
                    origin_chat_id=record.get("origin_chat_id", "direct"),
                    session_key=record.get("session_key"),
                    origin_message_id=record.get("origin_message_id"),
                    temperature=record.get("temperature"),
                    model_preset=record.get("model_preset"),
                    workspace_scope=ws_scope,
                    runtime=runtime,
                    task_id=task_id,
                    initial_messages=initial_messages,
                )
                if initial_messages is not None:
                    logger.info(
                        "Resumed subagent {} from checkpoint at iteration {}",
                        task_id,
                        checkpoint.get("iteration", 0),
                    )
                resumed.append(task_id)
            except Exception:
                logger.exception("Failed to resume subagent {}", task_id)
                self._pending_records.append(record)
        return resumed

    def get_status(self, task_id: str) -> SubagentStatus | None:
        """Return the current status snapshot for ``task_id``.

        Evicts finished snapshots older than ``SUBAGENT_STATUS_TTL_S`` so the
        HTTP fetch endpoint doesn't leak stale state forever.
        """
        status = self._task_statuses.get(task_id) or self._finished_statuses.get(task_id)
        if status is None:
            return None
        if status.phase in ("done", "error") and status.finished_at is not None:
            age = time.monotonic() - status.finished_at
            if age > SUBAGENT_STATUS_TTL_S:
                self._finished_statuses.pop(task_id, None)
                return None
        return status

    def set_provider(self, provider: LLMProvider, model: str) -> None:
        """Update the deprecated runtime source used by legacy ``spawn`` calls."""
        warnings.warn(
            "SubagentManager.set_provider() is deprecated; pass runtime=... to spawn() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        context_window_tokens = (
            self._compat_runtime.context_window_tokens
            if self._compat_runtime is not None
            else AgentDefaults().context_window_tokens
        )
        self._compat_runtime = LLMRuntime.capture(
            provider,
            model,
            context_window_tokens=context_window_tokens,
        )

    def _compat_spawn_runtime(self) -> LLMRuntime:
        runtime = self._compat_runtime
        if runtime is None:
            raise TypeError(
                "SubagentManager.spawn() missing required keyword-only argument: 'runtime'"
            )
        warnings.warn(
            "SubagentManager.spawn() without runtime is deprecated; pass runtime=... explicitly",
            DeprecationWarning,
            stacklevel=3,
        )
        return LLMRuntime.capture(
            runtime.provider,
            runtime.model,
            context_window_tokens=runtime.context_window_tokens,
        )

    def _resolve_preset_runtime(
        self,
        model_preset: str,
        base: LLMRuntime,
    ) -> LLMRuntime:
        """Resolve a named model preset into a runtime for a subagent.

        Uses the resolver attached to the loop (via ``set_runtime_resolver``)
        when available so presets resolve against the live catalog. Falls back
        to the base runtime's provider when no resolver is configured.
        """
        resolver = self._runtime_resolver
        if resolver is not None:
            try:
                return resolver.resolve_preset(model_preset)
            except Exception:
                logger.exception(
                    "Could not resolve model_preset {} for subagent; falling back to base runtime",
                    model_preset,
                )
                return base
        return base

    def _subagent_tools_config(self) -> ToolsConfig:
        """Build a ToolsConfig scoped for subagent use."""
        return ToolsConfig(
            exec=self.tools_config.exec,
            web=self.tools_config.web,
            file=self.tools_config.file,
            restrict_to_workspace=self.restrict_to_workspace,
        )

    def _build_tools(
        self,
        workspace: Path | None = None,
        tools_config: ToolsConfig | None = None,
        *,
        scope: str = "subagent",
    ) -> ToolRegistry:
        """Build an isolated subagent tool registry via ToolLoader."""
        root = self.workspace if workspace is None else workspace
        registry = ToolRegistry()
        cfg = tools_config if tools_config is not None else self._subagent_tools_config()
        ctx = ToolContext(
            config=cfg,
            workspace=str(root.resolve()),
            exec_session_manager=self._exec_session_manager,
            file_state_store=FileStates(),
            workspace_sandbox=workspace_sandbox_status(
                restrict_to_workspace=cfg.restrict_to_workspace,
                workspace=root,
            ),
        )
        ToolLoader().load(ctx, registry, scope=scope)
        return registry

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        temperature: float | None = None,
        workspace_scope: WorkspaceScope | None = None,
        *,
        runtime: LLMRuntime | None = None,
        task_id: str | None = None,
        model_preset: str | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
        tool_scope: str = "subagent",
        extra_metadata: dict[str, Any] | None = None,
        on_announce: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        if runtime is None:
            runtime = self._compat_spawn_runtime()
        if model_preset is not None:
            runtime = self._resolve_preset_runtime(model_preset, runtime)
        if temperature is not None:
            runtime = runtime.with_generation_overrides(temperature=temperature)
        task_id = task_id or str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id, "session_key": session_key}

        status = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task_description=task,
            started_at=time.monotonic(),
            chat_id=origin_chat_id,
        )
        self._task_statuses[task_id] = status
        _persist_subagent_pending(
            self.workspace,
            task_id,
            task,
            label,
            origin_channel,
            origin_chat_id,
            session_key,
            origin_message_id,
            temperature,
            workspace_scope,
            model_preset,
        )

        bg_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin,
                status,
                runtime,
                origin_message_id,
                workspace_scope,
                initial_messages=initial_messages,
                tool_scope=tool_scope,
                extra_metadata=extra_metadata,
                on_announce=on_announce,
            )
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            # Move finished snapshots to a separate dict so HTTP fetch can
            # still serve them within the TTL window — ``_task_statuses``
            # stays clean for lifecycle tests that expect immediate removal.
            finished = self._task_statuses.pop(task_id, None)
            if finished is not None and finished.phase in ("done", "error"):
                self._finished_statuses[task_id] = finished
                _persist_subagent_status(self.workspace, session_key, finished)
                _delete_subagent_pending(self.workspace, session_key, task_id)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def run_inline(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
        origin_message_id: str | None = None,
        temperature: float | None = None,
        workspace_scope: WorkspaceScope | None = None,
        *,
        runtime: LLMRuntime | None = None,
        model_preset: str | None = None,
        tool_scope: str = "subagent",
    ) -> str:
        """Run a subagent synchronously and return its result to the caller."""
        if runtime is None:
            runtime = self._compat_spawn_runtime()
        if model_preset is not None:
            runtime = self._resolve_preset_runtime(model_preset, runtime)
        if temperature is not None:
            runtime = runtime.with_generation_overrides(temperature=temperature)
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
            "session_key": session_key,
        }
        status = SubagentStatus(
            task_id=task_id,
            label=display_label,
            task_description=task,
            started_at=time.monotonic(),
            chat_id=origin_chat_id,
        )
        self._task_statuses[task_id] = status
        logger.info("Running inline subagent [{}]: {}", task_id, display_label)
        inline_task = asyncio.create_task(
            self._run_subagent(
                task_id,
                task,
                display_label,
                origin,
                status,
                runtime,
                origin_message_id,
                workspace_scope,
                announce=False,
                tool_scope=tool_scope,
            )
        )
        self._running_tasks[task_id] = inline_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)
        try:
            result = await inline_task
            if status.phase == "error" or status.stop_reason in {"error", "tool_error"}:
                return ToolResult.error(result)
            return result
        finally:
            self._running_tasks.pop(task_id, None)
            finished = self._task_statuses.pop(task_id, None)
            if finished is not None and finished.phase in ("done", "error"):
                self._finished_statuses[task_id] = finished
                _persist_subagent_status(self.workspace, session_key, finished)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        status: SubagentStatus,
        runtime: LLMRuntime,
        origin_message_id: str | None = None,
        workspace_scope: WorkspaceScope | None = None,
        *,
        announce: bool = True,
        initial_messages: list[dict[str, Any]] | None = None,
        tool_scope: str = "subagent",
        extra_metadata: dict[str, Any] | None = None,
        on_announce: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> str:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        _last_checkpoint_time = [time.monotonic()]

        async def _on_checkpoint(payload: dict) -> None:
            status.phase = payload.get("phase", status.phase)
            status.iteration = payload.get("iteration", status.iteration)
            now = time.monotonic()
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                return
            # Throttle: persist every 5 iterations or 30 seconds, whichever comes first.
            throttle = (
                status.iteration % 5 == 0
                or now - _last_checkpoint_time[0] >= 30
            )
            if not throttle:
                return
            _last_checkpoint_time[0] = now
            _persist_subagent_pending(
                self.workspace,
                task_id,
                task,
                label,
                origin["channel"],
                origin["chat_id"],
                origin.get("session_key"),
                origin_message_id,
                None,
                workspace_scope,
                runtime.model_preset,
                checkpoint={
                    "phase": status.phase,
                    "iteration": status.iteration,
                    "messages": messages,
                },
            )

        try:
            root = workspace_scope.project_path if workspace_scope is not None else self.workspace
            cfg = None
            if workspace_scope is not None:
                cfg = self._subagent_tools_config()
                cfg.restrict_to_workspace = workspace_scope.restrict_to_workspace
            # Construct from the agent workspace; the bound scope below supplies the project cwd.
            tools = self._build_tools(tools_config=cfg, scope=tool_scope)
            system_prompt = self._build_subagent_prompt(workspace=root)
            initial_system_prompt = system_prompt
            if initial_messages:
                messages = list(initial_messages)
                # Ensure the first system message uses the current subagent prompt,
                # so resumed subagents pick up any workspace/template changes.
                if messages and messages[0].get("role") == "system":
                    messages[0] = dict(messages[0])
                    messages[0]["content"] = initial_system_prompt
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task},
                ]

            sess_key = origin.get("session_key")
            llm_timeout = (
                self._llm_wall_timeout_for_session(sess_key)
                if self._llm_wall_timeout_for_session
                else None
            )
            request_token = bind_request_context(RequestContext(
                channel=origin["channel"],
                chat_id=origin["chat_id"],
                message_id=origin_message_id,
                session_key=sess_key,
                runtime=runtime,
            ))
            token = bind_workspace_scope(workspace_scope) if workspace_scope is not None else None
            try:
                result = await self.runner.run(AgentRunSpec(
                    initial_messages=messages,
                    tools=tools,
                    runtime=runtime,
                    max_iterations=self.max_iterations,
                    max_tool_result_chars=self.max_tool_result_chars,
                    hook=_SubagentHook(task_id, status),
                    max_iterations_message="Task completed but no final response was generated.",
                    finalize_on_max_iterations=False,
                    error_message=None,
                    fail_on_tool_error=self.fail_on_tool_error,
                    checkpoint_callback=_on_checkpoint,
                    session_key=sess_key,
                    workspace=root,
                    llm_timeout_s=llm_timeout,
                ))
            finally:
                if token is not None:
                    reset_workspace_scope(token)
                reset_request_context(request_token)
            status.phase = "done"
            status.stop_reason = result.stop_reason

            if result.stop_reason == "tool_error":
                status.tool_events = list(result.tool_events)
                final_result = self._format_partial_progress(result)
                final_status = "error"
            elif result.stop_reason == "error":
                final_result = result.error or "Error: subagent execution failed."
                final_status = "error"
            else:
                final_result = result.final_content or "Task completed but no final response was generated."
                final_status = "ok"
                logger.info("Subagent [{}] completed successfully", task_id)
            status.result = final_result
            status.chat_id = origin.get("chat_id")
            status.finished_at = time.monotonic()
            self._finished_statuses[task_id] = status
            _persist_subagent_status(self.workspace, origin.get("session_key"), status)
            await self._emit_event(status, "done")
            if announce:
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    final_result,
                    origin,
                    final_status,
                    origin_message_id,
                    extra_metadata=extra_metadata,
                    on_announce=on_announce,
                )
            return final_result

        except Exception as e:
            status.phase = "error"
            status.error = str(e)
            status.result = f"Error: {e}"
            status.chat_id = origin.get("chat_id")
            status.finished_at = time.monotonic()
            logger.exception("Subagent [{}] failed", task_id)
            final_result = f"Error: {e}"
            self._finished_statuses[task_id] = status
            _persist_subagent_status(self.workspace, origin.get("session_key"), status)
            await self._emit_event(status, "error")
            if announce:
                await self._announce_result(
                    task_id,
                    label,
                    task,
                    final_result,
                    origin,
                    "error",
                    origin_message_id,
                    extra_metadata=extra_metadata,
                    on_announce=on_announce,
                )
            return final_result

    async def _emit_event(self, status: SubagentStatus, event: str) -> None:
        callback = self._event_callback
        if callback is None:
            return
        payload = status.to_payload()
        payload["event"] = event
        try:
            await callback(payload)
        except Exception:
            logger.exception("Subagent event callback raised; ignoring")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
        origin_message_id: str | None = None,
        *,
        extra_metadata: dict[str, Any] | None = None,
        on_announce: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = render_template(
            "agent/subagent_announce.md",
            label=label,
            status_text=status_text,
            task=task,
            result=result,
        )

        # Inject as system message to trigger main agent.
        # Use session_key_override to align with the main agent's effective
        # session key (which accounts for unified sessions) so the result is
        # routed to the correct pending queue (mid-turn injection) instead of
        # being dispatched as a competing independent task.
        override = origin.get("session_key") or f"{origin['channel']}:{origin['chat_id']}"
        metadata: dict[str, Any] = {
            "injected_event": "subagent_result",
            "subagent_task_id": task_id,
        }
        if origin_message_id:
            metadata["origin_message_id"] = origin_message_id
        if extra_metadata:
            metadata.update(extra_metadata)
        if on_announce is not None:
            await on_announce(result, metadata)
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
            session_key_override=override,
            metadata=metadata,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    @staticmethod
    def _format_partial_progress(result) -> str:
        completed = [e for e in result.tool_events if e["status"] == "ok"]
        failure = next((e for e in reversed(result.tool_events) if e["status"] == "error"), None)
        lines: list[str] = []
        if completed:
            lines.append("Completed steps:")
            for event in completed[-3:]:
                lines.append(f"- {event['name']}: {event['detail']}")
        if failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {failure['name']}: {failure['detail']}")
        if result.error and not failure:
            if lines:
                lines.append("")
            lines.append("Failure:")
            lines.append(f"- {result.error}")
        return "\n".join(lines) or (result.error or "Error: subagent execution failed.")

    def _build_subagent_prompt(self, workspace: Path | None = None) -> str:
        """Build a focused system prompt for the subagent."""
        from nanobot.agent.skills import SkillsLoader

        agent_workspace = self.workspace.expanduser().resolve()
        project_workspace = workspace.expanduser().resolve() if workspace else agent_workspace
        skills_summary = SkillsLoader(
            self.workspace,
            disabled_skills=self.disabled_skills,
        ).build_skills_summary()
        return render_template(
            "agent/subagent_system.md",
            workspace=str(project_workspace),
            agent_workspace=str(agent_workspace),
            history_log=str(agent_workspace / "memory" / "history.jsonl"),
            skills_summary=skills_summary or "",
        )

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [self._running_tasks[tid] for tid in self._session_tasks.get(session_key, [])
                 if tid in self._running_tasks and not self._running_tasks[tid].done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._exec_session_manager.terminate_by_owner(session_key)
        return len(tasks)

    async def close(self) -> None:
        """Cancel running subagents and close their shared exec sessions."""
        tasks = [task for task in self._running_tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._exec_session_manager.close_all()

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    def get_running_count_by_session(self, session_key: str) -> int:
        """Return the number of currently running subagents for a session."""
        tids = self._session_tasks.get(session_key, set())
        return sum(
            1 for tid in tids
            if tid in self._running_tasks and not self._running_tasks[tid].done()
        )
