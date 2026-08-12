"""Workflow runner: deterministic multi-step agent orchestration."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import WorkflowUpdateEvent, outbound_message_for_event

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus
    from nanobot.security.workspace_access import WorkspaceScope
    from nanobot.utils.llm_runtime import LLMRuntime

from nanobot.workflows.loader import WorkflowLoader

_RESULT_TEXT_MAX = 4000


def parse_workflow_args(raw: str | None) -> dict[str, str]:
    """Parse ``key=value key2=value2`` into a dict; values may contain spaces."""
    text = (raw or "").strip()
    if not text:
        return {}
    args: dict[str, str] = {}
    for chunk in re.split(r"\s+(?=\w+=)", text):
        key, sep, value = chunk.partition("=")
        key = key.strip()
        if sep and key:
            args[key] = value.strip()
    return args


@dataclass
class AgentResult:
    """Result of one ``ctx.agent(...)`` call."""

    text: str
    label: str = ""


class WorkflowContext:
    """``ctx`` handed to a workflow's ``run(args, ctx)``."""

    def __init__(
        self,
        runner: "WorkflowRunner",
        run_id: str,
        *,
        runtime: "LLMRuntime",
        session_key: str | None,
        channel: str,
        chat_id: str,
        workspace_scope: "WorkspaceScope | None",
    ) -> None:
        self._runner = runner
        self._run_id = run_id
        self._runtime = runtime
        self._session_key = session_key
        self._channel = channel
        self._chat_id = chat_id
        self._workspace_scope = workspace_scope
        self._phase: str | None = None
        self._phases: list[dict[str, Any]] = []

    @property
    def phases(self) -> list[dict[str, Any]]:
        return list(self._phases)

    @property
    def current_phase(self) -> str | None:
        return self._phase

    def set_phase(self, phase: str) -> None:
        """Record a phase transition for this run."""
        now = time.time()
        if self._phases:
            self._phases[-1]["ended_at"] = _iso(now)
        self._phases.append({"name": phase, "started_at": _iso(now), "ended_at": None})
        self._phase = phase
        logger.info("Workflow [%s] entering phase: %s", self._run_id, phase)
        self._runner._publish_update(
            run_id=self._run_id,
            workflow=self._runner._name_for(self._run_id),
            phase=phase,
            status="running",
        )

    async def agent(
        self,
        *,
        agent: str = "default",
        prompt: str,
        temperature: float | None = None,
    ) -> AgentResult:
        """Consult a focused subagent and return its text result."""
        text = await self._runner.subagents.run_inline(
            task=prompt,
            label=agent,
            origin_channel=self._channel,
            origin_chat_id=self._chat_id,
            session_key=self._session_key,
            temperature=temperature,
            workspace_scope=self._workspace_scope,
            runtime=self._runtime,
        )
        return AgentResult(text=str(text), label=agent)

    async def parallel(
        self,
        fns: list[Callable[[], Awaitable[AgentResult]]],
    ) -> list[AgentResult]:
        """Run zero-arg async callables concurrently; results preserve input order."""
        return list(await asyncio.gather(*[fn() for fn in fns]))

    async def pipeline(
        self,
        stages: list[Callable[[str | None], Awaitable[AgentResult]]],
        initial: str | None = None,
    ) -> AgentResult:
        """Chain stages; each stage receives the previous stage's ``.text``."""
        result: AgentResult | None = None
        for index, stage in enumerate(stages):
            prev = initial if index == 0 else (result.text if result is not None else None)
            result = await stage(prev)
        return result if result is not None else AgentResult(text="")


class WorkflowRunner:
    """Orchestrates background workflow runs for one ``AgentLoop``."""

    def __init__(
        self,
        *,
        subagents: "SubagentManager",
        bus: "MessageBus",
        loader: WorkflowLoader,
        workspace: Path,
    ) -> None:
        self.subagents = subagents
        self.bus = bus
        self.loader = loader
        self.workspace = workspace
        self._running: dict[str, asyncio.Task] = {}
        self._session_tasks: dict[str, set[str]] = {}
        self._run_names: dict[str, str] = {}

    def _name_for(self, run_id: str) -> str:
        return self._run_names.get(run_id, "?")

    async def _publish_update(
        self,
        run_id: str,
        workflow: str,
        phase: str | None = None,
        status: str = "running",
        error: str | None = None,
        result_preview: str | None = None,
    ) -> None:
        await self.bus.publish_outbound(
            outbound_message_for_event(
                channel="websocket",
                chat_id="*",
                event=WorkflowUpdateEvent(
                    run_id=run_id,
                    workflow=workflow,
                    phase=phase,
                    status=status,
                    error=error,
                    result_preview=result_preview,
                ),
            )
        )

    def list_workflows(self) -> list[dict[str, str]]:
        return self.loader.list_workflows()

    def list_workflow_names(self) -> list[str]:
        return [entry["name"] for entry in self.list_workflows()]

    def get_workflow_description(self, name: str) -> str:
        module = self.loader.load(name)
        if module is None:
            return name
        doc = (module.__doc__ or "").strip()
        return doc.splitlines()[0] if doc else name

    async def start(
        self,
        *,
        name: str,
        args: dict[str, str] | None = None,
        runtime: "LLMRuntime",
        session_key: str | None,
        channel: str,
        chat_id: str,
        workspace_scope: "WorkspaceScope | None" = None,
        origin_message_id: str | None = None,
    ) -> str:
        """Start a workflow in the background; returns its run id."""
        run_id = str(uuid.uuid4())[:8]
        task = asyncio.create_task(
            self._run(
                run_id=run_id,
                name=name,
                args=args or {},
                runtime=runtime,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                workspace_scope=workspace_scope,
                origin_message_id=origin_message_id,
            )
        )
        self._running[run_id] = task
        self._run_names[run_id] = name
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(run_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running.pop(run_id, None)
            self._run_names.pop(run_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(run_id)
                if not ids:
                    del self._session_tasks[session_key]

        task.add_done_callback(_cleanup)
        logger.info("Started workflow '%s' (run: %s)", name, run_id)
        asyncio.create_task(self._publish_update(run_id=run_id, workflow=name, status="running"))
        return run_id

    async def _run(
        self,
        *,
        run_id: str,
        name: str,
        args: dict[str, str],
        runtime: "LLMRuntime",
        session_key: str | None,
        channel: str,
        chat_id: str,
        workspace_scope: "WorkspaceScope | None",
        origin_message_id: str | None,
    ) -> None:
        started = time.time()
        ctx: WorkflowContext | None = None
        status = "failed"
        result_text = ""
        try:
            module = self.loader.load(name)
            if module is None:
                raise ValueError(f"Workflow '{name}' not found or invalid")
            ctx = WorkflowContext(
                self,
                run_id,
                runtime=runtime,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                workspace_scope=workspace_scope,
            )
            raw = await module.run(args, ctx)
            result_text = raw.text if isinstance(raw, AgentResult) else str(raw)
            status = "completed"
        except asyncio.CancelledError:
            status = "cancelled"
            result_text = "Workflow was cancelled before completion."
            raise
        except Exception as exc:
            logger.exception("Workflow '%s' (run: %s) failed", name, run_id)
            result_text = str(exc)
            status = "failed"
        finally:
            phases = ctx.phases if ctx is not None else []
            self._save_history(run_id, name, args, started, status, phases, result_text)
            asyncio.create_task(
                self._publish_update(
                    run_id=run_id,
                    workflow=name,
                    status=status,
                    error=result_text if status in ("cancelled", "failed") else None,
                    result_preview=result_text[:500] if result_text else None,
                )
            )
            await self._announce(
                run_id,
                name,
                status,
                result_text,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                origin_message_id=origin_message_id,
            )

    async def _announce(
        self,
        run_id: str,
        name: str,
        status: str,
        text: str,
        *,
        session_key: str | None,
        channel: str,
        chat_id: str,
        origin_message_id: str | None,
    ) -> None:
        status_text = (
            "completed successfully"
            if status == "completed"
            else "was cancelled"
            if status == "cancelled"
            else f"failed ({status})"
        )
        override = session_key or f"{channel}:{chat_id}"
        metadata: dict[str, Any] = {
            "injected_event": "workflow_result",
            "workflow_run_id": run_id,
        }
        if origin_message_id:
            metadata["origin_message_id"] = origin_message_id
        msg = InboundMessage(
            channel="system",
            sender_id="workflow",
            chat_id=f"{channel}:{chat_id}",
            content=f"Workflow `{name}` {status_text} (run: {run_id}).\n\n{text}",
            session_key_override=override,
            metadata=metadata,
        )
        await self.bus.publish_inbound(msg)
        logger.debug("Workflow [%s] announced result to %s:%s", run_id, channel, chat_id)

    def _save_history(
        self,
        run_id: str,
        name: str,
        args: dict[str, str],
        started: float,
        status: str,
        phases: list[dict[str, Any]],
        text: str,
    ) -> None:
        history_dir = self.workspace / "workflow_history"
        try:
            history_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        payload = {
            "run_id": run_id,
            "workflow": name,
            "args": args,
            "started_at": _iso(started),
            "ended_at": _iso(time.time()),
            "status": status,
            "phases": phases,
            "result_text": (text or "")[:_RESULT_TEXT_MAX],
        }
        tmp = history_dir / f"{run_id}.json.tmp"
        path = history_dir / f"{run_id}.json"
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            logger.warning("Failed to write workflow history for run %s", run_id)

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all running workflows for one session. Returns count cancelled."""
        run_ids = list(self._session_tasks.get(session_key, set()))
        tasks = [
            self._running[rid]
            for rid in run_ids
            if rid in self._running and not self._running[rid].done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    async def close(self) -> None:
        """Cancel and await all running workflows."""
        tasks = [task for task in self._running.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat()
