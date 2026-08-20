"""Tests for CronTurnCoordinator."""

from __future__ import annotations

import asyncio

import pytest

from nanobot.agent.automation_turns import AutomationTurnError
from nanobot.agent.cron_turns import CronTurnCoordinator
from nanobot.bus.events import InboundMessage, OutboundMessage


def _make_coordinator(
    *,
    running: bool = True,
    dispatch_delay_s: float = 0.0,
    dispatch_result: OutboundMessage | Exception | None = None,
) -> tuple[CronTurnCoordinator, list[InboundMessage], list[InboundMessage]]:
    published: list[InboundMessage] = []
    dispatched: list[InboundMessage] = []
    coordinator: CronTurnCoordinator | None = None

    async def publish(msg: InboundMessage) -> None:
        published.append(msg)

    async def dispatch(msg: InboundMessage) -> OutboundMessage | None:
        dispatched.append(msg)
        if dispatch_delay_s:
            await asyncio.sleep(dispatch_delay_s)
        assert coordinator is not None
        if isinstance(dispatch_result, Exception):
            coordinator.complete(msg, error=dispatch_result)
            return None
        coordinator.complete(msg, response=dispatch_result)
        return dispatch_result

    coordinator = CronTurnCoordinator(
        publish_inbound=publish,
        dispatch=dispatch,
        is_running=lambda: running,
    )
    return coordinator, published, dispatched


def _cron_msg(
    *,
    run_id: str,
    job_id: str,
    session_key: str,
    defer: bool = False,
) -> InboundMessage:
    metadata: dict[str, object] = {
        "_cron_trigger": {"run_id": run_id, "job_id": job_id},
        "_cron_defer_until_session_idle": defer,
        "webui": True,
    }
    return InboundMessage(
        channel="cron",
        sender_id="cron",
        chat_id=session_key.split(":", 1)[1] if ":" in session_key else session_key,
        content="scheduled",
        metadata=metadata,
        session_key_override=session_key,
    )


@pytest.mark.asyncio
async def test_submit_dispatches_when_not_running() -> None:
    coordinator, published, dispatched = _make_coordinator(running=False)
    msg = _cron_msg(run_id="run-1", job_id="job-1", session_key="websocket:chat-1")

    result = await asyncio.wait_for(coordinator.submit(msg), timeout=1.0)
    assert result is None
    assert len(dispatched) == 1
    assert len(published) == 0


@pytest.mark.asyncio
async def test_submit_publishes_when_running() -> None:
    coordinator, published, dispatched = _make_coordinator(running=True)
    msg = _cron_msg(run_id="run-1", job_id="job-1", session_key="websocket:chat-1")

    # complete must be called from another task because submit waits on a future.
    async def complete_later() -> None:
        await asyncio.sleep(0.01)
        coordinator.complete(msg, response=OutboundMessage(channel="cron", chat_id="chat-1", content="ok"))

    task = asyncio.create_task(complete_later())
    result = await asyncio.wait_for(coordinator.submit(msg), timeout=1.0)
    await task

    assert isinstance(result, OutboundMessage)
    assert len(published) == 1
    assert len(dispatched) == 0


@pytest.mark.asyncio
async def test_duplicate_run_id_raises() -> None:
    coordinator, published, _ = _make_coordinator(running=True)
    msg = _cron_msg(run_id="run-1", job_id="job-1", session_key="websocket:chat-1")

    first = asyncio.create_task(coordinator.submit(msg))
    await asyncio.sleep(0)  # let submit register the waiter

    with pytest.raises(RuntimeError, match="already pending"):
        await coordinator.submit(msg)

    coordinator.complete(msg, response=OutboundMessage(channel="cron", chat_id="chat-1", content="ok"))
    await first
    assert len(published) == 1


@pytest.mark.asyncio
async def test_missing_run_id_raises() -> None:
    coordinator, _, _ = _make_coordinator()
    msg = InboundMessage(channel="cron", sender_id="cron", chat_id="chat-1", content="scheduled")

    with pytest.raises(ValueError, match="run_id"):
        await coordinator.submit(msg)


@pytest.mark.asyncio
async def test_defer_if_active_queues_when_session_active() -> None:
    coordinator, published, dispatched = _make_coordinator(running=True)
    msg = _cron_msg(
        run_id="run-1", job_id="job-1", session_key="websocket:chat-1", defer=True
    )

    deferred = coordinator.defer_if_active(
        msg, session_key="websocket:chat-1", active_session_keys={"websocket:chat-1"}
    )
    assert deferred is True
    assert coordinator.pending_job_ids_for_session("websocket:chat-1") == {"job-1"}
    assert len(published) == 0
    assert len(dispatched) == 0


@pytest.mark.asyncio
async def test_defer_if_active_skips_when_session_idle() -> None:
    coordinator, _, _ = _make_coordinator(running=True)
    msg = _cron_msg(
        run_id="run-1", job_id="job-1", session_key="websocket:chat-1", defer=True
    )

    deferred = coordinator.defer_if_active(
        msg, session_key="websocket:chat-1", active_session_keys=set()
    )
    assert deferred is False
    assert coordinator.pending_job_ids_for_session("websocket:chat-1") == set()


@pytest.mark.asyncio
async def test_complete_with_error_sets_exception() -> None:
    coordinator, published, _ = _make_coordinator(running=True)
    msg = _cron_msg(run_id="run-1", job_id="job-1", session_key="websocket:chat-1")

    async def complete_later() -> None:
        await asyncio.sleep(0.01)
        coordinator.complete(msg, error=RuntimeError("boom"))

    task = asyncio.create_task(complete_later())
    with pytest.raises(AutomationTurnError):
        await asyncio.wait_for(coordinator.submit(msg), timeout=1.0)
    await task
    assert len(published) == 1
