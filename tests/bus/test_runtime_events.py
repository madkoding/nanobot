import asyncio

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.bus.runtime_events import (
    RuntimeEventBus,
    RuntimeEventContext,
    RuntimeEventPublisher,
    RuntimeModelChanged,
    SessionTurnStarted,
    TurnCompleted,
    TurnRunStatusChanged,
)


@pytest.mark.asyncio
async def test_runtime_event_bus_filters_by_event_type() -> None:
    bus = RuntimeEventBus()
    seen: list[str] = []

    async def handle_run_status(event: TurnRunStatusChanged) -> None:
        seen.append(event.status)

    bus.subscribe(handle_run_status, TurnRunStatusChanged)

    await bus.publish(RuntimeModelChanged(model="m", model_preset=None))
    await bus.publish(
        TurnRunStatusChanged(
            context=RuntimeEventContext(
                channel="cli",
                chat_id="direct",
                session_key="cli:direct",
            ),
            status="running",
        )
    )

    assert seen == ["running"]


@pytest.mark.asyncio
async def test_runtime_event_bus_keeps_catch_all_subscription() -> None:
    bus = RuntimeEventBus()
    seen: list[str] = []

    def handle_any(event) -> None:
        seen.append(type(event).__name__)

    bus.subscribe(handle_any)

    await bus.publish(RuntimeModelChanged(model="m", model_preset=None))

    assert seen == ["RuntimeModelChanged"]


@pytest.mark.asyncio
async def test_runtime_event_publisher_builds_context_from_inbound_message() -> None:
    bus = RuntimeEventBus()
    seen: list[object] = []
    publisher = RuntimeEventPublisher(bus)
    msg = InboundMessage(
        channel="websocket",
        sender_id="user",
        chat_id="chat-a",
        content="hello",
        metadata={"trace_id": "turn-1"},
    )

    bus.subscribe(seen.append)

    await publisher.session_turn_started(msg, "websocket:chat-a")
    await publisher.run_status_changed(
        msg,
        "websocket:chat-a",
        "running",
        started_at=12.5,
    )

    started = seen[0]
    running = seen[1]
    assert isinstance(started, SessionTurnStarted)
    assert started.context.channel == "websocket"
    assert started.context.chat_id == "chat-a"
    assert started.context.session_key == "websocket:chat-a"
    assert started.context.metadata == {"trace_id": "turn-1"}
    assert started.context.metadata is not msg.metadata
    assert isinstance(running, TurnRunStatusChanged)
    assert running.status == "running"
    assert running.started_at == 12.5


@pytest.mark.asyncio
async def test_runtime_event_publisher_consumes_turn_metadata_on_complete() -> None:
    bus = RuntimeEventBus()
    seen: list[object] = []
    publisher = RuntimeEventPublisher(bus)

    bus.subscribe(seen.append)
    publisher.record_turn_runtime("cli:direct", "runtime")
    publisher.record_turn_latency("cli:direct", 123)

    await publisher.turn_completed(
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        metadata={"source": "test"},
    )
    await publisher.turn_completed(
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        metadata=None,
    )

    first = seen[0]
    second = seen[1]
    assert isinstance(first, TurnCompleted)
    assert first.context.metadata == {"source": "test"}
    assert first.latency_ms == 123
    assert first.runtime == "runtime"
    assert isinstance(second, TurnCompleted)
    assert second.latency_ms is None
    assert second.runtime is None


@pytest.mark.asyncio
async def test_publish_nowait_tracks_and_awaits_background_tasks() -> None:
    bus = RuntimeEventBus()
    seen: list[object] = []
    slow_handled = asyncio.Event()

    async def slow_handler(event: RuntimeModelChanged) -> None:
        seen.append(event.model)
        slow_handled.set()

    bus.subscribe(slow_handler, RuntimeModelChanged)

    bus.publish_nowait(RuntimeModelChanged(model="m1", model_preset=None))
    # The task should exist until it completes.
    assert len(bus._background_tasks) >= 1

    await slow_handled.wait()
    # Allow done callbacks to run so the set is cleaned up.
    await asyncio.sleep(0)
    assert bus._background_tasks == set()


@pytest.mark.asyncio
async def test_shutdown_awaits_pending_publish_nowait_tasks() -> None:
    bus = RuntimeEventBus()
    started = asyncio.Event()
    gate = asyncio.Event()
    seen: list[object] = []

    async def gated_handler(event: RuntimeModelChanged) -> None:
        started.set()
        await gate.wait()
        seen.append(event.model)

    bus.subscribe(gated_handler, RuntimeModelChanged)

    bus.publish_nowait(RuntimeModelChanged(model="m2", model_preset=None))
    await started.wait()

    assert len(bus._background_tasks) == 1
    gate.set()
    await bus.shutdown()

    assert seen == ["m2"]
    assert bus._background_tasks == set()


@pytest.mark.asyncio
async def test_publish_nowait_drops_event_without_running_loop() -> None:
    bus = RuntimeEventBus()

    def call_sync() -> None:
        bus.publish_nowait(RuntimeModelChanged(model="m3", model_preset=None))

    # Running outside an event loop should log and return without raising.
    call_sync()
