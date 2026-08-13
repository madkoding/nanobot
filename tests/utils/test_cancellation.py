"""Tests for async cancellation helpers."""

from __future__ import annotations

import asyncio

import pytest

from nanobot.utils.cancellation import task_is_cancelling


@pytest.mark.asyncio
async def test_task_is_cancelling_false_in_normal_task() -> None:
    assert task_is_cancelling() is False


@pytest.mark.asyncio
async def test_task_is_cancelling_true_when_cancelled() -> None:
    state: dict[str, bool] = {}

    async def inner() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            state["was_cancelling"] = task_is_cancelling()
            raise

    task = asyncio.create_task(inner())
    await asyncio.sleep(0)  # let inner enter sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert state.get("was_cancelling") is True


@pytest.mark.asyncio
async def test_task_is_cancelling_false_when_not_current_task() -> None:
    # asyncio.current_task() returns None when called outside the event loop.
    assert task_is_cancelling() is False
