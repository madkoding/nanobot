"""Tests for the todos tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, patch

import pytest

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.todos import TodosTool


@pytest.fixture
def tool(tmp_path: Path) -> TodosTool:
    return TodosTool(workspace=str(tmp_path))


class FakeScope:
    pass


async def _run(tool: TodosTool, action: str, **kwargs: object) -> object:
    with patch("nanobot.agent.tools.todos.default_workspace_scope", return_value=FakeScope()):
        return await tool.execute(action, **kwargs)


def test_validate_params_requires_name_for_add_list(tool: TodosTool) -> None:
    errors = tool.validate_params({"action": "add_list"})
    assert any("name is required" in e for e in errors)


def test_validate_params_requires_slug_for_item_ops(tool: TodosTool) -> None:
    for action in ("add_item", "list_items", "update_item", "delete_item", "delete_list"):
        errors = tool.validate_params({"action": action})
        assert any("slug is required" in e for e in errors), action


def test_validate_params_requires_text_for_add_item(tool: TodosTool) -> None:
    errors = tool.validate_params({"action": "add_item", "slug": "x"})
    assert any("text is required" in e for e in errors)


def test_validate_params_requires_item_id_for_update_delete(tool: TodosTool) -> None:
    for action in ("update_item", "delete_item"):
        errors = tool.validate_params({"action": action, "slug": "x"})
        assert any("item_id is required" in e for e in errors), action


@pytest.mark.asyncio
async def test_add_list(tool: TodosTool) -> None:
    with patch("nanobot.agent.tools.todos.create_todo_list", return_value={
        "name": "Shopping", "slug": "shopping", "item_count": 0, "done_count": 0
    }) as mock_create:
        result = await _run(tool, "add_list", name="Shopping", slug="shopping")

    mock_create.assert_called_once_with("Shopping", ANY, slug="shopping")
    assert "Created todo list 'Shopping'" in result
    assert "slug: shopping" in result


@pytest.mark.asyncio
async def test_add_list_reports_error(tool: TodosTool) -> None:
    with patch("nanobot.agent.tools.todos.create_todo_list", return_value={
        "error": "slug exists"
    }):
        result = await _run(tool, "add_list", name="Shopping", slug="shopping")

    assert isinstance(result, ToolResult)
    assert "slug exists" in str(result)


@pytest.mark.asyncio
async def test_list_lists(tool: TodosTool) -> None:
    with patch("nanobot.agent.tools.todos.list_todo_lists", return_value={
        "lists": [
            {"name": "A", "slug": "a"},
            {"name": "B", "slug": "b"},
        ]
    }) as mock_list:
        result = await _run(tool, "list_lists")

    mock_list.assert_called_once()
    assert "A" in result and "B" in result


@pytest.mark.asyncio
async def test_add_item(tool: TodosTool) -> None:
    with patch("nanobot.agent.tools.todos.create_item", return_value={
        "item": {"id": "item-1", "text": "buy milk"},
        "list": {"slug": "shop", "item_count": 1, "done_count": 0},
    }) as mock_create:
        result = await _run(tool, "add_item", slug="shop", text="buy milk", assignee="alice")

    mock_create.assert_called_once()
    call_args = mock_create.call_args.args
    assert call_args[0] == "shop"
    assert call_args[1]["text"] == "buy milk"
    assert call_args[1]["assignee"] == "alice"
    assert "buy milk" in result


@pytest.mark.asyncio
async def test_list_items(tool: TodosTool) -> None:
    with patch("nanobot.agent.tools.todos.fetch_todo_list", return_value={
        "list": {
            "slug": "shop",
            "items": [
                {"id": "1", "text": "a", "done": False},
                {"id": "2", "text": "b", "done": True},
            ],
        },
        "users": {},
    }) as mock_fetch:
        result = await _run(tool, "list_items", slug="shop")

    mock_fetch.assert_called_once_with("shop", ANY)
    assert "a" in result
    assert "b" in result


@pytest.mark.asyncio
async def test_update_item(tool: TodosTool) -> None:
    with patch("nanobot.agent.tools.todos.update_item", return_value={
        "item": {"id": "1", "text": "updated", "done": True},
        "list": {"slug": "shop", "item_count": 1, "done_count": 1},
    }) as mock_update:
        result = await _run(
            tool, "update_item", slug="shop", item_id="1", done=True, notes="n"
        )

    mock_update.assert_called_once()
    call_args = mock_update.call_args.args
    assert call_args[0] == "shop"
    assert call_args[1] == "1"
    assert call_args[2]["done"] is True
    assert call_args[2]["notes"] == "n"
    assert "updated" in result


@pytest.mark.asyncio
async def test_delete_item(tool: TodosTool) -> None:
    with patch("nanobot.agent.tools.todos.delete_item", return_value={
        "ok": True, "item_id": "1", "list": {"slug": "shop", "item_count": 0, "done_count": 0}
    }) as mock_delete:
        result = await _run(tool, "delete_item", slug="shop", item_id="1")

    mock_delete.assert_called_once_with("shop", "1", ANY)
    assert "deleted" in result.lower()


@pytest.mark.asyncio
async def test_delete_list(tool: TodosTool) -> None:
    with patch("nanobot.agent.tools.todos.delete_todo_list", return_value={
        "ok": True, "slug": "shop"
    }) as mock_delete:
        result = await _run(tool, "delete_list", slug="shop")

    mock_delete.assert_called_once_with("shop", ANY)
    assert "shop" in result


@pytest.mark.asyncio
async def test_unknown_action(tool: TodosTool) -> None:
    result = await _run(tool, "nope")
    assert "Unknown action: nope" == result
