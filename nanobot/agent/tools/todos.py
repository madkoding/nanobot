"""Todos tool for managing per-list todo items."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.schema import (
    BooleanSchema,
    NumberSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.security.workspace_access import default_workspace_scope

# TODO(architecture): webui/todos_api is business state; this agent tool depends
# on it, inverting the dependency (agent should own it, WebUI should consume it).
from nanobot.webui.todos_api import (
    create_item,
    create_todo_list,
    delete_item,
    delete_todo_list,
    fetch_todo_list,
    list_todo_lists,
    update_item,
)

_TODOS_PARAMETERS = tool_parameters_schema(
    action=StringSchema("Action", enum=["add_list", "list_lists", "add_item", "list_items", "update_item", "delete_item", "delete_list"]),
    slug=StringSchema("List slug (required for item/list operations)"),
    name=StringSchema("List display name (add_list)"),
    text=StringSchema("Todo item text (add_item)"),
    item_id=StringSchema("Item ID (update_item/delete_item)"),
    done=BooleanSchema(description="Mark completion (update_item)."),
    due_date=StringSchema("Due date YYYY-MM-DD (add/update_item)"),
    link=StringSchema("URL to associate (add/update_item)"),
    price_clp=NumberSchema("Price in CLP (add/update_item)."),
    notes=StringSchema("Notes (add/update_item)."),
    assignee=StringSchema("Assignee (add/update_item)."),
    required=["action"],
    description="Manage todo lists and items. Use list_lists to discover slugs.",
)


@tool_parameters(_TODOS_PARAMETERS)
class TodosTool(Tool):
    """Tool to manage todo lists and items."""

    def __init__(self, workspace: str, default_assignee: str = "user"):
        self._workspace = workspace
        self._default_assignee = default_assignee

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return True

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(workspace=ctx.workspace, default_assignee=getattr(ctx, "sender_id", "user"))

    @property
    def name(self) -> str:
        return "todos"

    @property
    def description(self) -> str:
        return (
            "Manage todo lists and items: add/list/update/delete. "
            "Lists are addressed by slug. Use list_lists to find slugs."
        )

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        errors = super().validate_params(params)
        action = params.get("action")
        if action == "add_list" and not str(params.get("name") or "").strip():
            errors.append("name is required when action='add_list'")
        if action in ("add_item", "list_items", "update_item", "delete_item", "delete_list"):
            if not str(params.get("slug") or "").strip():
                errors.append("slug is required for this action")
        if action == "add_item" and not str(params.get("text") or "").strip():
            errors.append("text is required when action='add_item'")
        if action in ("update_item", "delete_item") and not str(params.get("item_id") or "").strip():
            errors.append("item_id is required when action='update_item' or 'delete_item'")
        return errors

    async def execute(
        self,
        action: str,
        slug: str | None = None,
        name: str | None = None,
        text: str | None = None,
        item_id: str | None = None,
        done: bool | None = None,
        due_date: str | None = None,
        link: str | None = None,
        price_clp: int | float | None = None,
        notes: str | None = None,
        assignee: str | None = None,
        **kwargs: Any,
    ) -> str:
        scope = default_workspace_scope(self._workspace, restrict_to_workspace=True)
        if action == "add_list":
            return self._add_list(scope, name=name or "", slug=slug)
        if action == "list_lists":
            return self._list_lists(scope)
        if action == "add_item":
            return self._add_item(
                scope,
                slug=slug or "",
                text=text or "",
                due_date=due_date,
                link=link,
                price_clp=price_clp,
                notes=notes,
                assignee=assignee or self._default_assignee,
            )
        if action == "list_items":
            return self._list_items(scope, slug=slug or "")
        if action == "update_item":
            return self._update_item(
                scope,
                slug=slug or "",
                item_id=item_id or "",
                done=done,
                due_date=due_date,
                link=link,
                price_clp=price_clp,
                notes=notes,
                assignee=assignee,
            )
        if action == "delete_item":
            return self._delete_item(scope, slug=slug or "", item_id=item_id or "")
        if action == "delete_list":
            return self._delete_list(scope, slug=slug or "")
        return f"Unknown action: {action}"

    def _add_list(self, scope: Any, *, name: str, slug: str | None) -> str:
        payload = create_todo_list(name, scope, slug=slug.strip() if slug else None)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        return f"Created todo list '{payload['name']}' (slug: {payload['slug']})."

    def _list_lists(self, scope: Any) -> str:
        payload = list_todo_lists(scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        lists = payload.get("lists", [])
        if not lists:
            return "No todo lists found."
        lines = ["Todo lists:"]
        for lst in lists:
            lines.append(
                f"- {lst.get('name')} (slug: {lst.get('slug')}, "
                f"items: {lst.get('item_count', 0)}, done: {lst.get('done_count', 0)})"
            )
        return "\n".join(lines)

    def _add_item(
        self,
        scope: Any,
        *,
        slug: str,
        text: str,
        due_date: str | None,
        link: str | None,
        price_clp: int | float | None,
        notes: str | None,
        assignee: str,
    ) -> str:
        item: dict[str, Any] = {
            "text": text,
            "assignee": assignee,
        }
        if due_date is not None:
            item["due_date"] = due_date
        if link is not None:
            item["link"] = link
        if price_clp is not None:
            item["price_clp"] = price_clp
        if notes is not None:
            item["notes"] = notes
        payload = create_item(slug, item, scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        created = payload["item"]
        return f"Created todo item '{created['text']}' (id: {created['id']}) in list '{slug}'."

    def _list_items(self, scope: Any, *, slug: str) -> str:
        payload = fetch_todo_list(slug, scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        lst = payload.get("list")
        if not isinstance(lst, dict):
            return "List not found."
        items = lst.get("items", [])
        if not items:
            return f"No items in list '{lst.get('name', slug)}'."
        lines = [f"Items in '{lst.get('name', slug)}':"]
        for it in items:
            status = "✓" if it.get("done") else "○"
            due = f" due {it.get('due_date')}" if it.get("due_date") else ""
            lines.append(f"- {status} {it.get('text')}{due} (id: {it.get('id')})")
        return "\n".join(lines)

    def _update_item(
        self,
        scope: Any,
        *,
        slug: str,
        item_id: str,
        done: bool | None,
        due_date: str | None,
        link: str | None,
        price_clp: int | float | None,
        notes: str | None,
        assignee: str | None,
    ) -> str:
        changes: dict[str, Any] = {}
        if done is not None:
            changes["done"] = done
        if due_date is not None:
            changes["due_date"] = due_date
        if link is not None:
            changes["link"] = link
        if price_clp is not None:
            changes["price_clp"] = price_clp
        if notes is not None:
            changes["notes"] = notes
        if assignee is not None:
            changes["assignee"] = assignee
        if not changes:
            return "No changes provided."
        payload = update_item(slug, item_id, changes, scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        updated = payload["item"]
        return f"Updated todo item '{updated['text']}' (id: {updated['id']})."

    def _delete_item(self, scope: Any, *, slug: str, item_id: str) -> str:
        payload = delete_item(slug, item_id, scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        return f"Deleted todo item {payload.get('item_id')} from list '{slug}'."

    def _delete_list(self, scope: Any, *, slug: str) -> str:
        payload = delete_todo_list(slug, scope)
        if payload.get("error"):
            return ToolResult.error(f"Error: {payload['error']}")
        return f"Deleted todo list '{payload.get('slug')}'."
